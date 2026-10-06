import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
import numpy as np
import torch
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool, Manager, TimeoutError
from network import Network, buildNetwork, DEVICE
from training import buildPopulation
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sprudelJump"))
from env import SprudelJumpEnv
try:
    from fastEnv import FastBatch     # numba-compiled copy of the env (identical results, see sprudelJump/test_fastEnv.py)
except ImportError:
    FastBatch = None


# Tensors sent to pool workers go through shared memory. The Linux default ('file_descriptor') keeps one open
# fd per tensor, which blows past the 1024-fd limit at ~200 networks ("Too many open files"). 'file_system'
# uses named shared-memory files instead. (macOS already uses this by default.)
torch.multiprocessing.set_sharing_strategy("file_system")

LONG_GAME_REPORT_FRAMES = 25_000  # a game running longer than this prints a status line every this many frames
MAX_FRAMES_PER_GAME = 100_000  # default hard cap per game, so a network that climbs forever can't stall a generation
LIVE_UPDATE_FRAMES = 5_000  # how often (in frames) a worker publishes its current game's score to liveStatus

liveStatus = None  # per-worker shared dict {pid: (score, frames)}, set in each worker by workerInit


def workerInit(sharedStatus):
    '''Runs once in every pool worker when it starts.'''
    global liveStatus
    liveStatus = sharedStatus
    torch.set_num_threads(1)


def saveNetwork(network: Network, networkName: str, path: str | None = None) -> str:
    '''
    Saves the network's layers (weights + biases) and score to
    'models/<networkName>_<timestamp>.pt' or optionally to a 
    individually chosen path and returns that.
    '''
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if path is None:
        path = f"models/{networkName}_{timestamp}.pt"
    torch.save({"layers": [(w.clone(), b.clone()) for w, b in network.layers], "score": network.score}, path)
    return path


def poolPath(path: str) -> str:
    '''models/x.pt -> models/x_pool.pt'''
    return path[:-3] + "_pool.pt"


def savePool(networks: list[Network], path: str) -> str:
    '''
    Saves a whole group of networks (the selected parents of the last generation, best first) to 'path',
    so a later run can continue with the full gene pool instead of one single network.
    Tensors are cloned: children are views into big stacked tensors and would drag those along into the file.
    '''
    torch.save({"pool": [{"layers": [(w.clone(), b.clone()) for w, b in n.layers], "score": n.score} for n in networks]}, path)
    return path


def loadPool(path: str, targetDevice: str = "cpu") -> list[Network]:
    '''Loads a group saved by 'savePool' (best first), to pass as 'startNetwork' of sprudlerTrainingLoop.'''
    data = torch.load(path, map_location=targetDevice, weights_only=True)
    return [Network(layers=d["layers"], score=d["score"], source=path) for d in data["pool"]]


def loadNetwork(path: str, targetDevice: str = "cpu") -> Network:
    '''Loads a network saved by 'saveNetwork', e.g. to continue training via 'startNetwork'.'''
    data = torch.load(path, map_location=targetDevice, weights_only=True)
    return Network(layers=data["layers"], score=data["score"], source=path)


def addInputs(network: Network, extra: int = 2) -> Network:
    '''
    Returns a copy of the network with 'extra' more inputs whose weights are all ZERO: it plays exactly like before,
    until mutations start to use the new inputs (23 -> 25: the committed target platform's relative x and y; use addOutputs for the 5 choice outputs).
    '''
    w0, b0 = network.layers[0]
    w0 = torch.cat([w0, torch.zeros(w0.shape[0], extra, dtype=w0.dtype, device=w0.device)], dim=1)
    layers = [(w0.clone(), b0.clone())] + [(w.clone(), b.clone()) for w, b in network.layers[1:]]
    return Network(layers=layers, score=network.score, source=network.source)


def addOutputs(network: Network, extra: int = 5) -> Network:
    '''Returns a copy with 'extra' more output neurons (zero weights and bias, so the old outputs are unchanged): 2 -> 7 for target mode.'''
    w, b = network.layers[-1]
    w = torch.cat([w, torch.zeros(extra, w.shape[1], dtype=w.dtype, device=w.device)], dim=0)
    b = torch.cat([b, torch.zeros(extra, dtype=b.dtype, device=b.device)])
    layers = [(x.clone(), y.clone()) for x, y in network.layers[:-1]] + [(w, b)]
    return Network(layers=layers, score=network.score, source=network.source)


def runEnv(network: Network, maxStartHeight: int | None = None):
    '''
    Runs the sprudelJump game steered by the given Neural Network.
    Loops through forwardPasses and returns them to the game as inputs as long as the player is alive.
    Returns the network with the 'score'-attribute set.
    '''
    targetDevice = network.layers[0][0].device
    env = SprudelJumpEnv()  # initializing a new game instance
    startState = env.reset(maxStartHeight)  # gameState in first iteration
    startTensor = torch.tensor(startState, device=targetDevice).unsqueeze(0)
    startAction = network.forwardPass(startTensor).squeeze(0).tolist()
    currState = env.step(startAction) # caclulating first game input
    alive = not (currState[2])
    frames = 1

    while alive:
        action = torch.tensor(currState[0], device=targetDevice).unsqueeze(0) # transforming the action into a tensor for the NN
        nextAction = network.forwardPass(action).squeeze(0).tolist() #  calculated action by the NN
        currState = env.step(nextAction) # calculating the next State based on the prev action
        alive = not (currState[2])
        frames += 1
        if liveStatus is not None and frames % LIVE_UPDATE_FRAMES == 0:
            liveStatus[os.getpid()] = (currState[1], frames)
        # status line for unusually long games, so a run that seems stuck can be told apart from one that's still climbing
        if frames % LONG_GAME_REPORT_FRAMES == 0:
            print(f"[worker {os.getpid()}] long game still running: {frames:,} frames, score {currState[1]:,.0f}", flush=True)
    if liveStatus is not None:
        liveStatus.pop(os.getpid(), None)  # game over, no longer a live run
    return currState[1]


def SprudlerSelection(networks: list[Network], ranks: torch.Tensor, keepPart: float, count: int | None = None) -> list[Network]:
    '''
    Takes the networks and one average rank per network (low = good) and keeps the best ones:
    'count' of them, or - if not given - the 'keepPart' fraction of 'networks'.
    Returns this elite selection as a _sorted_ list, best first.
    '''
    keep = count if count is not None else math.floor(len(networks) * keepPart)
    sel = ranks.topk(keep, largest=False, sorted=True)
    result = [networks[el] for el in sel.indices]
    return result


def evaluateNetwork(network: Network, seeds: list, iterations: int = 5, maxFrames: int = MAX_FRAMES_PER_GAME, maxStartHeight: int | None = None, minStartHeight: int = 0, zeroFraction: float = 0.0, steerSmoothing: float = 1.0, stableSlots: bool = False, monsterFraction: float = 0.0, monsterMult: float = 1.0, fast: bool = False, actionRepeat: int = 1):
    '''
    Plays 'iterations' games with the network side by side (lockstep) and
    returns its score, averaged over these games.
    Every frame, the states of all still-running games are stacked into one
    [numAlive, 23] batch, so the network does ONE forwardPass per frame instead
    of one per game - for a network this small, torch's per-call overhead
    dominates, so this is ~5x faster than playing the games one after another.
    Games still running after 'maxFrames' frames are stopped and count with
    the score they reached by then.
    TARGET MODE (25 inputs, 7 outputs): outputs 2..6 are the network's choice among the 5 platform slots (env.slotPlatforms:
    3 nearest below, 2 nearest above). At every bounce, and whenever its target is lost (gone, or passed), the slot with the highest
    output (among the platforms in the direction it is moving: above while rising, below while falling) becomes the committed
    target, and stays until then; the target's relative x/y are the 2 extra inputs. The env chooses nothing.
    'steerSmoothing' (0 < a <= 1, 1 = off): the steer that reaches the game is
    applied + a * (output - applied), i.e. the steering can't flip fully from one frame to the next.
    'fast': play with the numba-compiled env (fastEnv, needs numba; same rules, same levels, same scores for actionRepeat=1).
    'actionRepeat': the network decides every actionRepeat-th frame and its action is repeated in between (fast=True only).
    'monsterFraction'/'monsterMult': monster practice - that share of the games has monsterMult x the normal monster spawn chance.
    'stableSlots': observation with fixed slot meanings (3 platforms below, 2 above; see SprudelJumpEnv).
    '''
    if fast:
        return evaluateNetworkFast(network, seeds, maxFrames, maxStartHeight, minStartHeight, zeroFraction, steerSmoothing, stableSlots, monsterFraction, monsterMult, actionRepeat)
    assert actionRepeat == 1, "actionRepeat > 1 needs fast=True"
    # the network as plain numpy arrays: for batches this small torch's per-call overhead dominates.
    # same computation as Network.forwardPass (ReLU hidden layers, sigmoid output), same float32 results.
    layers = [(w.detach().cpu().numpy().astype(np.float32).T.copy(), b.detach().cpu().numpy().astype(np.float32)) for w, b in network.layers]
    # a network with 24 inputs gets the state with the extra difficulty input, one with 23 the classic state
    nIn = network.layers[0][0].shape[1]
    targetMode = nIn == 25 and layers[-1][0].shape[1] == 7
    targets = [None] * iterations    # committed target platform (the env's own platform list object) per game
    envs = [SprudelJumpEnv(difficultyInput=nIn == 24, stableSlots=stableSlots) for i in range(0, iterations)]
    states = [env.reset(maxStartHeight, seed, minStartHeight, zeroFraction, monsterFraction, monsterMult) for env, seed in zip(envs, seeds)]
    scores = [0.0] * iterations
    # indices of the games still running. Row k of the batch belongs to game alive[k] -
    # that's how each output row gets back to the right game once some games have died.
    alive = list(range(0, iterations))
    frames = 0
    applied = [0.5] * iterations                          # the steer that was actually given to the game

    while alive and frames < maxFrames:
        x = np.array([states[g] + envs[g].relativeTo(targets[g]) if targetMode else states[g] for g in alive], dtype=np.float32)  # [numAlive, 23 / 24 / 25]
        for L, (wT, b) in enumerate(layers):
            x = x @ wT + b
            if L < len(layers) - 1:
                x = np.maximum(x, 0)
            else:
                with np.errstate(over="ignore"):  # exp overflow -> inf -> sigmoid 0.0, as in torch
                    x = 1 / (1 + np.exp(-x))
        actions = x.tolist()  # [numAlive, 2], row k -> game alive[k]
        stillAlive = []
        for g, action in zip(alive, actions):
            if targetMode:
                env = envs[g]
                t = targets[g]
                feet = env.playerY + 40
                if env.bounced or t is None or not any(t is q for q in env.platforms) or (env.velY > 0 and t[1] < feet) or (env.velY < 0 and t[1] >= feet):
                    slots = env.slotPlatforms()
                    # candidates: platforms in the direction of travel (rising: the 2 above, falling: the 3 below); any existing one if none there
                    idx = [k for k in (range(3, 5) if env.velY < 0 else range(0, 3)) if slots[k] is not None] or [k for k in range(5) if slots[k] is not None]
                    targets[g] = slots[max(idx, key=lambda k: action[2 + k])] if idx else None
                action = action[:2]
            if steerSmoothing < 1.0:
                applied[g] += steerSmoothing * (action[0] - applied[g])
                action = [applied[g], action[1]]
            states[g], scores[g], done = envs[g].step(action)
            if not done:
                stillAlive.append(g)
        alive = stillAlive
        frames += 1

        if liveStatus is not None and alive and frames % LIVE_UPDATE_FRAMES == 0:
            liveStatus[os.getpid()] = (max(scores[g] for g in alive), frames)
        # status line for unusually long games, so a run that seems stuck can be told apart from one that's still climbing
        if alive and frames % LONG_GAME_REPORT_FRAMES == 0:
            print(f"[worker {os.getpid()}] long game still running: {frames:,} frames, best live score {max(scores[g] for g in alive):,.0f}", flush=True)

    if liveStatus is not None:
        liveStatus.pop(os.getpid(), None)  # all games over, no longer a live run
    return scores


def evaluateNetworkFast(network: Network, seeds: list, maxFrames: int, maxStartHeight, minStartHeight, zeroFraction, steerSmoothing, stableSlots,
                        monsterFraction, monsterMult, actionRepeat):
    '''
    Same as evaluateNetwork, but the games run in the numba-compiled env (fastEnv.FastBatch) and the network decides only every
    'actionRepeat' frames (its action is repeated in between). With actionRepeat=1 the scores are identical to evaluateNetwork's.
    Frames are counted in game frames, so maxFrames means the same as before.
    '''
    layers = [(w.detach().cpu().numpy().astype(np.float32).T.copy(), b.detach().cpu().numpy().astype(np.float32)) for w, b in network.layers]
    nIn, nOut = layers[0][0].shape[0], layers[-1][0].shape[1]
    targetMode = nIn == 25 and nOut == 7
    batch = FastBatch(seeds, maxStartHeight, minStartHeight, zeroFraction, monsterFraction, monsterMult, stable=stableSlots, difficulty=nIn == 24)
    alive = np.arange(len(seeds), dtype=np.int64)
    applied = np.full(len(seeds), 0.5)
    frames = 0
    while alive.size and frames < maxFrames:
        x = batch.observe(alive, nIn).astype(np.float32)
        for L, (wT, b) in enumerate(layers):
            x = x @ wT + b
            if L < len(layers) - 1:
                x = np.maximum(x, 0)
            else:
                with np.errstate(over="ignore"):
                    x = 1 / (1 + np.exp(-x))
        if targetMode:
            batch.decide(alive, x[:, 2:7].astype(np.float64))
        steer = x[:, 0].astype(np.float64)
        if steerSmoothing < 1.0:
            applied[alive] += steerSmoothing * (steer - applied[alive])
            steer = applied[alive]
        batch.step(alive, steer, x[:, 1].astype(np.float64), actionRepeat)
        alive = alive[~batch.done()[alive]]
        before, frames = frames, frames + actionRepeat
        if liveStatus is not None and alive.size and before // LIVE_UPDATE_FRAMES != frames // LIVE_UPDATE_FRAMES:
            liveStatus[os.getpid()] = (float(batch.scores()[alive].max()), frames)
        if alive.size and before // LONG_GAME_REPORT_FRAMES != frames // LONG_GAME_REPORT_FRAMES:
            print(f"[worker {os.getpid()}] long game still running: {frames:,} frames, best live score {batch.scores()[alive].max():,.0f}", flush=True)
    if liveStatus is not None:
        liveStatus.pop(os.getpid(), None)
    return batch.scores().tolist()


def logGeneration(logPath: str, meta: str, row: list):
    '''
    Appends one generation's statistics as a line to the run's CSV file (created with a
    '# key=value ...' metadata line + a header on the first call), for plotRun.py.
    Costs microseconds per generation - the plot runs in a separate process.
    '''
    newFile = not os.path.exists(logPath)
    with open(logPath, "a") as f:
        if newFile:
            f.write(f"# {meta}\n")
            f.write("gen,elapsed_s,mean,median,p10,p90,best,chosen\n")
        f.write(",".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in row) + "\n")


def evaluatePopulation(p, networks: list[Network], seeds: list, maxFrames: int, maxStartHeight: int | None,
                       sharedStatus, desc: str, minStartHeight: int = 0, zeroFraction: float = 0.0, steerSmoothing: float = 1.0, stableSlots: bool = False, monsterFraction: float = 0.0, monsterMult: float = 1.0, fast: bool = False, actionRepeat: int = 1) -> list[list[float]]:
    '''
    Plays every network in 'networks' on the same 'seeds' (one game per seed) using the worker pool 'p'.
    Returns one list of per-game scores per network, in the same order as 'networks'.
    '''
    evaluate = partial(evaluateNetwork, seeds=seeds, iterations=len(seeds), maxFrames=maxFrames, maxStartHeight=maxStartHeight, minStartHeight=minStartHeight, zeroFraction=zeroFraction, steerSmoothing=steerSmoothing, stableSlots=stableSlots, monsterFraction=monsterFraction, monsterMult=monsterMult, fast=fast, actionRepeat=actionRepeat)
    results = p.imap(evaluate, networks)  # yields each network's result as soon as it is done, in order
    scores = []
    with tqdm(total=len(networks), desc=desc, unit="net", leave=False) as bar:
        while len(scores) < len(networks):
            try:
                scores.append(results.next(timeout=1.0))
                bar.update(1)
            except TimeoutError:
                pass
            live = sorted(sharedStatus.values(), reverse=True)  # [(score, frames), ...] best first
            top = "  ".join(f"{sc:,.0f}@{fr // 1000}k" for sc, fr in live[:5])
            bar.set_postfix_str(f"live={len(live)}  top: {top}" if live else "")
    return scores


def averageRanks(scores: torch.Tensor) -> torch.Tensor:
    '''[networks, games] scores -> one average rank per network (0 = best), ranking all networks against each other in every game.'''
    return scores.argsort(dim=0, descending=True).argsort(dim=0).float().mean(1)


def sprudlerTrainingLoop(concInstances: int,            perNetworkIterations: int,              generations: int,                            
                         populationSize: int,           mutationRate: float,                    sigma: float, eliteCount: int,
                         keepPartSelection: float,      crossover: bool = False,                maxFramesPerGame: int = MAX_FRAMES_PER_GAME, 
                         startNetwork: Network | list[Network] | None = None,                   livePlot: bool = True,
                         inputSize: int | None = None,  hiddenSizes: list[int] | None = None,   outputSize: int | None = None,
                         networkName: str = "network",  targetDevice: str = "cpu",              maxStartHeight: int |None = None,
                         stage1Games: int | None = None,finalistFraction: float = 0.3,          maxHours: float | None = None,               
                         minStartHeight: int = 0,       zeroFraction: float = 0.0,              snapshotAt: tuple = (), steerSmoothing: float = 1.0, stableSlots: bool = False, monsterFraction: float = 0.0, monsterMult: float = 1.0, fast: bool = False, actionRepeat: int = 1) -> Network:
    '''
    creates/takes a Network instance and trains it for a certain 
    amount of times. Then it returns the trained network and writes its weights into a file.

    Evaluation: every network plays 'perNetworkIterations' games on the generation's shared seeds.
    Two-stage option ('stage1Games' set, smaller than perNetworkIterations): ALL networks play only the first
    'stage1Games' games; the best 'finalistFraction' of them (by average rank) then play the remaining games,
    and the selection is made among these finalists using all games. Networks that were clearly bad after
    stage 1 never use up the rest of the games, which saves most of the evaluation time.

    'maxHours': the run stops cleanly after the generation in which this many hours have passed (everything is saved).
    Ctrl-C does the same after the current generation was interrupted: best network + pool of the last finished generation are saved.
    '''
    start = time.perf_counter()
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"models/{networkName}_{timestamp}.pt"
    print("Start of training loop!")
    if livePlot:
        # learning curve in its own process (own window, redraws every few seconds) - training never waits for it
        subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "plotRun.py"),
                          path.replace(".pt", ".csv"), "--live", "--chain"])
    print("Generating first population...")
    # if no startNetwork is given generate a population of networks with completely random weights and biases
    if inputSize is not None and outputSize is not None and hiddenSizes is not None:
        currGen = []
        for i in range(0, populationSize):
            currGen.append(buildNetwork(inputSize, hiddenSizes, outputSize, targetDevice))

    # if startNetwork is given build a population of slight variations of itself, through mutation
    elif startNetwork is not None:
        # one network: all children are mutated copies of it. A pool (list, best first, see loadPool): children are bred
        # from the whole group, the best 'eliteCount' of it stay unchanged - so the diversity of the last run survives.
        selection = list(startNetwork) if isinstance(startNetwork, list) else [startNetwork for i in range(0, populationSize)]
        currGen = buildPopulation(populationSize, selection, sigma, mutationRate, eliteCount, crossover=crossover)
    else:
        raise ValueError("Please provide a starting network OR input- output- and hiddenSizes.")

    ## TRAININGLOOP
    # each loop runs one generation of networks, selects the best and breeds them, then the next
    # iteration does the same until the training is done.  
    evalTime = 0.0
    breedTime = 0.0
    bestEver = float("-inf")
    manager = Manager()
    sharedStatus = manager.dict()
    gensDone = 0
    try:
        with Pool(concInstances, initializer=workerInit, initargs=(sharedStatus,)) as p:
            with tqdm(total=generations, desc=networkName, unit="gen") as pbar:
                for i in range(0, generations):
                    ## EVALUATION
                    evalStart = time.perf_counter()
                    seeds = [random.randrange(2**32) for _ in range(perNetworkIterations)]  # shared by all networks this generation
                    keepCount = math.floor(populationSize * keepPartSelection)
                    twoStage = stage1Games is not None and 0 < stage1Games < perNetworkIterations

                    if not twoStage:
                        resultsTensor = torch.tensor(evaluatePopulation(p, currGen, seeds, maxFramesPerGame, maxStartHeight, sharedStatus, f"  gen {i + 1}", minStartHeight, zeroFraction, steerSmoothing, stableSlots, monsterFraction, monsterMult, fast, actionRepeat))
                        popScores = resultsTensor                           # [population, games]: everyone, all games
                        candidates, candidateScores = currGen, resultsTensor
                    else:
                        # stage 1: everyone, only the first few games
                        popScores = torch.tensor(evaluatePopulation(p, currGen, seeds[:stage1Games], maxFramesPerGame, maxStartHeight, sharedStatus, f"  gen {i + 1} stage 1", minStartHeight, zeroFraction, steerSmoothing, stableSlots, monsterFraction, monsterMult, fast, actionRepeat))
                        finalistCount = max(keepCount, math.ceil(populationSize * finalistFraction))
                        finalistIdx = averageRanks(popScores).topk(finalistCount, largest=False).indices
                        candidates = [currGen[k] for k in finalistIdx]
                        # stage 2: only the finalists play the remaining games; selection sees ALL games of each finalist
                        stage2 = torch.tensor(evaluatePopulation(p, candidates, seeds[stage1Games:], maxFramesPerGame, maxStartHeight, sharedStatus, f"  gen {i + 1} stage 2", minStartHeight, zeroFraction, steerSmoothing, stableSlots, monsterFraction, monsterMult, fast, actionRepeat))
                        candidateScores = torch.cat([popScores[finalistIdx], stage2], dim=1)   # [finalists, all games]
                    evalTime += time.perf_counter() - evalStart

                    # saving the score (mean points per game) for every network that was selection-relevant
                    for j, net in enumerate(currGen):
                        net.score = popScores[j].mean().item()               # stage-1 / all-games mean for everyone ...
                    for j, net in enumerate(candidates):
                        net.score = candidateScores[j].mean().item()         # ... overwritten by the all-games mean for the candidates

                    ## SELECTION
                    # rank the candidates against each other in every game, average the ranks, keep the best ones
                    selection = SprudlerSelection(candidates, averageRanks(candidateScores), keepPartSelection, count=keepCount)

                    ## REPRODUCTION
                    breedStart = time.perf_counter()
                    currGen = buildPopulation(populationSize, selection, sigma, mutationRate, eliteCount, crossover=crossover)
                    breedTime += time.perf_counter() - breedStart
                
                    # saving a network every 50 iterations as a 'Checkpoint' if the run fails
                    if (i + 1) % 50 == 0:
                        saveNetwork(selection[0], networkName, path)
                        savePool(selection, poolPath(path))
                    if (i + 1) in snapshotAt:     # a copy of the best network at this generation (the normal checkpoint file is overwritten)
                        saveNetwork(selection[0], networkName, path.replace(".pt", f"_g{i + 1}.pt"))

                    bestEver = max(bestEver, selection[0].score)
                    # one CSV line per generation (see plotRun.py): mean score per network -> population statistics
                    perNetwork = popScores.mean(dim=1)
                    q = torch.quantile(perNetwork, torch.tensor([0.1, 0.5, 0.9]))
                    logGeneration(path.replace(".pt", ".csv"),
                                  f"name={networkName} generations={generations} population={populationSize} games={perNetworkIterations} "
                                  f"startFrom={(startNetwork[0] if isinstance(startNetwork, list) else startNetwork).source if startNetwork is not None else None} "
                                  f"stage1Games={stage1Games if twoStage else 'None'} "
                                  f"mutationRate={mutationRate} sigma={sigma} elites={eliteCount} keepPart={keepPartSelection} minStartHeight={minStartHeight} maxStartHeight={maxStartHeight} zeroFraction={zeroFraction} steerSmoothing={steerSmoothing} stableSlots={stableSlots} monsterFraction={monsterFraction} monsterMult={monsterMult} fast={fast} actionRepeat={actionRepeat}",
                                  [i + 1, time.perf_counter() - start, perNetwork.mean().item(), q[1].item(), q[0].item(), q[2].item(),
                                   candidateScores.mean(dim=1).max().item(), selection[0].score])
                    pbar.set_postfix(best=f"{selection[0].score:.0f}", avg=f"{perNetwork.mean().item():.0f}", bestEver=f"{bestEver:.0f}")
                    pbar.update(1)
                    gensDone = i + 1
                    if maxHours is not None and time.perf_counter() - start > maxHours * 3600:
                        print(f"\nTime limit of {maxHours} h reached after {gensDone} generations.")
                        break

    except KeyboardInterrupt:
        if gensDone == 0:
            raise
        print(f"\nInterrupted after {gensDone} generations - saving the last finished generation.")

    manager.shutdown()
    savePath = saveNetwork(selection[0], networkName, path)
    savePool(selection, poolPath(path))
    csvPath = path.replace(".pt", ".csv")
    if os.path.exists(csvPath):  # learning curve (whole lineage) as a PNG next to the model
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "plotRun.py"), csvPath, "--chain",
                        "--save", path.replace(".pt", ".png")], check=False)
    totalTime = time.perf_counter() - start
    print(
        f"\n{'=' * 44}\n"
        f"{networkName:^44}\n"
        f"{'=' * 44}\n"
        f"Population size:     {populationSize}\n"
        f"Generations:         {gensDone} of {generations}\n"
        f"Games per network:   {perNetworkIterations}\n"
        f"Max frames per game: {maxFramesPerGame:,}\n"
        f"Crossover:           {crossover}\n"
        f"Worker processes:    {concInstances}\n"
        f"Avg. EvalTime:       {evalTime / gensDone:.3f} seconds\n"
        f"Avg. BreedTime:      {breedTime / gensDone:.3f} seconds\n"
        f"Total time:          {totalTime:.1f} seconds\n"
        f"Best avg. score:     {selection[0].score:.1f} (last generation)\n"
        f"Saved to:            {savePath}\n"
        f"Pool (top {len(selection)}):      {poolPath(savePath)}\n"
        f"{'=' * 44}"
    )

    return selection[0] # return the best network of the last trained generation


if __name__ == "__main__":
    #inS = 24  # inputs: 
    #hiS = [24, 24, 24]
    #outS = 2 # [steer, shoot]
    gamesPerNetwork = 200
    generations = 3000   # the time limit decides, not this
    model = loadPool("models/careful_2026-10-05_18-43-51_pool.pt")   # target mode net (25 inputs, 7 outputs), best so far
    steerSmoothing = 0.5   # 1 = off
    fast = True            # numba-compiled env (identical results, several times faster)
    actionRepeat = 1       # the network decides every n-th frame (needs fast=True; models trained with 1 lose ~20% at 2 until retrained)
    stableSlots = True     # fixed-meaning observation slots (3 platforms below, 2 above) instead of sorted by |distance|
    popSize = 200
    mutRate = 0.7   # careful settings: polish what the explore run found
    sigma = 0.02
    monsterFraction = 0.0   # monster practice (no gain in the 2x2 test): share of the games with monsterMult x the normal monster spawn chance
    monsterMult = 2.0
    eliteCount = 3
    keepPart = 0.1
    concurrent = 23
    maxStartHeight = 30000
    minStartHeight = 20000
    zeroFraction = 0.5
    maxHours = 5
    bestSprudler = sprudlerTrainingLoop(concurrent, gamesPerNetwork, generations, popSize, mutRate, sigma, eliteCount, keepPart, maxStartHeight=maxStartHeight, minStartHeight=minStartHeight, zeroFraction=zeroFraction, maxHours=maxHours, startNetwork=model, networkName="careful2", livePlot=True, stage1Games=20, finalistFraction=0.3, steerSmoothing=steerSmoothing, stableSlots=stableSlots, monsterFraction=monsterFraction, monsterMult=monsterMult, fast=fast, actionRepeat=actionRepeat)
