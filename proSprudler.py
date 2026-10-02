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


def evaluateNetwork(network: Network, seeds: list, iterations: int = 5, maxFrames: int = MAX_FRAMES_PER_GAME, maxStartHeight: int | None = None):
    '''
    Plays 'iterations' games with the network side by side (lockstep) and
    returns its score, averaged over these games.
    Every frame, the states of all still-running games are stacked into one
    [numAlive, 23] batch, so the network does ONE forwardPass per frame instead
    of one per game - for a network this small, torch's per-call overhead
    dominates, so this is ~5x faster than playing the games one after another.
    Games still running after 'maxFrames' frames are stopped and count with
    the score they reached by then.
    '''
    # the network as plain numpy arrays: for batches this small torch's per-call overhead dominates.
    # same computation as Network.forwardPass (ReLU hidden layers, sigmoid output), same float32 results.
    layers = [(w.detach().cpu().numpy().astype(np.float32).T.copy(), b.detach().cpu().numpy().astype(np.float32)) for w, b in network.layers]
    envs = [SprudelJumpEnv() for i in range(0, iterations)]
    states = [env.reset(maxStartHeight, seed) for env, seed in zip(envs, seeds)]
    scores = [0.0] * iterations
    # indices of the games still running. Row k of the batch belongs to game alive[k] -
    # that's how each output row gets back to the right game once some games have died.
    alive = list(range(0, iterations))
    frames = 0

    while alive and frames < maxFrames:
        x = np.array([states[g] for g in alive], dtype=np.float32)  # [numAlive, 23]
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
                       sharedStatus, desc: str) -> list[list[float]]:
    '''
    Plays every network in 'networks' on the same 'seeds' (one game per seed) using the worker pool 'p'.
    Returns one list of per-game scores per network, in the same order as 'networks'.
    '''
    evaluate = partial(evaluateNetwork, seeds=seeds, iterations=len(seeds), maxFrames=maxFrames, maxStartHeight=maxStartHeight)
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


def sprudlerTrainingLoop(concInstances: int, perNetworkIterations: int, generations: int, populationSize: int, mutationRate: float, sigma: float, eliteCount: int,
                keepPartSelection: float, crossover: bool = False, maxFramesPerGame: int = MAX_FRAMES_PER_GAME, startNetwork: Network | list[Network] | None = None,
                inputSize: int | None = None, hiddenSizes: list[int] | None = None, outputSize: int | None = None,
                networkName: str = "network", targetDevice: str = "cpu", maxStartHeight: int |None = None, livePlot: bool = True,
                stage1Games: int | None = None, finalistFraction: float = 0.3) -> Network:
    '''
    creates/takes a Network instance and trains it for a certain 
    amount of times. Then it returns the trained network and writes its weights into a file.

    Evaluation: every network plays 'perNetworkIterations' games on the generation's shared seeds.
    Two-stage option ('stage1Games' set, smaller than perNetworkIterations): ALL networks play only the first
    'stage1Games' games; the best 'finalistFraction' of them (by average rank) then play the remaining games,
    and the selection is made among these finalists using all games. Networks that were clearly bad after
    stage 1 never use up the rest of the games, which saves most of the evaluation time.
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
    with Pool(concInstances, initializer=workerInit, initargs=(sharedStatus,)) as p:
        with tqdm(total=generations, desc=networkName, unit="gen") as pbar:
            for i in range(0, generations):
                ## EVALUATION
                evalStart = time.perf_counter()
                seeds = [random.randrange(2**32) for _ in range(perNetworkIterations)]  # shared by all networks this generation
                keepCount = math.floor(populationSize * keepPartSelection)
                twoStage = stage1Games is not None and 0 < stage1Games < perNetworkIterations

                if not twoStage:
                    resultsTensor = torch.tensor(evaluatePopulation(p, currGen, seeds, maxFramesPerGame, maxStartHeight, sharedStatus, f"  gen {i + 1}"))
                    popScores = resultsTensor                           # [population, games]: everyone, all games
                    candidates, candidateScores = currGen, resultsTensor
                else:
                    # stage 1: everyone, only the first few games
                    popScores = torch.tensor(evaluatePopulation(p, currGen, seeds[:stage1Games], maxFramesPerGame, maxStartHeight, sharedStatus, f"  gen {i + 1} stage 1"))
                    finalistCount = max(keepCount, math.ceil(populationSize * finalistFraction))
                    finalistIdx = averageRanks(popScores).topk(finalistCount, largest=False).indices
                    candidates = [currGen[k] for k in finalistIdx]
                    # stage 2: only the finalists play the remaining games; selection sees ALL games of each finalist
                    stage2 = torch.tensor(evaluatePopulation(p, candidates, seeds[stage1Games:], maxFramesPerGame, maxStartHeight, sharedStatus, f"  gen {i + 1} stage 2"))
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

                bestEver = max(bestEver, selection[0].score)
                # one CSV line per generation (see plotRun.py): mean score per network -> population statistics
                perNetwork = popScores.mean(dim=1)
                q = torch.quantile(perNetwork, torch.tensor([0.1, 0.5, 0.9]))
                logGeneration(path.replace(".pt", ".csv"),
                              f"name={networkName} generations={generations} population={populationSize} games={perNetworkIterations} "
                              f"startFrom={(startNetwork[0] if isinstance(startNetwork, list) else startNetwork).source if startNetwork is not None else None} "
                              f"stage1Games={stage1Games if twoStage else 'None'} "
                              f"mutationRate={mutationRate} sigma={sigma} elites={eliteCount} keepPart={keepPartSelection} maxStartHeight={maxStartHeight}",
                              [i + 1, time.perf_counter() - start, perNetwork.mean().item(), q[1].item(), q[0].item(), q[2].item(),
                               candidateScores.mean(dim=1).max().item(), selection[0].score])
                pbar.set_postfix(best=f"{selection[0].score:.0f}", avg=f"{perNetwork.mean().item():.0f}", bestEver=f"{bestEver:.0f}")
                pbar.update(1)

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
        f"Generations:         {generations}\n"
        f"Games per network:   {perNetworkIterations}\n"
        f"Max frames per game: {maxFramesPerGame:,}\n"
        f"Crossover:           {crossover}\n"
        f"Worker processes:    {concInstances}\n"
        f"Avg. EvalTime:       {evalTime / generations:.3f} seconds\n"
        f"Avg. BreedTime:      {breedTime / generations:.3f} seconds\n"
        f"Total time:          {totalTime:.1f} seconds\n"
        f"Best avg. score:     {selection[0].score:.1f} (last generation)\n"
        f"Saved to:            {savePath}\n"
        f"Pool (top {len(selection)}):      {poolPath(savePath)}\n"
        f"{'=' * 44}"
    )

    return selection[0] # return the best network of the last trained generation


if __name__ == "__main__":
    inS = 23  # inputs: 
    hiS = [23, 23, 23]
    outS = 2 # [steer, shoot]
    gamesPerNetwork = 50
    generations = 200
    model = loadNetwork("models/sprudler_2026-10-02_17-00-59.pt")
    popSize = 200
    mutRate = 0.7
    sigma = 0.02
    eliteCount = 3
    keepPart = 0.1
    concurrent = 24
    maxStartHeight = None
    bestSprudler = sprudlerTrainingLoop(concurrent, gamesPerNetwork, generations, popSize, mutRate, sigma, eliteCount, keepPart, maxStartHeight=maxStartHeight, startNetwork=model, networkName="sprudler", livePlot=True, stage1Games=None, finalistFraction=0.3)
