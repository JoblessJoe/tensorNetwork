# One worker process per hardware thread already uses every core: numpy's BLAS must NOT start helper threads of its own (they spin and oversubscribe
# the machine: 31 workers x ~2.5 busy threads each, load average ~96 on 32 threads, 4x slower generations). Must be set before numpy is imported.
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
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
from network import Network, buildNetwork, DEVICE, flattenNetwork, unflattenNetwork
from training import buildPopulation
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sprudelJump"))
from env import SprudelJumpEnv, OBSERVATION_LENGTH, STEER_DEAD_ZONE
try:
    from fastEnv import FastBatch, forwardLogits, HAVE_BLAS
except ImportError:
    FastBatch = None
    HAVE_BLAS = False
# the matmuls of the forward pass in compiled code (needs scipy for numba's BLAS binding): same results, fewer numpy calls per decision.
# SPRUDLER_NUMPY_FORWARD=1 switches it off (for comparisons)
COMPILED_FORWARD = HAVE_BLAS and os.environ.get("SPRUDLER_NUMPY_FORWARD") != "1"

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


def sprudlerSelection(networks: list[Network], ranks: torch.Tensor, keepPart: float, count: int | None = None) -> list[Network]:
    '''
    Takes the networks and one average rank per network (low = good) and keeps the best ones:
    'count' of them, or - if not given - the 'keepPart' fraction of 'networks'.
    Returns this elite selection as a _sorted_ list, best first.
    '''
    keep = count if count is not None else math.floor(len(networks) * keepPart)
    sel = ranks.topk(keep, largest=False, sorted=True)
    result = [networks[el] for el in sel.indices]
    return result


def scheduledValue(start: float, end: float | None, progress: float) -> float:
    '''
    A value that moves from 'start' to 'end' over a run (used for sigma and the learning rate in trainingLoopES).
    - 'progress': 0.0 = beginning of the run, 1.0 = end (clamped to that range)
    - 'end' is None: the value stays at 'start' (fixed value, the old behaviour)
    - the path is geometric (start * (end / start) ** progress): both values are scale factors, so a halving costs the same
      "distance" early and late in the run; a straight line would spend most of the run near the start value
    '''
    if end is None:
        return start
    progress = min(max(progress, 0.0), 1.0)
    return start * (end / start) ** progress


def runProgress(generation: int, generations: int, startTime: float, maxHours: float | None) -> float:
    '''How far a run is (0.0 .. 1.0): by the clock if the run has a time limit (then 'generations' is only an upper bound), else by generations.'''
    if maxHours is not None:
        return (time.perf_counter() - startTime) / (maxHours * 3600)
    return generation / generations


def betterSelection(parent: torch.Tensor, noise: torch.Tensor, learningRate: float, sigma: float,
                    ranks: torch.Tensor, populationSize: int) -> tuple:
    '''
    Takes the parent, its children's noise and their ranks, moves the parent toward the better children and returns
    (newParent, newChildren, newNoise) for the next generation.
    - 'noise': [populationSize, P] tensor, row i = the noise that made child i
    - 'ranks': one rank per child (0 = best); turned into vote weights 0.5 - rank / (N - 1), i.e. +0.5 best ... -0.5 worst
    - 'learningRate': how far the parent moves along the average vote
    - 'sigma', 'populationSize': passed on to betterBreed for the new children
    '''
    weightedRanks = [0.5 - rank / (len(ranks) - 1) for rank in ranks]
    avgVote = sum([noise[i] * weightedRanks[i] for i in range(0, populationSize)]) / populationSize
    newParent = parent + learningRate * avgVote
    newPopulation, newNoise = betterBreed(newParent, sigma, populationSize)
    return (newParent, newPopulation, newNoise) 


def betterBreed(parent: torch.Tensor, sigma: float, populationSize: int) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Takes a parent vector, makes 'populationSize' children from it (parent + sigma * noise) and returns (children, noise).
    - 'parent': flattened network, shape [P]
    - 'sigma': size of the noise (std of the gaussian)
    - returns: children and noise, both [populationSize, P]; row i of noise is what made child i
    '''
    noise = torch.randn(populationSize, *parent.shape) # creating the new random noise
    children = parent + sigma * noise
    return (children, noise)


def evaluateNetwork(network: Network, seeds: list, iterations: int = 5, maxFrames: int = MAX_FRAMES_PER_GAME, maxStartHeight: int | None = None, minStartHeight: int = 0, zeroFraction: float = 0.0, steerSmoothing: float = 1.0, monsterFraction: float = 0.0, monsterMult: float = 1.0, fast: bool = False, actionRepeat: int = 1, returnCauses: bool = False, penalties: tuple = (0.0, 0.0, 0.0), deadZone: float | None = None):
    '''
    Plays 'iterations' games with the network side by side (lockstep) and
    returns its score, averaged over these games.
    Every frame, the states of all still-running games are stacked into one
    [numAlive, 33] batch, so the network does ONE forwardPass per frame instead
    of one per game - for a network this small, torch's per-call overhead
    dominates, so this is ~5x faster than playing the games one after another.
    Games still running after 'maxFrames' frames are stopped and count with
    the score they reached by then.
    'steerSmoothing' (0 < a <= 1, 1 = off): the steer that reaches the game is
    applied + a * (output - applied), i.e. the steering can't flip fully from one frame to the next.
    'fast': play with the numba-compiled env (fastEnv, needs numba; same rules, same levels, same scores for actionRepeat=1).
    'actionRepeat': the network decides every actionRepeat-th frame and its action is repeated in between (fast=True only).
    'penalties' (fast=True only): see behaviourPenalty; (0, 0, 0) = the plain game score.
    'deadZone': the steer dead zone of the games (env.STEER_DEAD_ZONE when None; trainingLoopES lowers it during a penalty ramp).
    'monsterFraction'/'monsterMult': monster practice - that share of the games has monsterMult x the normal monster spawn chance.
    'returnCauses' (fast=True only): return (scores, causes) instead of scores; causes[i] = how game i ended: 1 monster while rising, 2 monster otherwise,
    3 fell, 4 no progress, 0 = still running (hit maxFrames).
    '''
    if fast:
        return evaluateNetworkFast(network, seeds, maxFrames, maxStartHeight, minStartHeight, zeroFraction, steerSmoothing, monsterFraction, monsterMult, actionRepeat, returnCauses, penalties, deadZone)
    assert not returnCauses, "returnCauses needs fast=True"
    assert actionRepeat == 1, "actionRepeat > 1 needs fast=True"
    # the network as plain numpy arrays: for batches this small torch's per-call overhead dominates.
    # same computation as Network.forwardPass (ReLU hidden layers, sigmoid output), same float32 results.
    layers = [(w.detach().cpu().numpy().astype(np.float32).T.copy(), b.detach().cpu().numpy().astype(np.float32)) for w, b in network.layers]
    envs = [SprudelJumpEnv() if deadZone is None else SprudelJumpEnv(steerDeadZone=deadZone) for i in range(0, iterations)]
    states = [env.reset(maxStartHeight, seed, minStartHeight, zeroFraction, monsterFraction, monsterMult) for env, seed in zip(envs, seeds)]
    scores = [0.0] * iterations
    # indices of the games still running. Row k of the batch belongs to game alive[k] -
    # that's how each output row gets back to the right game once some games have died.
    alive = list(range(0, iterations))
    frames = 0
    applied = [0.5] * iterations                          # the steer that was actually given to the game

    while alive and frames < maxFrames:
        x = np.array([states[g] for g in alive], dtype=np.float32)  # [numAlive, 33]
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


def behaviourPenalty(batch, penalties: tuple) -> np.ndarray:
    '''Per game: shotFree * (bullets fired while no monster was on screen) + shotAll * (all bullets fired) + steer * (summed absolute change of the applied
    steer between two decisions) + shotMiss * (bullets that hit nothing = bullets fired - monsters killed by a bullet). Subtracted from the score, so ES ranks a
    net that sprays bullets / wiggles lower; + detour * (sideways distance travelled minus the net displacement, summed over the jumps in px: going out and
    coming back costs, going straight to the landing spot costs nothing); + deviation * (squared distance from a constant-speed straight line between the start and the
    landing spot of every jump, in px^2 / 1000: rushing to the landing spot and resting costs, steering steadily all the way costs nothing).
    penalties = (shotFree, shotAll, steer[, shotMiss[, detour[, deviation]]]); all zero = no penalty. A kill is worth +100 points.'''
    shotFree, shotAll, steer, shotMiss, detour, deviation = (tuple(penalties) + (0.0, 0.0, 0.0))[:6]
    shots, shotsFree, steerChange = batch.behaviour()
    return shotFree * shotsFree + shotAll * shots + steer * steerChange + shotMiss * (shots - batch.hits()) + detour * batch.detours() + deviation * batch.deviations() / 1000.0


def evaluateNetworkFast(network: Network, seeds: list, maxFrames: int, maxStartHeight, minStartHeight, zeroFraction, steerSmoothing,
                        monsterFraction, monsterMult, actionRepeat, returnCauses=False, penalties=(0.0, 0.0, 0.0), deadZone=None):
    '''
    Same as evaluateNetwork, but the games run in the numba-compiled env (fastEnv.FastBatch) and the network decides only every
    'actionRepeat' frames (its action is repeated in between). With actionRepeat=1 the scores are identical to evaluateNetwork's.
    Frames are counted in game frames, so maxFrames means the same as before.
    '''
    layers = [(w.detach().cpu().numpy().astype(np.float32).T.copy(), b.detach().cpu().numpy().astype(np.float32)) for w, b in network.layers]
    nIn, nOut = layers[0][0].shape[0], layers[-1][0].shape[1]
    batch = FastBatch(seeds, maxStartHeight, minStartHeight, zeroFraction, monsterFraction, monsterMult, **({} if deadZone is None else dict(deadZone=deadZone)))
    # fused loop: stepObserve does the smoothing, the frames, dropping finished games and the next observation in one
    # compiled call (same results as observe -> step -> filter, ~13% faster). alive[:n] = games still running, obs[:n] = their observations
    alive, obs = batch.startLoop()
    n = len(alive)
    applied = np.full(len(seeds), 0.5)
    frames = 0
    wTs, biases = tuple(wT for wT, _ in layers), tuple(b for _, b in layers)
    if COMPILED_FORWARD and hasattr(batch, "run"):
        # FastBatch.run plays the decisions in a tight loop (sigmoid split into compiled parts + numpy's exp, one error-state switch per call, same
        # results bit for bit, 25-50% less per-decision overhead); it runs in chunks that end where the live status / long-game lines are due
        while n and frames < maxFrames:
            before = frames
            stop = min(maxFrames, (frames // LIVE_UPDATE_FRAMES + 1) * LIVE_UPDATE_FRAMES)
            n, frames = batch.run(alive, n, wTs, biases, applied, steerSmoothing, actionRepeat, frames, stop)
            if liveStatus is not None and n and before // LIVE_UPDATE_FRAMES != frames // LIVE_UPDATE_FRAMES:
                liveStatus[os.getpid()] = (float(batch.scores()[alive[:n]].max()), frames)
            if n and before // LONG_GAME_REPORT_FRAMES != frames // LONG_GAME_REPORT_FRAMES:
                print(f"[worker {os.getpid()}] long game still running: {frames:,} frames, best live score {batch.scores()[alive[:n]].max():,.0f}", flush=True)
    while n and frames < maxFrames:    # the same loop decision by decision (numpy forward, or an older fastEnv without FastBatch.run)
        if COMPILED_FORWARD:
            x = forwardLogits(obs[:n], wTs, biases)
            with np.errstate(over="ignore"):
                x = 1 / (1 + np.exp(-x))      # numpy's own float32 exp on purpose (a compiled one differs in the last digit)
        else:
            x = obs[:n]
            for L, (wT, b) in enumerate(layers):
                x = x @ wT + b
                if L < len(layers) - 1:
                    x = np.maximum(x, 0)
                else:
                    with np.errstate(over="ignore"):
                        x = 1 / (1 + np.exp(-x))
        n = batch.stepObserve(alive, n, x, applied, steerSmoothing, actionRepeat)
        before, frames = frames, frames + actionRepeat
        if liveStatus is not None and n and before // LIVE_UPDATE_FRAMES != frames // LIVE_UPDATE_FRAMES:
            liveStatus[os.getpid()] = (float(batch.scores()[alive[:n]].max()), frames)
        if n and before // LONG_GAME_REPORT_FRAMES != frames // LONG_GAME_REPORT_FRAMES:
            print(f"[worker {os.getpid()}] long game still running: {frames:,} frames, best live score {batch.scores()[alive[:n]].max():,.0f}", flush=True)
    if liveStatus is not None:
        liveStatus.pop(os.getpid(), None)
    scores = batch.scores() - behaviourPenalty(batch, penalties) if any(penalties) else batch.scores()
    return (scores.tolist(), batch.causes().tolist()) if returnCauses else scores.tolist()


def evaluateNetworksFast(networks: list, seeds: list, maxFrames: int, maxStartHeight, minStartHeight, zeroFraction, steerSmoothing,
                         monsterFraction, monsterMult, actionRepeat, returnCauses=False, penalties=(0.0, 0.0, 0.0), deadZone=None):
    '''
    evaluateNetworkFast for a GROUP of networks of the same shape: they all play the same 'seeds' in ONE lockstep loop (one FastBatch of
    len(networks) * len(seeds) games, FastBatch.runGroup). Every game plays exactly as it would alone (bit-identical scores), but the fixed cost of a
    decision (the Python loop, np.exp, the compiled step's dispatch) is shared by the whole group: with 50 games per child that cost is about a
    third of an evaluation. Returns one result per network (scores, or (scores, causes) with returnCauses), in the order of 'networks'.
    '''
    k, S = len(networks), len(seeds)
    per = [[(w.detach().cpu().numpy().astype(np.float32).T.copy(), b.detach().cpu().numpy().astype(np.float32)) for w, b in net.layers] for net in networks]
    nIn, nOut = per[0][0][0].shape[0], per[0][-1][0].shape[1]
    wsStack = tuple(np.ascontiguousarray(np.stack([layers[L][0] for layers in per])) for L in range(len(per[0])))
    bsStack = tuple(np.ascontiguousarray(np.stack([layers[L][1] for layers in per])) for L in range(len(per[0])))
    batch = FastBatch(list(seeds) * k, maxStartHeight, minStartHeight, zeroFraction, monsterFraction, monsterMult, **({} if deadZone is None else dict(deadZone=deadZone)))
    alive, obs = batch.startLoop()
    n = len(alive)
    applied = np.full(k * S, 0.5)
    frames = 0
    while n and frames < maxFrames:
        before = frames
        stop = min(maxFrames, (frames // LIVE_UPDATE_FRAMES + 1) * LIVE_UPDATE_FRAMES)
        n, frames = batch.runGroup(alive, n, wsStack, bsStack, S, applied, steerSmoothing, actionRepeat, frames, stop)
        if liveStatus is not None and n and before // LIVE_UPDATE_FRAMES != frames // LIVE_UPDATE_FRAMES:
            liveStatus[os.getpid()] = (float(batch.scores()[alive[:n]].max()), frames)
        if n and before // LONG_GAME_REPORT_FRAMES != frames // LONG_GAME_REPORT_FRAMES:
            print(f"[worker {os.getpid()}] long game still running: {frames:,} frames, best live score {batch.scores()[alive[:n]].max():,.0f}", flush=True)
    if liveStatus is not None:
        liveStatus.pop(os.getpid(), None)
    scores = (batch.scores() - behaviourPenalty(batch, penalties) if any(penalties) else batch.scores()).reshape(k, S).tolist()
    if returnCauses:
        causes = batch.causes().reshape(k, S).tolist()
        return [(scores[c], causes[c]) for c in range(k)]
    return scores


GROUP_SIZE_MAX = 4             # children per worker task at most
GROUP_TARGET_GAMES = 200       # a group should hold about this many games: below it the fixed cost per decision matters, above it nothing is gained
GROUP_MIN_TASKS_PER_WORKER = 8  # the end of a generation waits for the last tasks: bigger groups only while every worker still gets this many tasks


def groupSize(numNetworks: int, gamesPerNetwork: int, workers: int) -> int:
    '''How many networks one worker task plays together (1 = no grouping). SPRUDLER_GROUP=n overrides (1 switches it off).'''
    forced = os.environ.get("SPRUDLER_GROUP")
    if forced:
        return max(1, int(forced))
    return max(1, min(GROUP_SIZE_MAX, GROUP_TARGET_GAMES // max(1, gamesPerNetwork), numNetworks // (GROUP_MIN_TASKS_PER_WORKER * max(1, workers))))


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
                       sharedStatus, desc: str, minStartHeight: int = 0, zeroFraction: float = 0.0, steerSmoothing: float = 1.0, monsterFraction: float = 0.0, monsterMult: float = 1.0, fast: bool = False, actionRepeat: int = 1, penalties: tuple = (0.0, 0.0, 0.0), deadZone: float | None = None) -> list[list[float]]:
    '''
    Plays every network in 'networks' on the same 'seeds' (one game per seed) using the worker pool 'p'.
    Returns one list of per-game scores per network, in the same order as 'networks'.
    '''
    evaluate = partial(evaluateNetwork, seeds=seeds, iterations=len(seeds), maxFrames=maxFrames, maxStartHeight=maxStartHeight, minStartHeight=minStartHeight, zeroFraction=zeroFraction, steerSmoothing=steerSmoothing, monsterFraction=monsterFraction, monsterMult=monsterMult, fast=fast, actionRepeat=actionRepeat, penalties=penalties, deadZone=deadZone)
    k = groupSize(len(networks), len(seeds), getattr(p, "_processes", 1)) if fast and COMPILED_FORWARD and hasattr(FastBatch, "runGroup") else 1
    if k > 1:    # several children per task, played in one lockstep loop (same scores, less fixed cost per decision)
        evaluate = partial(evaluateNetworksFast, seeds=seeds, maxFrames=maxFrames, maxStartHeight=maxStartHeight, minStartHeight=minStartHeight, zeroFraction=zeroFraction, steerSmoothing=steerSmoothing, monsterFraction=monsterFraction, monsterMult=monsterMult, actionRepeat=actionRepeat, penalties=penalties, deadZone=deadZone)
        tasks = [networks[i:i + k] for i in range(0, len(networks), k)]
    else:
        tasks = networks
    results = p.imap(evaluate, tasks)  # yields each task's result as soon as it is done, in order
    scores = []
    with tqdm(total=len(networks), desc=desc, unit="net", leave=False) as bar:
        while len(scores) < len(networks):
            try:
                got = results.next(timeout=1.0)
                if k > 1:
                    scores.extend(got)
                    bar.update(len(got))
                else:
                    scores.append(got)
                    bar.update(1)
            except TimeoutError:
                pass
            live = sorted(sharedStatus.values(), reverse=True)  # [(score, frames), ...] best first
            top = "  ".join(f"{sc:,.0f}@{fr // 1000}k" for sc, fr in live[:5])
            bar.set_postfix_str(f"live={len(live)}  top: {top}" if live else "")
    return scores


class _GpuCounters:
    '''The counters of a GPU evaluation with FastBatch's method names, flat (child-major), so behaviourPenalty works on them unchanged.'''

    def __init__(self, r):
        self.r = r

    def behaviour(self):
        return self.r["shots"].ravel(), self.r["shotsFree"].ravel(), self.r["steerChange"].ravel()

    def hits(self):
        return self.r["hits"].ravel()

    def detours(self):
        return self.r["detour"].ravel()

    def deviations(self):
        return self.r["deviation"].ravel()


def evaluatePopulationGpu(state: dict, rows: torch.Tensor, template: Network, seeds: list, maxFrames: int, maxStartHeight, minStartHeight, zeroFraction,
                          steerSmoothing, monsterFraction, monsterMult, actionRepeat, penalties, deadZone=None) -> np.ndarray:
    '''evaluatePopulation on the GPU: every child (a flat parameter row, layout of flattenNetwork) plays all 'seeds'. Returns [children, games] scores minus the
    behaviour penalties, like the CPU path. 'state' keeps the GPU batch between generations.'''
    from gpuEnv import GpuBatch
    P, S = rows.shape[0], len(seeds)
    if state.get("shape") != (P, S):
        state["batch"], state["shape"] = GpuBatch(children=P, games=S), (P, S)
    flat = rows.detach().cpu().numpy().astype(np.float32)
    weights, biases, offset = [], [], 0
    for w, _ in template.layers:    # flattenNetwork: all weight matrices ([out, in]) first, then all biases
        n = w.numel()
        weights.append(flat[:, offset:offset + n].reshape(P, w.shape[0], w.shape[1]).transpose(0, 2, 1))
        offset += n
    for _, b in template.layers:
        biases.append(flat[:, offset:offset + b.numel()])
        offset += b.numel()
    r = state["batch"].evaluate([a for wb in zip(weights, biases) for a in wb], seeds, smooth=steerSmoothing, repeat=actionRepeat, maxStart=maxStartHeight,
                                minStart=minStartHeight, zeroFraction=zeroFraction, monsterFraction=monsterFraction, monsterMult=monsterMult, maxFrames=maxFrames,
                                **({} if deadZone is None else dict(deadZone=deadZone)))
    scores = r["score"]
    if any(penalties):
        scores = scores - behaviourPenalty(_GpuCounters(r), penalties).reshape(P, S)
    return scores.astype(np.float32)    # the CPU path's torch.tensor(list of floats) is float32 too


def averageRanks(scores: torch.Tensor) -> torch.Tensor:
    '''[networks, games] scores -> one average rank per network (0 = best), ranking all networks against each other in every game.'''
    return scores.argsort(dim=0, descending=True).argsort(dim=0).float().mean(1)


def trainingLoopES(concInstances: int,            perNetworkIterations: int,              generations: int,                            
                        populationSize: int,           mutationRate: float,                    sigma: float, eliteCount: int, learningRate: float, 
                        keepPartSelection: float,               maxFramesPerGame: int = MAX_FRAMES_PER_GAME, 
                        startNetwork: Network | list[Network] | None = None,                   livePlot: bool = True,
                        inputSize: int | None = None,  hiddenSizes: list[int] | None = None,   outputSize: int | None = None,
                        networkName: str = "network",  targetDevice: str = "cpu",              maxStartHeight: int |None = None,
                        maxHours: float | None = None,
                        minStartHeight: int = 0,       zeroFraction: float = 0.0,              snapshotAt: tuple = (), 
                        steerSmoothing: float = 1.0, monsterFraction: float = 0.0, monsterMult: float = 1.0,
                        fast: bool = False, actionRepeat: int = 1, replaceRate: float = 0.0,
                        sigmaEnd: float | None = None, learningRateEnd: float | None = None, penalties: tuple = (0.0, 0.0, 0.0), gpu: bool = False, penaltyRamp: float = 0.0,
                        penaltyStartScore: float | None = None, penaltyFullScore: float | None = None) -> Network:
    '''
    creates/takes a Network instance and trains it for a certain 
    amount of times. Then it returns the trained network and writes its weights into a file.

    Evaluation: every child plays all 'perNetworkIterations' games on the generation's shared seeds (no two-stage:
    in ES every child's rank is a vote, so the bad ones need full scores too).

    'sigmaEnd' / 'learningRateEnd': if given, sigma / the learning rate move geometrically from their start value to this value over the run
    (by the clock if 'maxHours' is set, else by generation, see scheduledValue); None = fixed.
    'penalties' (fast=True only): (shotFree, shotAll, steer[, shotMiss[, detour[, deviation]]]) per game, subtracted from the score the votes are made of (see behaviourPenalty); the CSV's
    mean / median / best are then penalised fitness, not game score. (0, 0, 0) = off.
    'penaltyRamp': the penalties AND the steer dead zone grow linearly from 0 to their full size over this share of the run (progress as in scheduledValue: the clock
    if 'maxHours' is set); 0 = full size from the first generation. A net from scratch dies within seconds, a full penalty on its tiny scores would drown the
    signal to jump, and inside the dead zone (steer output within 0.2 of 0.5) a random net never moves, so all children score the same and ES learns nothing.
    'penaltyStartScore' / 'penaltyFullScore': the ramp follows the net's COMPETENCE instead of the clock (overrides penaltyRamp): the factor is 0 until the smoothed generation
    mean first reaches penaltyStartScore, grows linearly to 1 at penaltyFullScore, and never shrinks again. Found 2026-10-29: from scratch even 1-3% of the penalties stop ES from
    learning to climb (the ranks follow the consistent penalty signal, not the weak score signal), so a fresh net must learn to play first and only then get penalised.
    'gpu': evaluate the population on the GPU (sprudelJump/gpuEnv.py, needs ~/venv-gpu; the worker pool is then unused, start the run with few workers).
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

    
    ## FROM SCRATCH RUN
    # if no startNetwork is given generate a population of networks with completely random weights and biases
    if inputSize is not None and outputSize is not None and hiddenSizes is not None:
        parent = buildNetwork(inputSize, hiddenSizes, outputSize, targetDevice)

    ## CONTINUING TRAINING OF A NETWORK
    # if startNetwork is given build a population of slight variations of itself, through mutation
    elif startNetwork is not None:
        # network start-pool
        if isinstance(startNetwork, list) :
            parent = startNetwork[0]
        # single start-network
        else:
            parent = startNetwork
    else:
        raise ValueError("Please provide a starting network OR input- output- and hiddenSizes.")

    template = parent
    parent = flattenNetwork(parent)
    currGen, noise = betterBreed(parent, sigma, populationSize)
    gpuState = {}    # the GPU batch, created on the first generation (after the worker pool has forked)
    competence = None    # smoothed generation mean (penaltyStartScore / penaltyFullScore ramp)
    competenceBest = 0.0

    evalTime = 0.0
    bestEver = float("-inf")
    manager = Manager()
    sharedStatus = manager.dict()
    gensDone = 0
    try:
        with Pool(concInstances, initializer=workerInit, initargs=(sharedStatus,)) as p:
            with tqdm(total=generations, desc=networkName, unit="gen") as pbar:
                
                ### START TRAININGLOOP ###
                    # each loop runs one generation of networks, selects the best and breeds them, then the next
                    # iteration does the same until the training is done. 
                for i in range(0, generations):
                    ## EVALUATION
                    evalStart = time.perf_counter()
                    seeds = [random.randrange(2**32) for _ in range(perNetworkIterations)]  # shared by all networks this generation
                    if penaltyStartScore is not None:
                        rampFactor = min(1.0, max(0.0, (competenceBest - penaltyStartScore) / (penaltyFullScore - penaltyStartScore)))
                    else:
                        rampFactor = 1.0 if penaltyRamp <= 0 else min(1.0, runProgress(i, generations, start, maxHours) / penaltyRamp)
                    genPenalties = tuple(x * rampFactor for x in penalties)    # the penalties of this generation (grow during the ramp)
                    genDeadZone = None if (penaltyRamp <= 0 and penaltyStartScore is None) else STEER_DEAD_ZONE * rampFactor    # the steer dead zone grows with them (a net from scratch never moves inside it)

                    if gpu:    # all children's games at once on the GPU (gpuEnv), straight from the flat parameter rows
                        popScores = torch.tensor(evaluatePopulationGpu(gpuState, currGen, template, seeds, maxFramesPerGame, maxStartHeight, minStartHeight, zeroFraction, steerSmoothing, monsterFraction, monsterMult, actionRepeat, genPenalties, genDeadZone))
                    else:
                        currGenUnflattened = [unflattenNetwork(row, template) for row in currGen]
                        popScores = torch.tensor(evaluatePopulation(p, currGenUnflattened, seeds, maxFramesPerGame, maxStartHeight, sharedStatus, f"  gen {i + 1}", minStartHeight, zeroFraction, steerSmoothing, monsterFraction, monsterMult, fast, actionRepeat, genPenalties, genDeadZone))                           # [population, games]: everyone, all games

                    evalTime += time.perf_counter() - evalStart

                    # saving the score (mean points per game) of every child
                    if not gpu:
                        for j, net in enumerate(currGenUnflattened):
                            net.score = popScores[j].mean().item()
                    
                    ## SELECTION & REPRODUCTION
                    # rank the children against each other in every game and average the ranks
                    # then rank those averages again: clean ranks 0..N-1 (0 = best), so the vote weights really span +0.5 .. -0.5
                    ranks = averageRanks(popScores).argsort().argsort().float()
                    progress = runProgress(i + 1, generations, start, maxHours)
                    currSigma = scheduledValue(sigma, sigmaEnd, progress)
                    currLearningRate = scheduledValue(learningRate, learningRateEnd, progress)
                    parent, currGen, noise =  betterSelection(parent, noise, currLearningRate, currSigma, ranks, populationSize)

                    genMean = popScores.mean().item()
                    competence = genMean if competence is None else 0.98 * competence + 0.02 * genMean
                    competenceBest = max(competenceBest, competence)
                    perNetwork = popScores.mean(dim=1)             # mean score per child
                    bestChild = perNetwork.max().item()            # the parent itself never plays, so the best child is the 'best' score
                    # saving a network every 50 iterations as a 'Checkpoint' if the run fails
                    if (i + 1) % 50 == 0:
                        unflattenedParent = unflattenNetwork(parent, template)
                        saveNetwork(unflattenedParent, networkName, path)
                        savePool([unflattenedParent], poolPath(path))
                    if (i + 1) in snapshotAt:     # a copy of the parent at this generation (the normal checkpoint file is overwritten)
                        saveNetwork(unflattenNetwork(parent, template), networkName, path.replace(".pt", f"_g{i + 1}.pt"))

                    bestEver = max(bestEver, bestChild)
                    # one CSV line per generation (see plotRun.py): mean score per network -> population statistics
                    q = torch.quantile(perNetwork, torch.tensor([0.1, 0.5, 0.9]))
                    logGeneration(path.replace(".pt", ".csv"),
                                  f"name={networkName} generations={generations} population={populationSize} games={perNetworkIterations} "
                                  f"startFrom={(startNetwork[0] if isinstance(startNetwork, list) else startNetwork).source if startNetwork is not None else None} "
                                  f"shape={'-'.join(str(n) for n in [template.layers[0][0].shape[1]] + [w.shape[0] for w, _ in template.layers])} "
                                  f"method=ES learningRate={learningRate} mutationRate={mutationRate} replaceRate={replaceRate} sigma={sigma} elites={eliteCount} keepPart={keepPartSelection} minStartHeight={minStartHeight} maxStartHeight={maxStartHeight} zeroFraction={zeroFraction} steerSmoothing={steerSmoothing} monsterFraction={monsterFraction} monsterMult={monsterMult} fast={fast} actionRepeat={actionRepeat} sigmaEnd={sigmaEnd} learningRateEnd={learningRateEnd} maxHours={maxHours} device={'gpu' if gpu else 'cpu'} penaltyRamp={penaltyRamp}",
                                  [i + 1, time.perf_counter() - start, perNetwork.mean().item(), q[1].item(), q[0].item(), q[2].item(), bestChild, bestChild])    # 'best' and 'chosen' columns: both the best child
                    pbar.set_postfix(best=f"{bestChild:.0f}", avg=f"{perNetwork.mean().item():.0f}", bestEver=f"{bestEver:.0f}", sigma=f"{currSigma:.4f}", lr=f"{currLearningRate:.3f}")
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
    parentNet = unflattenNetwork(parent, template)    # the trained result: the parent of the last finished generation
    savePath = saveNetwork(parentNet, networkName, path)
    savePool([parentNet], poolPath(path))             # one-element pool, so nightRun's afterPool still works
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
        f"Learning rate:       {learningRate}\n"
        f"Worker processes:    {concInstances}\n"
        f"Avg. EvalTime:       {evalTime / gensDone:.3f} seconds\n"
        f"Total time:          {totalTime:.1f} seconds\n"
        f"Best child score:    {bestChild:.1f} (last generation)\n"
        f"Saved to:            {savePath}\n"
        f"Pool (parent only):  {poolPath(savePath)}\n"
        f"{'=' * 44}"
    )

    return parentNet # the trained parent


def sprudlerTrainingLoop(concInstances: int,            perNetworkIterations: int,              generations: int,                            
                         populationSize: int,           mutationRate: float,                    sigma: float, eliteCount: int,
                         keepPartSelection: float,      crossover: bool = False,                maxFramesPerGame: int = MAX_FRAMES_PER_GAME, 
                         startNetwork: Network | list[Network] | None = None,                   livePlot: bool = True,
                         inputSize: int | None = None,  hiddenSizes: list[int] | None = None,   outputSize: int | None = None,
                         networkName: str = "network",  targetDevice: str = "cpu",              maxStartHeight: int |None = None,
                         stage1Games: int | None = None,finalistFraction: float = 0.3,          maxHours: float | None = None,               
                         minStartHeight: int = 0,       zeroFraction: float = 0.0,              snapshotAt: tuple = (), 
                        steerSmoothing: float = 1.0, monsterFraction: float = 0.0, monsterMult: float = 1.0,
                        fast: bool = False, actionRepeat: int = 1, replaceRate: float = 0.0) -> Network:
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
        currGen = buildPopulation(populationSize, selection, sigma, mutationRate, eliteCount, crossover=crossover, replaceRate=replaceRate)
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
                        resultsTensor = torch.tensor(evaluatePopulation(p, currGen, seeds, maxFramesPerGame, maxStartHeight, sharedStatus, f"  gen {i + 1}", minStartHeight, zeroFraction, steerSmoothing, monsterFraction, monsterMult, fast, actionRepeat))
                        popScores = resultsTensor                           # [population, games]: everyone, all games
                        candidates, candidateScores = currGen, resultsTensor
                    else:
                        # stage 1: everyone, only the first few games
                        popScores = torch.tensor(evaluatePopulation(p, currGen, seeds[:stage1Games], maxFramesPerGame, maxStartHeight, sharedStatus, f"  gen {i + 1} stage 1", minStartHeight, zeroFraction, steerSmoothing, monsterFraction, monsterMult, fast, actionRepeat))
                        finalistCount = max(keepCount, math.ceil(populationSize * finalistFraction))
                        finalistIdx = averageRanks(popScores).topk(finalistCount, largest=False).indices
                        candidates = [currGen[k] for k in finalistIdx]
                        # stage 2: only the finalists play the remaining games; selection sees ALL games of each finalist
                        stage2 = torch.tensor(evaluatePopulation(p, candidates, seeds[stage1Games:], maxFramesPerGame, maxStartHeight, sharedStatus, f"  gen {i + 1} stage 2", minStartHeight, zeroFraction, steerSmoothing, monsterFraction, monsterMult, fast, actionRepeat))
                        candidateScores = torch.cat([popScores[finalistIdx], stage2], dim=1)   # [finalists, all games]
                    evalTime += time.perf_counter() - evalStart

                    # saving the score (mean points per game) for every network that was selection-relevant
                    for j, net in enumerate(currGen):
                        net.score = popScores[j].mean().item()               # stage-1 / all-games mean for everyone ...
                    for j, net in enumerate(candidates):
                        net.score = candidateScores[j].mean().item()         # ... overwritten by the all-games mean for the candidates

                    ## SELECTION
                    # rank the candidates against each other in every game, average the ranks, keep the best ones
                    selection = sprudlerSelection(candidates, averageRanks(candidateScores), keepPartSelection, count=keepCount)

                    ## REPRODUCTION
                    breedStart = time.perf_counter()
                    currGen = buildPopulation(populationSize, selection, sigma, mutationRate, eliteCount, crossover=crossover, replaceRate=replaceRate)
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
                                  f"shape={'-'.join(str(n) for n in [selection[0].layers[0][0].shape[1]] + [w.shape[0] for w, _ in selection[0].layers])} "
                                  f"mutationRate={mutationRate} replaceRate={replaceRate} sigma={sigma} elites={eliteCount} keepPart={keepPartSelection} minStartHeight={minStartHeight} maxStartHeight={maxStartHeight} zeroFraction={zeroFraction} steerSmoothing={steerSmoothing} monsterFraction={monsterFraction} monsterMult={monsterMult} fast={fast} actionRepeat={actionRepeat}",
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
    model = loadPool("models/behaviour_control_2026-10-27_08-22-28_pool.pt")   # best so far (ES, repeat 5)
    steerSmoothing = 0.96875   # 1 = off (0.96875 = action repeat 5)
    fast = True            # numba-compiled env (identical results, several times faster)
    actionRepeat = 5       # the network decides every n-th frame (needs fast=True)
    popSize = 200
    mutRate = 0.7   # careful settings: polish what the explore run found
    sigma = 0.02
    monsterFraction = 0.0   # monster practice (no gain in the 2x2 test): share of the games with monsterMult x the normal monster spawn chance
    monsterMult = 2.0
    replaceRate = 0.001
    eliteCount = 3
    keepPart = 0.1
    concurrent = 23
    maxStartHeight = 30000
    minStartHeight = 20000
    zeroFraction = 0.5
    maxHours = 5
    bestSprudler = sprudlerTrainingLoop(concurrent, gamesPerNetwork, generations, popSize, mutRate, sigma, eliteCount, keepPart, replaceRate=replaceRate, maxStartHeight=maxStartHeight, minStartHeight=minStartHeight, zeroFraction=zeroFraction, maxHours=maxHours, startNetwork=model, networkName="careful2", livePlot=True, stage1Games=20, finalistFraction=0.3, steerSmoothing=steerSmoothing, monsterFraction=monsterFraction, monsterMult=monsterMult, fast=fast, actionRepeat=actionRepeat)
