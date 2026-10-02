import os
import sys
import time
from datetime import datetime
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


def SprudlerSelection(networks: list[Network], keepPart: float) -> list[Network]:
    '''
    Takes a list of Networks after training and keeps
    only the networks with the best 'SprudelJump' score. 
    Chosen by 'keepPart' and returns this elite selection as a _sorted_ list.
    '''
    score = []
    for n in networks:
        score += [n.score]
    scoreTensor = torch.tensor(score, device="cpu")
    split = int(len(networks) * keepPart)
    sel = torch.topk(scoreTensor, split)
    result = [networks[el] for el in sel.indices]
    return result


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
    torch.save({"layers": network.layers, "score": network.score}, path)
    return path


def loadNetwork(path: str, targetDevice: str = "cpu") -> Network:
    '''Loads a network saved by 'saveNetwork', e.g. to continue training via 'startNetwork'.'''
    data = torch.load(path, map_location=targetDevice, weights_only=True)
    return Network(layers=data["layers"], score=data["score"])


def runEnv(network: Network, startHeight: int | None = None):
    '''
    Runs the sprudelJump game steered by the given Neural Network.
    Loops through forwardPasses and returns them to the game as inputs as long as the player is alive.
    Returns the network with the 'score'-attribute set.
    '''
    targetDevice = network.layers[0][0].device
    env = SprudelJumpEnv()  # initializing a new game instance
    startState = env.reset(startHeight)  # gameState in first iteration
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


def evaluateNetwork(network: Network, iterations: int = 5, maxFrames: int = MAX_FRAMES_PER_GAME, startHeight: int | None = None):
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
    targetDevice = network.layers[0][0].device
    envs = [SprudelJumpEnv() for i in range(0, iterations)]
    states = [env.reset(startHeight) for env in envs]
    scores = [0.0] * iterations
    # indices of the games still running. Row k of the batch belongs to game alive[k] -
    # that's how each output row gets back to the right game once some games have died.
    alive = list(range(0, iterations))
    frames = 0

    while alive and frames < maxFrames:
        batch = torch.tensor([states[g] for g in alive], device=targetDevice)  # [numAlive, 23]
        actions = network.forwardPass(batch).tolist()  # [numAlive, 2], row k -> game alive[k]
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
    return sum(scores) / iterations


def sprudlerTrainingLoop(concInstances: int, perNetworkIterations: int, generations: int, populationSize: int, mutationRate: float, sigma: float, eliteCount: int,
                keepPartSelection: float, crossover: bool = False, maxFramesPerGame: int = MAX_FRAMES_PER_GAME, startNetwork: Network | None = None,
                inputSize: int | None = None, hiddenSizes: list[int] | None = None, outputSize: int | None = None,
                networkName: str = "network", targetDevice: str = "cpu", startHeight: int |None = None) -> Network:
    '''
    creates/takes a Network instance and trains it for a certain 
    amount of times. Then it returns the trained network and writes its weights into a file.
    '''
    start = time.perf_counter()
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = f"models/{networkName}_{timestamp}.pt"
    print("Start of training loop!")
    print("Generating first population...")
    # if no startNetwork is given generate a population of networks with completely random weights and biases
    if inputSize is not None and outputSize is not None and hiddenSizes is not None:
        currGen = []
        for i in range(0, populationSize):
            currGen.append(buildNetwork(inputSize, hiddenSizes, outputSize, targetDevice))

    # if startNetwork is given build a population of slight variations of itself, through mutation
    elif startNetwork is not None:
        selection = [startNetwork for i in range(0, populationSize)]
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
                evalStart = time.perf_counter()
                evaluate = partial(evaluateNetwork, iterations=perNetworkIterations, maxFrames=maxFramesPerGame, startHeight=startHeight)
                # imap instead of map: same results in the same order, but yields each one as soon as it's done,
                # so the inner bar can show how many networks of this generation have finished.
                # next(timeout=1) wakes up every second even if nothing finished, to refresh the live scores.
                genResults = []
                results = p.imap(evaluate, currGen)
                with tqdm(total=len(currGen), desc=f"  gen {i + 1}", unit="net", leave=False) as genBar:
                    while len(genResults) < len(currGen):
                        try:
                            genResults.append(results.next(timeout=1.0))
                            genBar.update(1)
                        except TimeoutError:
                            pass
                        live = sorted(sharedStatus.values(), reverse=True)  # [(score, frames), ...] best first
                        top = "  ".join(f"{sc:,.0f}@{fr // 1000}k" for sc, fr in live[:5])
                        genBar.set_postfix_str(f"live={len(live)}  top: {top}" if live else "")
                evalTime += time.perf_counter() - evalStart

                # saving the score for every network
                for j in range(0, len(genResults)):
                    currGen[j].score = genResults[j]
                selection = SprudlerSelection(currGen, keepPartSelection)
                breedStart = time.perf_counter()
                currGen = buildPopulation(populationSize, selection, sigma, mutationRate, eliteCount, crossover=crossover)
                breedTime += time.perf_counter() - breedStart
                # saving a network every 50 iterations as a 'Checkpoint' if the run fails
                if (i + 1) % 50 == 0:
                    saveNetwork(selection[0], networkName, path)

                bestEver = max(bestEver, selection[0].score)
                pbar.set_postfix(best=f"{selection[0].score:.0f}", avg=f"{sum(genResults) / len(genResults):.0f}", bestEver=f"{bestEver:.0f}")
                pbar.update(1)

    manager.shutdown()
    savePath = saveNetwork(selection[0], networkName, path)
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
        f"{'=' * 44}"
    )

    return selection[0] # return the best network of the last trained generation

if __name__ == "__main__":
    inS = 23  # inputs: 
    hiS = [23, 23, 23]
    outS = 2 # [steer, shoot]
    startNet = loadNetwork("models/sprudler_2026-10-01_19-16-47.pt")
    perNetworkIterations = 50
    generations = 200
    popSize = 200
    mutRate = 1.0
    sigma = 0.03
    eliteCount = 3
    keepPart = 0.1
    concurrent = 20
    startHeight = 20000
    bestSprudler = sprudlerTrainingLoop(concurrent, perNetworkIterations, generations, popSize, mutRate, sigma, eliteCount, keepPart, startHeight=startHeight, startNetwork=startNet, networkName="sprudler")
