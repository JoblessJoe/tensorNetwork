import os
import sys
import time
import torch
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool, Manager, TimeoutError
from network import Network, buildNetwork, DEVICE
from training import buildPopulation
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sprudelJump"))
from env import SprudelJumpEnv

LONG_GAME_REPORT_FRAMES = 1_000_000  # a game running longer than this prints a status line every this many frames
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


def runEnv(network: Network):
    '''
    Runs the sprudelJump game steered by the given Neural Network.
    Loops through forwardPasses and returns them to the game as inputs as long as the player is alive.
    Returns the network with the 'score'-attribute set.
    '''
    targetDevice = network.layers[0][0].device
    env = SprudelJumpEnv()  # initializing a new game instance
    startState = env.reset()  # gameState in first iteration
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


def evaluateNetwork(network: Network, iterations: int = 5):
    '''
    Runs one network for a given number of iterations and 
    returns its score, averaged over these number of runs.
    '''
    score = 0
    for i in range(0, iterations):
        res = runEnv(network)
        score += res

    return score / iterations


def sprudlerTrainingLoop(concInstances: int, perNetworkIterations: int, generations: int, populationSize: int, mutationRate: float, sigma: float, eliteCount: int,
                keepPartSelection: float, startNetwork: Network | None = None,
                inputSize: int | None = None, hiddenSizes: list[int] | None = None, outputSize: int | None = None,
                networkName: str = "network", targetDevice: str = "cpu") -> Network:
    '''
    creates/takes a Network instance and trains it for a certain 
    amount of times. Then it returns the trained network and writes its weights into a file.
    '''
    start = time.perf_counter()
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
        currGen = buildPopulation(populationSize, selection, sigma, mutationRate, eliteCount)
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
                evaluate = partial(evaluateNetwork, iterations=perNetworkIterations)
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
                currGen = buildPopulation(populationSize, selection, sigma, mutationRate, eliteCount)
                breedTime += time.perf_counter() - breedStart

                bestEver = max(bestEver, selection[0].score)
                pbar.set_postfix(best=f"{selection[0].score:.0f}", avg=f"{sum(genResults) / len(genResults):.0f}", bestEver=f"{bestEver:.0f}")
                pbar.update(1)

    manager.shutdown()
    totalTime = time.perf_counter() - start
    print(
        f"\n{'=' * 44}\n"
        f"{networkName:^44}\n"
        f"{'=' * 44}\n"
        f"Population size:     {populationSize}\n"
        f"Generations:         {generations}\n"
        f"Games per network:   {perNetworkIterations}\n"
        f"Worker processes:    {concInstances}\n"
        f"Avg. EvalTime:       {evalTime / generations:.3f} seconds\n"
        f"Avg. BreedTime:      {breedTime / generations:.3f} seconds\n"
        f"Total time:          {totalTime:.1f} seconds\n"
        f"Best avg. score:     {selection[0].score:.1f} (last generation)\n"
        f"{'=' * 44}"
    )

    return selection[0] # return the best network of the last trained generation

if __name__ == "__main__":
    inS = 23  # inputs: 
    hiS = [23, 23, 23]
    outS = 2 # [steer, shoot]
    sprudler = buildNetwork(inS, hiS, outS, "cpu") # build the network
    perNetworkIterations = 10
    generations = 50
    popSize = 18
    mutRate = 0.1
    sigma = 0.1
    eliteCount = 3
    keepPart = 0.7
    concurrent = 18
    bestSprudler = sprudlerTrainingLoop(concurrent, perNetworkIterations, generations, popSize, mutRate, sigma, eliteCount, keepPart, startNetwork=sprudler, networkName="sprudler")
