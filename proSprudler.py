import os
import sys
import time
import torch
from multiprocessing import Pool, Process
from network import Network, buildNetwork, DEVICE
from training import breed, pickRandomParents
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sprudelJump"))
from env import SprudelJumpEnv


def SprudlerSelection(networks: list[Network], keepPart: float) -> list[Network]:
    '''
    Takes a list of Networks after training and keeps
    only the networks with the best 'SprudelJump' score. 
    Chosen by 'keepPart' and returns this elite selection as a list.
    '''
    score = []
    for n in networks:
        score += [n.score]
    scoreTensor = torch.tensor(score, device=DEVICE)
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

    while alive:
        action = torch.tensor(currState[0], device=targetDevice).unsqueeze(0) # transforming the action into a tensor for the NN
        nextAction = network.forwardPass(action).squeeze(0).tolist() #  calculated action by the NN
        currState = env.step(nextAction) # calculating the next State based on the prev action
        alive = not (currState[2])

    # Assigning score value to the network object and returning it
    network.score = currState[1]
    return network


def evaluateNetwork(populationSize: int, network: Network, iterations: int = 5):
    '''
    Runs one network for a give number of iterations and 
    returns its fitness, averaged over these number of runs.
    '''
    fitness = 0
    for i in range(0, iterations):
        res = runEnv(network).score
        fitness += [res]

    return fitness / iterations


def sprudlerTrainingLoop(iterations: int, populationSize: int, mutationRate: float, sigma: float, eliteCount: int,
                keepPartSelection: float, startNetwork: Network | None = None,
                inputSize: int | None = None, hiddenSizes: list[int] | None = None, outputSize: int | None = None,
                networkName: str = "network", targetDevice: str = "cpu") -> Network:
    '''
    creates/takes a Network instance and trains it for a certain 
    amount of times. Then it returns the trained network and writes its weights into a file.
    '''
    functionStart = time.perf_counter()
    print("Start of training loop!")
    print("Generating first population...")
    network = startNetwork
    # if no startNetwork is given generate a population of networks with completely random weights and biases
    if inputSize is not None and outputSize is not None and hiddenSizes is not None:
        currPopulation = []
        for i in range(0, populationSize):
            currPopulation.append(buildNetwork(inputSize, hiddenSizes, outputSize, targetDevice))

    # if startNetwork is given build a population of slight variations of itself, through mutation
    elif startNetwork is not None:
        selection = [startNetwork for i in range(0, populationSize)]
        parents = pickRandomParents(selection, populationSize, 0)
        currPopulation = breed(selection, parents, sigma, mutationRate)
    else:
        raise ValueError("Please provide a starting network OR input- output- and hiddenSizes.")

    ## TRAININGLOOP
    # each loop runs one generation of networks, selects the best and breeds them, then the next
    # iteration does the same until the training is done.  
    for i in range(0, iterations):
        # - run a population of sprudelJump instances in parallel, (cpu parallel)
        # - select the best
        # - breed them
        # - rerun the loop until iterations are done
        # - run idk 5-10 concurrent instances each a different network. 
        #  -> run each network for 5-10 times and then avg the score out over those runs.
        #  -> select on this avg score/fitness
        ...


if __name__ == "__main__":
    inS = 23  # inputs: 
    hiS = [23, 23, 23]
    outS = 2 # [steer, shoot]
    sprudler = buildNetwork(inS, hiS, outS, "cpu") # build the network
