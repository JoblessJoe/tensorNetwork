import os
import sys
import time
import torch
import asyncio
from network import Network, DEVICE, buildNetwork
from training import breed
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


async def runEnv(network: Network):
    '''
    Runs the sprudelJump game steered by the given Neural Network.
    Loops through forwardPasses and returns them to the game as inputs as long as the player is alive.
    Returns the network with the 'score'-attribute set.
    '''
    env = SprudelJumpEnv()  # initializing a new game instance
    startState = env.reset()  # gameState in first iteration
    startTensor = torch.tensor(startState, device=DEVICE).unsqueeze(0)
    startAction = network.forwardPass(startTensor).squeeze(0).tolist()
    currState = env.step(startAction) # caclulating first game input
    alive = not (currState[2])

    while alive:
        action = torch.tensor(currState[0], device=DEVICE).unsqueeze(0) # transforming the action into a tensor for the NN
        nextAction = network.forwardPass(action).squeeze(0).tolist() #  calculated action by the NN
        currState = env.step(nextAction) # calculating the next State based on the prev action
        alive = not (currState[2])

    # Assigning score value to the network object and returning it
    network.score = currState[1]
    return network


async def evaluateNetwork(populationSize: int, network: Network, iterations: int = 5):
    '''
    Runs one network for a give number of iterations and 
    returns its fitness, averaged over these number of runs.
    '''
    fitness = 0
    for i in range(0, iterations):
        res = await runEnv(network).score
        fitness += [res]

    return fitness / iterations


def sprudlerTrainingLoop(iterations: int, populationSize: int, mutationRate: float, sigma: float, eliteCount: int,
                keepPartSelection: float, startNetwork: Network | None = None,
                inputSize: int | None = None, hiddenSizes: list[int] | None = None, outputSize: int | None = None,
                networkName: str = "network") -> Network:
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
            currPopulation.append(buildNetwork(inputSize, hiddenSizes, outputSize))

    # if startNetwork is given build a population of slight variations of itself, through mutation
    elif startNetwork is not None:
        currPopulation = []
        for i in range(0, populationSize):
            network = breed(startNetwork, startNetwork, sigma, mutationRate)
            currPopulation.append(network)
    else:
        raise ValueError("Please provide a starting network OR input- output- and hiddenSizes.")

    for i in range(0, iterations):
        # - run a population of sprudelJump instances in parallel, (cpu parallel)
        # - select the best
        # - breed them
        # - rerun the loop until iterations are done

    
### current issues:
# -  parallelizing sprudelJump instances.
# -  how many different networks per generation?
# - 
# 

### idea:
# - run idk 5-10 concurrent instances eacha different network. 
#  and run each network for 5-10 times and then avg the fitness out over those runs.
#  and then select on this avg fitness. not on one runs result alones


if __name__ == "__main__":
    inS = 23  # inputs: 
    hiS = [23, 23, 23]
    outS = 2 # [steer, shoot]
    sprudler = buildNetwork(inS, hiS, outS) # build the network
