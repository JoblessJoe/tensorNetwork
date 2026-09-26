import os
import sys
import torch
import asyncio
from network import Network, DEVICE, buildNetwork
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


async def runGeneration(populationSize: int, startNetwork: Network | None = None, inputSize: int | None = None, hiddenSizes: list[int] | None = None, outputSize: int | None = None):
    network = startNetwork or buildNetwork(inputSize, hiddenSizes, outputSize)
    population = []

    # building the population of concurrent sprudelJump instances
    for i in range(0, populationSize):
        population += [runEnv(network)]
    await population
    return population
        


async def sprudlerTrainingLoop(iterations: int, populationSize: int, mutationRate: float, sigma: float, eliteCount: int,
                keepPartSelection: float, startNetwork: Network | None = None,
                inputSize: int | None = None, hiddenSizes: list[int] | None = None, outputSize: int | None = None,
                networkName: str = "network") -> Network:
    '''
    creates/takes a Network instance and trains it for a certain 
    amount of times. Then it returns the trained network and writes its weights into a file.
    '''
    network = startNetwork
    currGeneration = Network
    # training loop
    for i in range(0, iterations):
        currGeneration = await runGeneration(populationSize, network, inputSize, hiddenSizes, outputSize)

### current issues:
# -  parallelizing sprudelJump instances.
# -  how many different networks per generation?
# -  


if __name__ == "__main__":
    inS = 23  # inputs: 
    hiS = [23, 23, 23]
    outS = 2 # [steer, shoot]
    sprudler = buildNetwork(inS, hiS, outS) # build the network
