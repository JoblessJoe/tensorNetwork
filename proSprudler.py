import os
import sys

import torch
from network import Network, DEVICE
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sprudelJump"))
from env import SprudelJumpEnv


def runEnv(network: Network):
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


def gameLoop(network, ):
    ...


if __name__ == "__main__":
    inputSize = 23  # inputs: 
    hiddenSizes = [23, 23, 23]
    outputSize = 2 # [steer, shoot]
