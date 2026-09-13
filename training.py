from __future__ import annotations
import math
from random import uniform, randint
import time
import torch
from network import Network, buildNetwork


def inCircle(inx, iny) -> float:
    '''Circle center at x=2, y=1, radius = 0.4'''
    dist = math.sqrt(((inx-2)**2) + (iny-1)**2)
    if dist <= 0.4:
        return 1.0
    return 0.0


def generateTrainingData(dataPoints=1000):
    '''
    Generates a list of Points to train the model on.
    Each element consists of a tuple: (x,y, isInCircle)
    x and y coordinates and a binary bool value if it is in the circle.
    By default it generates a list with a 1000 data points, but that number can be chosen individually.
    '''
    if not (100 <= dataPoints <= 100000):
        raise ValueError(f"dataPoints value must be between 100 and 100000. Your value: {dataPoints}")
    data = []
    xBounds = (1.5, 2.5)
    yBounds = (0.5, 1.5)
    for i in range(0, dataPoints):
        xVal = uniform(xBounds[0], xBounds[1])
        yVal = uniform(yBounds[0], yBounds[1])
        isInCircle = inCircle(xVal, yVal)
        point = (xVal, yVal, isInCircle)
        data.append(point)
    return data


def crossEntropyLoss(target: float, prediction: torch.Tensor) -> float:
    '''Binary cross entropy between a 0/1 target and the network's (scalar tensor) prediction.'''
    pred = prediction.clamp(1e-7, 1 - 1e-7)  # forwardPass ends in ReLU, not a bounded probability, so clamp to keep log() finite
    loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    return loss.item()


def evaluateNetwork(network: Network, trainingData: list, detailed: bool = False) -> float | dict:
    '''
    Takes a List of TrainingPoints and a Network and returns the cross entropy loss of the network.
    \nIf parameter 'detailed' is set to True it will output a List of results for each trainingPoint.
    '''
    networkLoss = 0.0
    detailedRes = []
    for point in trainingData:
        inputTensor = torch.tensor([point[0], point[1]])
        networkPred = network.forwardPass(inputTensor)[0]
        loss = crossEntropyLoss(point[2], networkPred)
        networkLoss += loss
        if detailed:
            curr = {"point": (point[0], point[1]), "solution": point[2], "networkPrediction": networkPred.item(), "loss": loss}
            detailedRes.append(curr)

    networkLoss = networkLoss / len(trainingData)
    if detailed:
        return {"detailedResult": detailedRes, "NetworkLoss": networkLoss}
    return networkLoss


def selection(networks: list[Network], keepPart: float) -> list[Network]:
    '''Takes a list of Networks after Training and keep only the top percentage chosen by 'keepPart' and returns this elite selection as a list'''

    ranked = sorted(networks, key=lambda e: e.fitness, reverse=True)
    split = int(len(networks) * keepPart)
    return ranked[:split]


def pickRandomParents(selection: list[Network]) -> tuple[Network]:
    '''Picks two random Networks out of the selection of elite individuals and returns them as a tuple.'''

    parent1 = selection[randint(0, len(selection)-1)]
    parent2 = selection[randint(0, len(selection)-1)]
    while parent1 is parent2:
        parent1 = selection[randint(0, len(selection)-1)]
        parent2 = selection[randint(0, len(selection)-1)]
    return (parent1, parent2)


def crossoverTensor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    '''Builds a new tensor by picking each element from 'a' or 'b' with 50/50 chance.'''
    fromA = torch.rand_like(a) < 0.5
    return torch.where(fromA, a, b)


def mutateTensor(value: torch.Tensor, sigma: float, mutationRate: float) -> torch.Tensor:
    '''Each element has 'mutationRate' chance to get gaussian noise (std 'sigma') added to it.'''
    mutate = torch.rand_like(value) < mutationRate
    noise = torch.randn_like(value) * sigma
    return value + noise * mutate


def breed(parent1: Network, parent2: Network, sigma: float, mutationRate: float) -> Network:
    '''Takes two 'parent'-networks and randomly chooses edges and biases from them to build a new 'child'-network from them.'''
    newLayers = []
    for (weights1, bias1), (weights2, bias2) in zip(parent1.layers, parent2.layers):
        newWeights = crossoverTensor(weights1, weights2)
        newWeights = mutateTensor(newWeights, sigma, mutationRate)
        newBias = crossoverTensor(bias1, bias2)
        newBias = mutateTensor(newBias, sigma, mutationRate)
        newLayers.append((newWeights, newBias))
    return Network(layers=newLayers, loss=None)


def buildPopulation(populationSize: int, selection: list[Network], sigma: float, mutationRate: float, eliteCount: int = 3) -> list[Network]:
    '''Builds a new generation of Networks breeded from the selection of the previous generation and returns it as a list.'''
    currPopCount = 0
    newPopulation = []

    # Saving elite individuals from mutation
    for i in range(0, eliteCount):
        currPopCount += 1
        newPopulation.append(selection[i])

    while currPopCount < populationSize:
        currPopCount += 1
        parents = pickRandomParents(selection)
        child = breed(parents[0], parents[1], sigma, mutationRate)
        newPopulation.append(child)

    return newPopulation


def trainingLoop(iterations: int, populationSize: int, mutationRate: float, sigma: float, eliteCount: int,
                  keepPartSelection: float, trainingData: list, startNetwork: Network | None = None,
                  inputSize: int | None = None, hiddenSizes: list[int] | None = None, outputSize: int | None = None) -> Network:
    '''Runs the training of a network fora given amount of 'iterations'.
    \n It handles:
    * population generation
    * evaluation
    * selection
    * breeding
    * mutation
    And when training is finished it returns the trained network.
    '''
    functionStart = time.perf_counter()
    print("Start of training loop!")
    print("Generating first population...")
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

    loopCount = 0
    while loopCount < iterations:
        start = time.perf_counter()
        loopCount += 1
        # evaluating each network on the trainingdata
        for i in range(0, populationSize):
            loss = evaluateNetwork(currPopulation[i], trainingData)
            currPopulation[i].loss = loss

        # store only the best performers
        best = selection(currPopulation, keepPartSelection)
        if loopCount < iterations:
            currPopulation = buildPopulation(populationSize, best, sigma, mutationRate, eliteCount)
        elapsed = time.perf_counter() - start
        print(f"Iteration #{loopCount} finished in {elapsed:.2f} seconds")

    bestNetwork = selection(currPopulation, 1.0)[0]
    print(f"Finished {iterations} iterations, returning trained network. ")
    totalTrainingTime = time.perf_counter()-functionStart
    print(f"Training finished in {totalTrainingTime:.2f} seconds.\nFinal loss score: {bestNetwork.loss}\n")
    return bestNetwork  # returns the fittest network after all iterations are done


if __name__ == "__main__":
    trainingData = generateTrainingData(1000)
    NetworkOne = trainingLoop(1000, 300, 0.1, 0.05, 3, 0.1, trainingData, inputSize=2, outputSize=1, hiddenSizes=[2])
    print(NetworkOne.loss)

    NetworkTwo = trainingLoop(1000, 300, 0.1, 0.05, 3, 0.1, trainingData, inputSize=2, outputSize=1, hiddenSizes=[5])
    print(NetworkTwo.loss)
    print(torch.cuda.is_available())