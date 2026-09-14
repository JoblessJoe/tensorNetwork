from __future__ import annotations
import math
from random import uniform, randint
import time
import typing
import torch
from tqdm import tqdm
from network import Network, buildNetwork, DEVICE


def generateTrainingData(criterium: typing.Callable[[float, float], float], xBounds: tuple[float, float], yBounds: tuple[float, float], dataPoints=1000):
    '''
    Generates two tensors:
        - the tuples of points (x,y)
        - the binary truth values for those points (criteriumMet)
    and returns them as a tuple of Tensors, by the shape [pointsTensor, criteriumMetTensor]
    By default it generates a list with a 1000 data points, but that number can be chosen individually.
    'criterium' needs to be a callable that takes two floats and returns a float.
    '''
    if not (100 <= dataPoints <= 100000):
        raise ValueError(f"dataPoints value must be between 100 and 100000. Your value: {dataPoints}")
    points = []
    targets = []
    for i in range(0, dataPoints):
        xVal = uniform(xBounds[0], xBounds[1])
        yVal = uniform(yBounds[0], yBounds[1])
        criteriumMet = criterium(xVal, yVal)
        points.append((xVal, yVal))
        targets.append(criteriumMet)
    pointsTensor = torch.tensor(points, device=DEVICE)
    targetsTensor = torch.tensor(targets, device=DEVICE)
    return (pointsTensor, targetsTensor)


def crossEntropyLoss(target: torch.Tensor, prediction: torch.Tensor) -> float:
    '''Binary cross entropy between a 0/1 target and the network's (scalar tensor) prediction.'''
    pred = prediction.clamp(1e-7, 1 - 1e-7)  # forwardPass ends in ReLU, not a bounded probability, so clamp to keep log() finite
    loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    return loss.mean()

def getAccuracy(predictions: torch.Tensor, target: torch.Tensor) -> float:
    '''
    Returns the Accuracy of a networks predictions as a float between 0.0 and 1.0.
    '''
    correct = (predictions.round() == target).sum()
    total = len(predictions)
    return correct / total


def evaluateNetwork(network: Network, trainingData: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    '''
    Takes a tensor of TrainingPoints and a Network and returns the cross entropy loss and the accuracy of the network as a tuple.
    Output: (loss, accuracy)
    '''
    res = network.forwardPass(trainingData).squeeze(1) # Result Tensor for the predictions for each trainingPoint the network made
    loss = crossEntropyLoss(target, res)
    accuracy = getAccuracy(res, target)
    return (loss, accuracy)


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
    fromA = torch.rand_like(a, device=DEVICE) < 0.5
    return torch.where(fromA, a, b)


def mutateTensor(value: torch.Tensor, sigma: float, mutationRate: float) -> torch.Tensor:
    '''Each element has 'mutationRate' chance to get gaussian noise (std 'sigma') added to it.'''
    mutate = torch.rand_like(value, device=DEVICE) < mutationRate
    noise = torch.randn_like(value, device=DEVICE) * sigma
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
                keepPartSelection: float, trainingData: tuple[torch.Tensor, torch.Tensor], startNetwork: Network | None = None,
                inputSize: int | None = None, hiddenSizes: list[int] | None = None, outputSize: int | None = None,
                networkName: str = "network") -> Network:
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
    avgIterTime = 0
    evalTime = 0.0
    popTime = 0.0
    bestTime = 0.0
    with tqdm(total=iterations, desc=networkName, unit="gen") as pbar:
        while loopCount < iterations:
            start = time.perf_counter()
            loopCount += 1
            # evaluating each network on the trainingdata
            for i in range(0, populationSize):
                evalStart = time.perf_counter()
                evaluation = evaluateNetwork(currPopulation[i], trainingData[0], trainingData[1])
                torch.cuda.synchronize()
                evalTime += time.perf_counter()-evalStart
                loss = evaluation[0]
                accuracy = evaluation[1]
                currPopulation[i].loss = loss
                currPopulation[i].accuracy = accuracy

            # store only the best performers
            bestStart = time.perf_counter()
            best = selection(currPopulation, keepPartSelection)
            bestTime += time.perf_counter() - bestStart
            if loopCount < iterations:
                startPopBuild = time.perf_counter()
                currPopulation = buildPopulation(populationSize, best, sigma, mutationRate, eliteCount)
                torch.cuda.synchronize()
                popTime += time.perf_counter()-startPopBuild
            elapsed = time.perf_counter() - start
            avgIterTime += elapsed

            pbar.set_postfix(loss=f"{best[0].loss:.4f}", acc=f"{best[0].accuracy:.2%}")
            pbar.update(1)

    avgIterTime = avgIterTime / loopCount
    bestNetwork = selection(currPopulation, 1.0)[0]
    totalTrainingTime = time.perf_counter()-functionStart
    popTime = popTime/loopCount
    evalTime = evalTime/loopCount
    bestTime = bestTime/loopCount
    print(
        f"\n{'=' * 44}\n"
        f"{networkName:^44}\n"
        f"{'=' * 44}\n"
        f"Input size:          {inputSize}\n"
        f"Hidden sizes:        {hiddenSizes}\n"
        f"Output size:         {outputSize}\n"
        f"Population size:     {populationSize}\n"
        f"Training iterations: {iterations}\n"
        f"Average iteration:   {avgIterTime:.4f} seconds\n"
        f"Avg. EvalTime:       {evalTime:.2f} seconds\n"
        f"Avg. PopTime:        {popTime:.2f} seconds\n"
        f"Avg BestTime:        {bestTime:.2f} seconds\n"
        f"Training time:       {totalTrainingTime:.2f} seconds\n"
        f"Final loss:          {bestNetwork.loss.item():.4f}\n"
        f"Accuracy:            {bestNetwork.accuracy.item():.2%}\n"
        f"{'=' * 44}\n"
    )
    return bestNetwork  # returns the fittest network after all iterations are done
