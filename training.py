from __future__ import annotations
import math
from random import uniform, randint
import time
import typing
import torch
from tqdm import tqdm
from network import Network, buildNetwork, DEVICE, batchForwardPass


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


def evaluateNetwork(networks: list[Network], trainingData: torch.Tensor, target: torch.Tensor) -> list[Network]:
    '''
    Takes a tensor of TrainingPoints and a Network and returns the cross entropy loss and the accuracy of the network as a tuple.
    Output: (loss, accuracy)
    '''
    res = batchForwardPass(networks, trainingData).squeeze(-1) # Result Tensor for the predictions for each trainingPoint the network made
    for i in range(0, len(networks)):
        networks[i].loss = crossEntropyLoss(target, res[i])
        networks[i].accuracy = getAccuracy(res[i], target)
    return networks


def selection(networks: list[Network], keepPart: float) -> list[Network]:
    '''Takes a list of Networks after Training and keep only the top percentage chosen by 'keepPart' and returns this elite selection as a list'''
    fitness = []
    for n in networks:
        fitness += [n.fitness]
    fTensor = torch.stack(fitness)
    split = int(len(networks) * keepPart)
    sel = torch.topk(fTensor, split)
    result = [networks[el] for el in sel.indices]
    return result


def pickRandomParents(selection: list[Network], populationSize: int, eliteCount: int) -> tuple[Network]:
    '''Returns a Tensor of randomly picked parents.'''
    parents = torch.stack([torch.randint(0, len(selection), (populationSize -eliteCount,)), torch.randint(0, len(selection), (populationSize -eliteCount,))])
    return parents


def crossoverTensor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    '''Builds a new tensor by picking each element from 'a' or 'b' with 50/50 chance.'''
    fromA = torch.rand_like(a, device=DEVICE) < 0.5
    return torch.where(fromA, a, b)


def mutateTensor(value: torch.Tensor, sigma: float, mutationRate: float) -> torch.Tensor:
    '''Each element has 'mutationRate' chance to get gaussian noise (std 'sigma') added to it.'''
    mutate = torch.rand_like(value, device=DEVICE) < mutationRate
    noise = torch.randn_like(value, device=DEVICE) * sigma
    return value + noise * mutate


def breed(selection: list[Network], parents: torch.Tensor, sigma: float, mutationRate: float) -> list[Network]:
    '''Takes a tensor of the 'to breed-/parent'-networks and randomly chooses edges and biases from them and returns a list of 'child'-networks.
    '''

    newGen = []
    for i in range(0, len(selection[0].layers)):
        selWeights = torch.stack([n.layers[i][0] for n in selection]) # gets the weights for layer 'i' of all the selection networks and makes them into a tensor
        selBiases = torch.stack([n.layers[i][1] for n in selection])
        parentsAW = selWeights[parents[0]] # extracting only the to breed networks for that layer and handing it to the crossoverTensor function 
        parentsABias = selBiases[parents[0]]
        parentsBW = selWeights[parents[1]]
        parentsBBias = selBiases[parents[1]]
        newLayerWeights = mutateTensor(crossoverTensor(parentsAW, parentsBW), sigma, mutationRate)
        newLayerBiases = mutateTensor(crossoverTensor(parentsABias, parentsBBias), sigma, mutationRate)
        newGen += [(newLayerWeights, newLayerBiases)]

    # Now rebuilding a list of networks out of the batched layered 'newGen' variable
    networks = []
    numChildren = parents.shape[1]
    children = [[] for _ in range(numChildren)]
    for layerIdx in range(0, len(selection[0].layers)):
        for childIdx in range(0, numChildren):
            weights = newGen[layerIdx][0][childIdx]
            bias = newGen[layerIdx][1][childIdx]
            children[childIdx] += [(weights, bias)]

    for child in children:
        networks += [Network(child)]

    return networks


def buildPopulation(populationSize: int, selection: list[Network], sigma: float, mutationRate: float, eliteCount: int = 3) -> list[Network]:
    '''Builds a new generation of Networks breeded from the selection of the previous generation and returns it as a list.'''

    parents = pickRandomParents(selection, populationSize, eliteCount)
    newPopulation = breed(selection, parents, sigma, mutationRate)

    # Saving elite individuals from mutation and breeding
    for i in range(0, eliteCount):
        newPopulation.append(selection[i])
    return newPopulation


def trainingLoop(iterations: int, populationSize: int, mutationRate: float, sigma: float, eliteCount: int,
                keepPartSelection: float, trainingData: tuple[torch.Tensor, torch.Tensor], startNetwork: Network | None = None,
                inputSize: int | None = None, hiddenSizes: list[int] | None = None, outputSize: int | None = None,
                networkName: str = "network") -> Network:
    '''Runs the training of a network for a given amount of 'iterations'.
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
    popTime = 0.0
    sortTime = 0.0
    evalTime = 0.0
    with tqdm(total=iterations, desc=networkName, unit="gen") as pbar:
        while loopCount < iterations:
            start = time.perf_counter()
            loopCount += 1

            # evaluating each network on the trainingdata
            evalStart=time.perf_counter()
            currPopulation = evaluateNetwork(currPopulation, trainingData[0], trainingData[1])
            torch.cuda.synchronize()
            evalTime += time.perf_counter() - evalStart

            # store only the best performers
            sortStart = time.perf_counter()
            best = selection(currPopulation, keepPartSelection)
            torch.cuda.synchronize()
            sortTime += time.perf_counter() - sortStart
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
    totalTrainingTime = time.perf_counter() - functionStart
    popTime = popTime/loopCount
    evalTime = evalTime/loopCount
    sortTime = sortTime/loopCount
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
        f"Avg. EvalTime:       {evalTime:.3f} seconds\n"
        f"Avg. PopTime:        {popTime:.3f} seconds\n"
        f"Avg SortTime:        {sortTime:.3f} seconds\n"
        f"Training time:       {totalTrainingTime:.3f} seconds\n"
        f"Final loss:          {bestNetwork.loss.item():.4f}\n"
        f"Accuracy:            {bestNetwork.accuracy.item():.2%}\n"
        f"{'=' * 44}\n"
    )
    return bestNetwork  # returns the fittest network after all iterations are done
