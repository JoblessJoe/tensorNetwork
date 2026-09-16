from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor
from datetime import datetime
import torch
from network import Network, buildNetwork, DEVICE
from training import trainingLoop
import math
import matplotlib.pyplot as plt

'''
Plan here:
    - network gets a 32x32 image
        ==> inputSize = 32*32*3 = 3072
    - returns 1.0 for class Frog and 0.0 for everything else

We need:

'''


def generateCifarTrainingData(targetClass: str, dataPointsPerClass: int = 1000, datasetRoot: str = "./data") -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Builds a class-balanced binary training set from CIFAR-10: 'dataPointsPerClass' images of
    'targetClass' (target 1.0) and 'dataPointsPerClass' images drawn from all other classes
    combined (target 0.0). Each image is flattened from [3,32,32] to a [3072] tensor.
    Returns (pointsTensor [2*dataPointsPerClass, 3072], targetsTensor [2*dataPointsPerClass]),
    same shape convention as generateTrainingData.
    '''
    dataset = CIFAR10(root=datasetRoot, train=True, download=True)
    targetIdx = dataset.classes.index(targetClass)

    positives = []
    negatives = []
    for img, label in dataset:
        if label == targetIdx and len(positives) < dataPointsPerClass:
            positives.append(to_tensor(img).flatten())
        elif label != targetIdx and len(negatives) < dataPointsPerClass:
            negatives.append(to_tensor(img).flatten())
        if len(positives) >= dataPointsPerClass and len(negatives) >= dataPointsPerClass:
            break

    pointsTensor = torch.stack(positives + negatives).to(DEVICE)
    targetsTensor = torch.tensor([1.0] * len(positives) + [0.0] * len(negatives), device=DEVICE)
    return (pointsTensor, targetsTensor)


if __name__ == "__main__":
    pixelCount = 32 * 32
    perPixelVal = 3
    inSize = pixelCount * perPixelVal  # == 3072 input neurons
    hiddenSizesHunter = [2000, 1000, 750, 500, 250, 100, 50, 10]
    hiddenSizesLover = [2500, 2000, 2000, 500, 250, 50]
    outputSize = 1

    iterations = 500
    populationSize = 50
    mutationRate = 0.05
    sigma = 0.1
    eliteCount = 3
    keepPart = 0.7
    trainingData = generateCifarTrainingData("frog", dataPointsPerClass=1000)

    # two different Networks with different architectures
    untrainedFrogHunter = buildNetwork(inSize, hiddenSizesHunter, outputSize)
    untrainedFrogLover = buildNetwork(inSize, hiddenSizesLover, outputSize)

    frogHunter = trainingLoop(iterations, populationSize, mutationRate, sigma, eliteCount, keepPart, trainingData, untrainedFrogHunter, inSize, hiddenSizesHunter, outputSize, "FrogHunter")
    frogLover = trainingLoop(iterations, populationSize, mutationRate, sigma, eliteCount, keepPart, trainingData, untrainedFrogLover, inSize, hiddenSizesLover, outputSize, "FrogLover")
     

    