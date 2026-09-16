from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor
import torch
from network import buildNetwork, DEVICE
from training import trainingLoop
import os
from datetime import datetime


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
    hiddenSizesHunter = [256, 256, 128, 64, 32]
    hiddenSizesLover = [256, 256, 256, 256, 256]
    outputSize = 1

    iterations = 100000
    populationSize = 500
    populationSizeLover = 25  # FrogLover has ~1.9x FrogHunter's parameter count, scaled down for memory headroom
    mutationRate = 0.05
    sigma = 0.1
    eliteCount = 3
    keepPart = 0.7
    trainingData = generateCifarTrainingData("frog", dataPointsPerClass=1000)

    # two different Networks with different architectures
    untrainedFrogHunter = buildNetwork(inSize, hiddenSizesHunter, outputSize)
    untrainedFrogLover = buildNetwork(inSize, hiddenSizesLover, outputSize)

    #frogHunterT = trainingLoop(iterations, populationSize, mutationRate, sigma, eliteCount, keepPart, trainingData, untrainedFrogHunter, inSize, hiddenSizesHunter, outputSize, "FrogHunter")
    #frogHunter = frogHunterT[0]
    #frogHunterLog = frogHunterT[1]
    #plt.plot(frogHunterLog, label="FrogHunter")

    torch.cuda.empty_cache()
    frogLoverTuple = trainingLoop(iterations, populationSizeLover, mutationRate, sigma, eliteCount, keepPart, trainingData, untrainedFrogLover, inSize, hiddenSizesLover, outputSize, "FrogLover")
    frogLover = frogLoverTuple[0]
    frogLoverLog = frogLoverTuple[1]
    torch.save(frogLoverLog, f"visualizations/frogLoverLog_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pt")
    plt.plot(frogLoverLog, label="FrogLover")
    plt.legend()

    os.makedirs('visualizations', exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"visualizations/frog_accuracy_{timestamp}.png")
    