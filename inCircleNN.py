import os
from datetime import datetime

import torch
from network import Network, DEVICE
from training import generateTrainingData, trainingLoop
import math
import matplotlib.pyplot as plt


def inCircle(inx, iny) -> float:
    '''
    Circle center at x=2, y=1, radius = 0.4
    Takes two coordinates x and y and returns a binary truth value 1.0 or 0.0, 
    reflecting if that point is inside the circle (True/1.0).
    '''
    dist = math.sqrt(((inx-2)**2) + (iny-1)**2)
    if dist <= 0.4:
        return 1.0
    return 0.0


def visualize(xBounds: tuple[float, float], yBounds: tuple[float, float], pixelCount: int, circle: tuple[tuple[float, float], float], network: Network):
    xMin = xBounds[0]
    xMax = xBounds[1]
    yMin = yBounds[0]
    yMax = yBounds[1]
    xSpace = torch.linspace(xMin, xMax, pixelCount, device=DEVICE)
    ySpace = torch.linspace(yMin, yMax, pixelCount, device=DEVICE)
    X, Y = torch.meshgrid(xSpace, ySpace, indexing='ij') # creating the grid of pixels in the needed resolution
    prediction = network.forwardPass(torch.stack([X.flatten(), Y.flatten()], dim=1)) # running  forwardPass over the grid. Returns a prediction for EACH Pixel in the image
    prediction = prediction.reshape(pixelCount, pixelCount).cpu()

    fig, ax = plt.subplots()
    ax.imshow(prediction, extent=[xMin, xMax, yMin, yMax], origin='lower', cmap='viridis')
    circlePlot = plt.Circle((circle[0]), circle[1], fill=False, color='red', linewidth=2)
    ax.add_patch(circlePlot)
    os.makedirs('visualizations', exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"visualizations/circle_viz_{timestamp}.png")


if __name__ == "__main__":
    trainingData = generateTrainingData(inCircle, (1.5, 2.5), (0.5, 1.5), 5000)
    trainingIterations = 1000
    populationSize = 300
    circle = ((2.0, 1.0), 0.4)
    xBounds=(1.5, 2.5)
    yBounds=(0.5, 1.5)
    pixelCount = 200

    in1 = 2
    out1=1
    hidden1 =[7,4]
    NetworkOne = trainingLoop(trainingIterations, populationSize, 0.1, 0.05, 3, 0.1, trainingData,
                              inputSize=in1, hiddenSizes=hidden1, outputSize=out1, networkName="Network #1")
    visualize(xBounds, yBounds, pixelCount, circle, NetworkOne)
    
    in2 = 2
    out2 = 1
    hidden2 = [50]
    NetworkTwo = trainingLoop(trainingIterations, populationSize, 0.1, 0.05, 3, 0.1, trainingData,
                              inputSize=in2, hiddenSizes=hidden2, outputSize=out2, networkName="Network #2")
    visualize(xBounds, yBounds, pixelCount, circle, NetworkTwo) 