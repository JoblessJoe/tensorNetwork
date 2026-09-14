import torch
from network import Network
from training import generateTrainingData, trainingLoop
import math
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    gpu = "cuda:1"
elif torch.backends.mps.is_available():
    gpu = "mps"
else:
    gpu = "cpu"
# "cuda:1" == Tesla P40
# "cuda:0" == GTX 1660
# "cpu" == on macbook


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
    xSpace = torch.linspace(xMin, xMax, pixelCount, device=gpu)
    ySpace = torch.linspace(yMin, yMax, pixelCount, device=gpu)
    X, Y = torch.meshgrid(xSpace, ySpace) # creating the grid of pixels in the needed resolution
    prediction = network.forwardPass(torch.stack([X.flatten(), Y.flatten()], dim=1)) # running  forwardPass over the grid. Returns a prediction for EACH Pixel in the image
    prediction = prediction.reshape(pixelCount, pixelCount)

    fig, ax = plt.subplots()
    ax.imshow(prediction, extent=[xMin, xMax, yMin, yMax], origin='lower', cmap='viridis')
    circlePlot = plt.Circle((circle[0]), circle[1], fill=False, color='red', linewidth=2)
    ax.add_patch(circlePlot)
    plt.savefig('circle_viz.png')  # not plt.show() -- server has no display
    plt.show()


if __name__ == "__main__":
    trainingData = generateTrainingData(inCircle, (1.5, 2.5), (0.5, 1.5), 5000)
    trainingIterations = 2500
    populationSize = 300
    circle = ((1.5,2.5), (2, 1))
    xBounds=(1.5, 2.5)
    yBounds=(0.5, 1.5)
    pixelCount = 200

    in1 = 2
    out1=1
    hidden1 =[2,2]
    NetworkOne = trainingLoop(trainingIterations, populationSize, 0.1, 0.05, 3, 0.1, trainingData,
                              inputSize=in1, hiddenSizes=hidden1, outputSize=out1, networkName="Network #1")
    visualize(xBounds, yBounds, pixelCount, circle, NetworkOne)
    
    in2 = 2
    out2 = 1
    hidden2 = [5]
    NetworkTwo = trainingLoop(trainingIterations, populationSize, 0.1, 0.05, 3, 0.1, trainingData,
                              inputSize=in2, hiddenSizes=hidden2, outputSize=out2, networkName="Network #2")
    visualize(xBounds, yBounds, pixelCount, circle, NetworkTwo)