from training import generateTrainingData, trainingLoop
import math


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


if __name__ == "__main__":
    trainingData = generateTrainingData(inCircle, (1.5, 2.5), (0.5, 1.5), 5000)
    trainingIterations = 2500
    populationSize = 300

    in1 = 2
    out1=1
    hidden1 =[2,2]
    NetworkOne = trainingLoop(trainingIterations, populationSize, 0.1, 0.05, 3, 0.1, trainingData,
                              inputSize=in1, hiddenSizes=hidden1, outputSize=out1, networkName="Network #1")

    in2 = 2
    out2 = 1
    hidden2 = [5]
    NetworkTwo = trainingLoop(trainingIterations, populationSize, 0.1, 0.05, 3, 0.1, trainingData,
                              inputSize=in2, hiddenSizes=hidden2, outputSize=out2, networkName="Network #2")
