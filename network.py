from dataclasses import dataclass
import torch


@dataclass
class Neuron:
    '''
    One neuron inside a neural network. has the attributes:
    - 'bias'
    - 'type' of Neuron (either INPUT, HIDDEN or OUTPUT)
    - 'valIn' -> representing the sum of the inputs multiplied by their respective edge weights and added the bias
    - 'valOut' -> output value after activation function ran over valIn (eiother ReLu or sigmoid)
    - 'inputs' -> a list of 'Edge' coming into the neuron
    - 'outputs' -> list of 'Edge' going away from the neuron (using class 'Edge' with edge weight and origin and source neurons)
    '''