from __future__ import annotations
from dataclasses import dataclass
import typing
import torch



@dataclass
class Network:
    '''
    Represents a fully connected neural network as a stack of (weights, bias)
    tensor pairs, one pair per layer, going from the first hidden layer to
    the output layer.

    - 'layers': a list of tuples, each holding one layer's weight matrix
    (shape [out_features, in_features]) and bias vector (shape [out_features]).
    - 'loss': the network's average loss over a training set, set externally
    after evaluation. 'None' until evaluated for the first time.
    - 'fitness': derived from 'loss' (1 / (1 + loss)), or 'None' if the network
      hasn't been evaluated yet. Higher is better, bounded in (0, 1).
    '''

    layers: list[tuple[torch.Tensor, torch.Tensor]]
    loss: typing.Optional[float] = None

    @property
    def fitness(self):
        if self.loss is not None:
            return 1/(1 + self.loss)
        return None

    def forwardPass(self, inputs: torch.Tensor):
        if inputs.size()[0] != self.layers[0][0].size()[1]:
            raise ValueError(f"Given Inputs do not match the required input size. The Network has an inputSize of {self.layers[0][0].size()[1]}.")
        # start forward pass with the input layer
        outputPrev = inputs
        for idx in range(0, len(self.layers)):
            edgeWeights = self.layers[idx][0]
            bias = self.layers[idx][1]
            output = torch.relu(edgeWeights @ outputPrev + bias)
            outputPrev = output
        return outputPrev


def buildNetwork(inputSize: int, hiddenSizes: list[int], outputSize: int):
    '''
    Receives the dimensions and returns the new Network as a list of tuples containing 
    edgeWeight- and Bias-tensors with randomly generated values.
    '''

    # building input Layer
    newNetwork = list()
    inputWeights = torch.rand([hiddenSizes[0],inputSize], device="cuda:0")
    inputBiases = torch.rand([hiddenSizes[0]], device="cuda:0")
    newNetwork.append((inputWeights, inputBiases))

    # building hidden Layers
    for i in range(1,len(hiddenSizes)):
        newWeights = torch.rand(hiddenSizes[i], hiddenSizes[i-1], device="cuda:0")
        newBiases = torch.rand([hiddenSizes[i]], device="cuda:0")
        newNetwork.append((newWeights, newBiases))

    # building output layer
    newNetwork.append((torch.rand([outputSize, hiddenSizes[-1]], device="cuda:0"), torch.rand([outputSize], device="cuda:0")))
    return Network(layers=newNetwork, loss=None)
