from __future__ import annotations
from dataclasses import dataclass
import typing
import torch


if torch.cuda.is_available():
    DEVICE = "cuda:1"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
# "cuda:1" == Tesla P40
# "cuda:0" == GTX 1660
# "cpu" == on macbook


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
    accuracy: typing.Optional[float] = None
    score: typing.Optional[float] = None

    @property
    def fitness(self):
        if self.loss is not None:
            return 1/(1 + self.loss)
        return None
    
    def forwardPass(self, inputs: torch.Tensor):
        if inputs.size()[1] != self.layers[0][0].size()[1]:
            raise ValueError(f"Given Inputs do not match the required input size. The Network has an inputSize of {self.layers[0][0].size()[1]}.")
        outputPrev = inputs
        for idx in range(0, len(self.layers)-1):
            # for each layer multiply the previous outputMatrices and the weightMatrices and add the biastensor 
            output = torch.relu(outputPrev @ self.layers[idx][0].T + self.layers[idx][1])
            outputPrev = output
        return torch.sigmoid(outputPrev @ self.layers[len(self.layers)-1][0].T + self.layers[len(self.layers)-1][1])


def batchForwardPass(networks: list[Network], inputs: torch.Tensor) -> torch.Tensor:
    '''
    Runs forwardPass for an entire population of Networks at once, instead of once per Network.
    All Networks in 'networks' must share the same architecture (same layer shapes).
    Returns a tensor of shape [populationSize, batchSize, outputSize].
    '''
    if inputs.size()[1] != networks[0].layers[0][0].size()[1]:
        raise ValueError(f"Given Inputs do not match the required input size. The Networks have an inputSize of {networks[0].layers[0][0].size()[1]}.")

    popSize = len(networks)
    outputPrev = inputs.expand(popSize, -1, -1)  # [popSize, batch, inFeatures] - broadcast view, no copy

    for idx in range(0, len(networks[0].layers) - 1):
        popWeights = torch.stack([n.layers[idx][0] for n in networks])  # [popSize, out, in]
        popBiases = torch.stack([n.layers[idx][1] for n in networks])   # [popSize, out]
        output = torch.relu(outputPrev @ popWeights.transpose(1, 2) + popBiases.unsqueeze(1))
        outputPrev = output

    lastIdx = len(networks[0].layers) - 1
    popWeights = torch.stack([n.layers[lastIdx][0] for n in networks])
    popBiases = torch.stack([n.layers[lastIdx][1] for n in networks])
    return torch.sigmoid(outputPrev @ popWeights.transpose(1, 2) + popBiases.unsqueeze(1))


def buildNetwork(inputSize: int, hiddenSizes: list[int], outputSize: int):
    '''
    Receives the dimensions and returns the new Network as a list of tuples containing 
    edgeWeight- and Bias-tensors with randomly generated values.
    '''

    # building input Layer
    newNetwork = list()
    inputWeights = torch.empty([hiddenSizes[0],inputSize], device=DEVICE)
    torch.nn.init.kaiming_uniform_(inputWeights, nonlinearity='relu')
    inputBiases = torch.zeros([hiddenSizes[0]], device=DEVICE)
    newNetwork.append((inputWeights, inputBiases))

    # building hidden Layers
    for i in range(1,len(hiddenSizes)):
        newWeights = torch.empty(hiddenSizes[i], hiddenSizes[i-1], device=DEVICE)
        torch.nn.init.kaiming_uniform_(newWeights, nonlinearity='relu')
        newBiases = torch.zeros([hiddenSizes[i]], device=DEVICE)
        newNetwork.append((newWeights, newBiases))

    # building output layer
    outputWeights = torch.empty([outputSize, hiddenSizes[-1]], device=DEVICE)
    torch.nn.init.xavier_uniform_(outputWeights)
    newNetwork.append((outputWeights, torch.zeros([outputSize], device=DEVICE)))
    return Network(layers=newNetwork, loss=None)
