import torch
import torch.nn as nn

import pyvqnet
from pyvqnet.qnn.qdrl.vqnet_model import qdrl_circuit
from pyvqnet.qnn.quantumlayer import QuantumLayer
from pyvqnet.optim import adam
from pyvqnet.nn.loss import CategoricalCrossEntropy
from pyvqnet.tensor import QTensor
import numpy as np
from pyvqnet.nn.module import Module


class SimpleNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class QNet(pyvqnet.nn.Module):

    pass
