from pyvqnet import optim
import pyvqnet
from pyvqnet.nn import Sigmoid, ReLu
import pyvqnet.nn as nn
from pyvqnet.qnn.qdrl.vqnet_model import qdrl_circuit
from pyvqnet.qnn.quantumlayer import QuantumLayer
from pyvqnet.optim import adam
from pyvqnet.nn.loss import CategoricalCrossEntropy
from pyvqnet.tensor import QTensor
import numpy as np
from pyvqnet.nn.module import Module
from pyvqnet.nn.activation import *
from QLSTM import QLSTM

from constants import *


param_num = PARAM_NUM
qbit_num = QUBIT_NUM
num_classes = NUM_CLASSES


class ModelV1(Module):

    def __init__(self) -> None:
        super(ModelV1, self).__init__()

        # Quantum layer
        self.pqc = QuantumLayer(qdrl_circuit, param_num, "cpu", qbit_num)

        # Classical fully connected layer
        self.fn = pyvqnet.nn.Linear(2**qbit_num, num_classes)

    def forward(self, x):
        x = self.pqc(x)
        x = self.fn(x)
        return x


class ModelV2(Module):

    def __init__(self) -> None:
        super(ModelV2, self).__init__()

        self.conv = pyvqnet.nn.Conv2D(1, 1, (300,1))
        self.fn1 = pyvqnet.nn.Linear(180, 4)
        # Quantum layer
        self.pqc = QLSTM(4, 4)
        self.relu = ReLu()

        # Classical fully connected layer
        self.fn2 = pyvqnet.nn.Linear(4, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.fn1(x)
        x = self.relu(x)
        x = self.fn2(x)
        return x


class ModelV3(nn.Module):
    """
    constructor
    :param input_sz: num of features
    :param hidden_sz: num of hidden neurons
    """

    def __init__(self, input_sz, hidden_sz):
        super().__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.rnn = QLSTM(input_sz, hidden_sz)
        self.regression = nn.Linear(hidden_sz, num_classes)

    def forward(self, x):
        """Assumes x is of shape (batch, sequence, feature)"""
        output, (h_n, _) = self.rnn(x)
        # return self.fc(output.transpose(0, 1)[-1])
        output = self.regression(h_n)
        return output
