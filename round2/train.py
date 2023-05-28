import numpy as np

from pyvqnet.optim import adam
from pyvqnet.nn.loss import CategoricalCrossEntropy
from quantum_model import ModelV1, ModelV2, ModelV3
from utils import get_minibatch_data, get_score
from constants import BATCH_SIZE, EPOCH, LEARNING_RATE, NUM_CLASSES


model_v1 = ModelV1()
model_v2 = ModelV2()
model_v3 = ModelV3(8, 8)
optimizer_v1 = adam.Adam(model_v1.parameters(), lr=LEARNING_RATE)
optimizer_v2 = adam.Adam(model_v2.parameters(), lr=LEARNING_RATE)
optimizer_v3 = adam.Adam(model_v3.parameters(), lr=LEARNING_RATE)
loss_func = CategoricalCrossEntropy()
batch_size = BATCH_SIZE
epoch = EPOCH


def train_v1(x_train, y_train):
    x_train = np.hstack((x_train, np.zeros((x_train.shape[0], 1))))
    for i in range(epoch):
        model_v1.train()
        accuracy = 0
        count = 0
        loss = 0
        batch_idx = 0
        for data, label in get_minibatch_data(x_train, y_train, batch_size):
            optimizer_v1.zero_grad()
            output = model_v1(data)
            losses = loss_func(label, output)
            losses.backward()
            optimizer_v1._step()
            accuracy += get_score(output, label)
            loss += losses.item()
            count += batch_size

            print(f"epoch:{i}, batch:{batch_idx}, acc:{accuracy/count}")
            batch_idx += 1

        print(f"epoch:{i}, train_accuracy:{accuracy/count}")
        print(f"epoch:{i}, train_loss:{loss/count}\n")

def train_v2(x_train, y_train):
    for i in range(epoch):
        model_v2.train()
        accuracy = 0
        count = 0
        loss = 0
        batch_idx = 0
        for data, label in get_minibatch_data(x_train, y_train, batch_size):
            assert len(data.shape) == 3
            data = np.reshape(data, (data.shape[0], 1, data.shape[1], data.shape[2]))
            optimizer_v2.zero_grad()
            output = model_v2(data)
            output = output.reshape((batch_size, -1))
            losses = loss_func(label, output)
            losses.backward()
            optimizer_v2._step()
            accuracy += get_score(output, label)
            loss += losses.item()
            count += batch_size

            print(f"epoch:{i}, batch:{batch_idx}, acc:{accuracy/count}")
            batch_idx += 1

        print(f"epoch:{i}, train_accuracy:{accuracy/count}")
        print(f"epoch:{i}, train_loss:{loss/count}\n")


def train_v3(x_train, y_train):
    for i in range(epoch):
        model_v3.train()
        accuracy = 0
        count = 0
        loss = 0
        batch_idx = 0
        for data, label in get_minibatch_data(x_train, y_train, batch_size):
            optimizer_v3.zero_grad()
            output = model_v3(data)
            losses = loss_func(label, output)
            losses.backward()
            optimizer_v3._step()
            accuracy += get_score(output, label)
            loss += losses.item()
            count += batch_size

            print(f"epoch:{i}, batch:{batch_idx}, acc:{accuracy/count}")
            batch_idx += 1

        print(f"epoch:{i}, train_accuracy:{accuracy/count}")
        print(f"epoch:{i}, train_loss:{loss/count}\n")
