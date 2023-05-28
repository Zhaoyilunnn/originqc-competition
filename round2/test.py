import sys

import numpy as np
#import torch

from constants import HIDDEN_SIZE, INPUT_SIZE, NUM_CLASSES, BATCH_SIZE
from feature_extractor import get_train_data_v1, get_train_data_v2, get_train_data_v3
#from model import SimpleNet
#from torch.utils.data import DataLoader, TensorDataset
from train import train_v1, train_v2, train_v3
from utils import pca, to_one_hot
from vectorize import ForwardMaxMatch


#def train(x_train, y_train):
#
#    input_size = x_train.shape[1]
#    num_classes = NUM_CLASSES
#    hidden_size = HIDDEN_SIZE
#    batch_size = BATCH_SIZE
#
#    print("\n=============== Training start ====================\n")
#    print(f"input_size: {input_size}")
#    print(f"num_classes: {num_classes}")
#    print(f"hidden_size: {hidden_size}")
#
#    x_train = torch.from_numpy(x_train).float()
#    y_train = torch.from_numpy(y_train)
#    dataset = TensorDataset(x_train, y_train)
#    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#    net = SimpleNet(input_size, hidden_size, num_classes)
#    criterion = torch.nn.CrossEntropyLoss()
#    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
#
#    loss = None
#    for epoch in range(20):
#        for inputs, labels in dataloader:
#            optimizer.zero_grad()
#            y_pred = net(inputs)
#            loss = criterion(y_pred, labels)
#            loss.backward()
#            optimizer.step()
#        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 20, loss.item()))

def main():
    #tokenizer = ForwardMaxMatch('./sgns.weibo.word')

    #x_train, y_train = [], []
    #line_idx = 0
    #with open('./train.csv', 'r', encoding='utf-8') as f:
    #    for line in f:
    #        line_idx += 1
    #        if line_idx == 1:
    #            continue
    #        items = line.strip().split(',')
    #        label = int(items[0])
    #        sentence = ','.join(items[1:])
    #        words = tokenizer.cut(sentence)
    #        vec = tokenizer.to_vec(words)
    #        x_train.append(vec)
    #        y_train.append(label)

    #if x_train is [] or y_train is []:
    #    print("=========== Loading data failed =============")
    #    exit(1)

    #x_train = np.vstack(x_train)
    #x_train = pca(x_train, INPUT_SIZE)
    #y_train = np.array(y_train)
    #y_train = to_one_hot(y_train)

    #print("\n================= Loading data finished ===================\n")
    #print("Train data shape:\n")
    #print("x_train: {}".format(x_train.shape))
    #print("y_train: {}".format(y_train.shape))

    #x_train, y_train = get_train_data_v1()
    #x_train, y_train = get_train_data_v2()
    x_train, y_train = get_train_data_v3()
    x_train = np.transpose(x_train, (0,2,1))

    train_v3(x_train, y_train)


if __name__ == '__main__':
    main()
