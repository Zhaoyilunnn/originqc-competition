import numpy as np

from constants import HIDDEN_SIZE, INPUT_SIZE, NUM_CLASSES, BATCH_SIZE
#from model import SimpleNet
#from torch.utils.data import DataLoader, TensorDataset
from utils import pca, to_one_hot
from vectorize import ForwardMaxMatch


def get_train_data_v1():
    tokenizer = ForwardMaxMatch('./sgns.weibo.word')

    x_train, y_train = [], []
    line_idx = 0
    with open('./train.csv', 'r', encoding='utf-8') as f:
        for line in f:
            line_idx += 1
            if line_idx == 1:
                continue
            items = line.strip().split(',')
            label = int(items[0])
            sentence = ','.join(items[1:])
            words = tokenizer.cut(sentence)
            vec = tokenizer.to_one_dim_vec(words)
            x_train.append(vec)
            y_train.append(label)

    if x_train is [] or y_train is []:
        print("=========== Loading data failed =============")
        exit(1)

    x_train = np.vstack(x_train)
    x_train = pca(x_train, INPUT_SIZE)
    y_train = np.array(y_train)
    y_train = to_one_hot(y_train)

    print("\n================= Loading data finished ===================\n")
    print("Train data shape:\n")
    print("x_train: {}".format(x_train.shape))
    print("y_train: {}".format(y_train.shape))

    return x_train, y_train


def get_train_data_v2():
    tokenizer = ForwardMaxMatch('./sgns.weibo.word')

    x_train, y_train = [], []
    line_idx = 0
    words_list = []
    label_list = []
    with open('./train.csv', 'r', encoding='utf-8') as f:
        for line in f:
            line_idx += 1
            if line_idx == 1:
                continue
            items = line.strip().split(',')
            label = int(items[0])
            label_list.append(label)
            sentence = ','.join(items[1:])
            words = tokenizer.cut(sentence)
            words_list.append(words)

    for words in words_list:
        vec = tokenizer.to_two_dim_vec(words)
        x_train.append(vec)
    for label in label_list:
        y_train.append(label)

    if x_train is [] or y_train is []:
        print("=========== Loading data failed =============")
        exit(1)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = to_one_hot(y_train)

    print("\n================= Loading data finished ===================\n")
    print("Train data shape:\n")
    print("x_train: {}".format(x_train.shape))
    print("y_train: {}".format(y_train.shape))

    return x_train, y_train


def get_train_data_v3():
    tokenizer = ForwardMaxMatch('./sgns.weibo.word')

    x_train, y_train = [], []
    line_idx = 0
    words_list = []
    label_list = []
    with open('./train.csv', 'r', encoding='utf-8') as f:
        for line in f:
            line_idx += 1
            if line_idx == 1:
                continue
            items = line.strip().split(',')
            label = int(items[0])
            label_list.append(label)
            sentence = ','.join(items[1:])
            words = tokenizer.cut(sentence)
            words_list.append(words)

    for words in words_list:
        vec = tokenizer.to_two_dim_vec(words)
        x_train.append(vec)
    for label in label_list:
        y_train.append(label)

    if x_train is [] or y_train is []:
        print("=========== Loading data failed =============")
        exit(1)

    proj_dim = 8
    x_train = np.array(x_train)
    x_train_lower = np.zeros((x_train.shape[0], proj_dim, x_train.shape[2]))
    for i in range(x_train.shape[2]):
        x_train_lower[:,:,i] = pca(x_train[:,:,i], proj_dim)

    y_train = np.array(y_train)
    y_train = to_one_hot(y_train)

    print("\n================= Loading data finished ===================\n")
    print("Train data shape:\n")
    print("x_train_lower: {}".format(x_train_lower.shape))
    print("y_train: {}".format(y_train.shape))

    return x_train_lower, y_train


if __name__ == '__main__':
    get_train_data_v3()
