import numpy as np

def pca(X, n_components):
    # 均值归一化
    X_mean = np.mean(X, axis=0)
    X_norm = X - X_mean

    # 计算协方差矩阵
    cov_matrix = np.cov(X_norm, rowvar=False)

    # 计算特征值和特征向量
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)

    # 选取前n个特征向量，组成投影矩阵
    idx = eig_values.argsort()[::-1][:n_components]
    projection_matrix = eig_vectors[:, idx]

    # 对原始数据进行降维
    X_pca = np.dot(X_norm, projection_matrix)

    return X_pca


def get_minibatch_data(x_data, labels, batch_size):
    np.random.seed(42)
    np.random.shuffle(x_data)
    np.random.seed(42)
    np.random.shuffle(labels)
    for i in range(0, x_data.shape[0]-batch_size+1, batch_size):
        idxs = slice(i, i + batch_size)
        yield x_data[idxs], labels[idxs]


def to_one_hot(labels: np.ndarray):
    """Transform label (0, 1, 2, 3) to one-hot vector"""
    num_classes = np.max(labels)
    num_labels = len(labels)
    one_hot_vec = np.zeros((num_labels, num_classes+1))
    for i, label in enumerate(labels):
        one_hot_vec[i, label] = 1

    return one_hot_vec


def get_score(pred, label):
    pred, label = np.array(pred.data), np.array(label.data)
    pred = np.argmax(pred,axis=1)
    score = np.argmax(label,1)
    score = np.sum(pred == score)
    return score
