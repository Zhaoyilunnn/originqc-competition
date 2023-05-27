import numpy as np
# import pyvqnet
import pandas as pd
import pyqpanda as pq


def train():
    """
    Please ensure that your code can run correctly and completely in IDE
    :return:
    """
    pass


def question1(validation_set_name: str):
    """
    Please execute your model here to generate predictions for the validation set, which is not publicly
    available. The predictions should be returned in the form of an array. These returned values will be utilized to
    compute the F1-score and will serve as the criteria for evaluation.
    :param validation_set_name: string, dataset's file name.
    :return: ndarray, like np.array([0,1,2,3,1,1,2,0,3])
    """
    validation_set = pd.read_csv(validation_set_name)
    validation_set_predict_label = np.array([])
    # use your model predict labels in the validation set

    return validation_set_predict_label
