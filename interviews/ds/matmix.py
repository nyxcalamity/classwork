#!/usr/bin/env python
"""
    Licencing terms here
"""
import math
import pandas as pd
import matplotlib.pyplot as plt


__author__ = "Denys Sobchyshak"
__email__ = "denys.sobchyshak@gmail.com"


def rmse(ts1, ts2):
    """
    Calculate root-mean-square error (RMSE)
    :param ts1:
    :param ts2:
    :return:
    """
    if ts1.shape == ts2.shape:
        return math.sqrt(mse(ts1, ts2))


def mse(ts1, ts2):
    """
    Calculate mean squared error (MSE)
    :param ts1:
    :param ts2:
    :return:
    """
    if ts1.shape == ts2.shape:
        return ((ts1 - ts2) ** 2).sum() / ts1.shape[0]


def mae(ts1, ts2):
    """
    Calculate mean absolute error (MAE)
    :param ts1:
    :param ts2:
    :return:
    """
    if ts1.shape == ts2.shape:
        return (ts1 - ts2).abs().sum() / ts1.shape[0]


def mape(actual, forecast):
    """
    Calculate mean absolute percentage error (MAPE)
    :param actual:
    :param forecast:
    :return:
    """
    if actual.shape == forecast.shape:
        return ((actual - forecast) / actual).abs().sum() / actual.shape[0]


def mpe(actual, forecast):
    """
    Calculate mean percentage error (MPE)
    :param actual:
    :param forecast:
    :return:
    """
    if actual.shape == forecast.shape:
        return ((actual - forecast) / actual).sum() / actual.shape[0]


def hot_encode(features, labels):
    """
    Use inplace one-hot-encoding for categorical or insusceptible to distance features.
    :param features:
    :param labels:
    :return:
    """
    for label in labels:
        encoding = pd.get_dummies(features[label], prefix=label)
        # default DataFrame.join feature seems to have memory problems, doing the addition manually
        for column_label in encoding.columns.values:
            features[column_label] = encoding[column_label]
    features.drop(labels, axis=1, inplace=True)


def normalize(features):
    """
    Scale data in provided series into [0,1] range.
    :param features:
    :return:
    """
    return (features - features.min()) / (features.max() - features.min())


def show_prediction(actual, forecast, title=''):
    """
    Visualizes a comparison of actual data vs forecast.
    :param actual:
    :param forecast:
    :param title:
    :return:
    """
    plt.scatter(actual.index.values, actual, c='k', label='Data')
    plt.plot(forecast, c='g', label='Forecast')
    plt.title(title)
    plt.legend()
    plt.show()
