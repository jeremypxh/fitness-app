import os
import warnings
from abc import ABC
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
from collections import Counter
import torch.nn as nn
import gc
import time
import torch
import numpy as np
import random
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import math
import pywt
from math import log


def set_seed(seed):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed,details are ", e)
        pass
    import numpy as np
    np.random.seed(seed)
    import random as python_random
    python_random.seed(seed)


class Preprocessing:

    def drop_sensors(df, sensor):
        df0 = df.copy()
        df0.drop(sensor, axis=1, inplace=True)
        return df0

    def drop_units(df, unit_index):
        df0 = df.copy()
        df0.drop(df0[df0[df0.columns[0]].isin(unit_index)].index, axis=0, inplace=True)
        return df0.reset_index(drop=True)

    def add_timeseries(df):
        df0 = df.copy()
        df0["Time"] = df0.index.values + 1
        return df0


def plotserial(mat, fig, ax):
    Omega = mat[:, [0, 1, 2]]
    Ang = mat[:, [3, 4, 5]]
    Acc = mat[:, [6, 7, 8]]
    ax[0].plot(Omega[:, 0], color='r', label='og_x')
    ax[0].plot(Omega[:, 1], color='g', label='og_y')
    ax[0].plot(Omega[:, 2], color='b', label='og_z')
    ax[1].plot(Ang[:, 0], color='r', label='ag_x')
    ax[1].plot(Ang[:, 1], color='g', label='ag_y')
    ax[1].plot(Ang[:, 2], color='b', label='ag_z')
    ax[2].plot(Acc[:, 0], color='r', label='ac_x')
    ax[2].plot(Acc[:, 1], color='g', label='ac_y')
    ax[2].plot(Acc[:, 2], color='b', label='ac_z')
    ax[0].legend(loc="lower left")
    ax[1].legend(loc="lower left")
    ax[2].legend(loc="lower left")
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    plt.show()


def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def wavelet_denoising(x, wavelet, level, s_factor):
    """
    deconstructs, thresholds then reconstructs
    higher thresholds = less detailed reconstruction
    """
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * madev(coeff[-level]) * s_factor
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')


# https://stackoverflow.com/questions/56789030/why-is-wavelet-denoising-producing-identical-results-regardless-of-threshold-lev
# https://www.kaggle.com/code/theoviel/denoising-with-direct-wavelet-transform/notebook


class SportsDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.get_mat()

    def get_mat(self):
        mat = Preprocessing.drop_sensors(self.df, 'Time').values
        scaler = MinMaxScaler((-1, 1))
        mat_minmax = scaler.fit_transform(mat)
        self.mat = mat_minmax

    def __len__(self):
        return len(self.df)
