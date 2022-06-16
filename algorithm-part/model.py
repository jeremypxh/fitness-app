from utilities import *
import tsfresh
from tsfresh import feature_selection
from tsfresh import extract_features, select_features
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
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
from tsfresh import feature_extraction
from tsfresh import extract_features
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.decomposition import PCA
import tsfresh
from tsfresh import feature_selection
from tsfresh import extract_features, select_features
from scipy.signal import savgol_filter
import os
import warnings
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
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from utilities import *
from sklearn.cluster import KMeans
import torch.nn.functional as F


class CFG:
    num_feature = 100
    num_hidden = 128
    num_classes = 2
    batch_size = 10
    epoches = 200
    lr = 0.001
    weight_decay = 0.0001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FeatureDataset(Dataset):
    def __init__(self, feature_array, labels):
        self.features = feature_array
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if self.labels[idx] == 0:
            one_hot = torch.tensor([1,0],dtype=torch.float)
        elif self.labels[idx] == 1:
            one_hot = torch.tensor([0,1], dtype=torch.float)
        data = {"input": torch.tensor(self.features[idx], dtype=torch.float),
                "label": one_hot}
        return data


class CustomModel(nn.Module):
    def __init__(self, num_feature=CFG.num_feature, num_hidden=CFG.num_hidden):
        super(CustomModel, self).__init__()
        self.num_feature = num_feature
        self.num_hidden = num_hidden

        self.mlp = nn.Linear(self.num_feature, self.num_hidden)
        self.logits = nn.Linear(self.num_hidden, CFG.num_classes)

        for layer in [self.mlp, self.logits]:
            nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.mlp(x)
        x = F.relu(x)
        x = self.logits(x)
        x = F.log_softmax(x, dim=1)
        return x

class LSR(nn.Module):
    def __init__(self, n_classes=2, eps=0.05):
        super(LSR, self).__init__()
        self.n_classes = n_classes
        self.eps = eps

    def forward(self, outputs, label):
        # labels.shape: [b,]
        assert outputs.size(0) == label.size(0)
        mask = ~(label > 0)
        smooth_labels = torch.masked_fill(label, mask, self.eps / (self.n_classes - 1))
        smooth_labels = torch.masked_fill(smooth_labels, ~mask, 1 - self.eps)
        ce_loss = -torch.sum(outputs * smooth_labels, dim=1).mean()
        return ce_loss
