from torch import optim
from torch.nn.modules import module


import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.cluster import KMeans

class Trainer():
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam()
        self.criterion = nn.KLDivLoss()
        self.kmean = KMeans()