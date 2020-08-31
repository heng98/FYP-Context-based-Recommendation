import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear

from utils import similarity_q

class AutoEncoder(nn.Module):
    def __init__(self, d_channel):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
                            nn.Linear(d_channel, 500),
                            nn.ReLU(),
                            nn.Linear(500, 500),
                            nn.ReLU(),
                            nn.Linear(500, 2000),
                            nn.ReLU(),
                            nn.Linear(2000, 10)
        )

        self.decoder = nn.Sequential(
                            nn.Linear(10, 2000),
                            nn.ReLU(),
                            nn.Linear(2000, 500),
                            nn.ReLU(),
                            nn.Linear(500, 500),
                            nn.ReLU(),
                            nn.Linear(500, d_channel)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        y = self.encoder(z)

        return y, z

class IDEC(nn.Module):
    def __init__(self, d_channel, n_cluster):
        super(IDEC, self).__init__()
        self.ae = AutoEncoder(d_channel)
        self.cluster_layer = nn.Parameter(torch.Tensor(n_cluster, 10))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x):
        y, z = self.ae(x)
        q = similarity_q(z, self.cluster_layer)

        return y, q