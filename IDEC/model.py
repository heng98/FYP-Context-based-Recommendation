import torch
from torch import dropout
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear

from utils import soft_assign

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
        y = self.decoder(z)

        return y, z

    def from_pretrain(self, path):
        state_dict = torch.load(path)
        enc_dict = self.encoder.state_dict()
        dec_dict = self.decoder.state_dict()

        pretrained_enc_dict = {k: v for k, v in state_dict['encoder'].items() if k in enc_dict}
        pretrained_dec_dict = {k: v for k, v in state_dict['decoder'].items() if k in dec_dict}

        self.encoder.load_state_dict(pretrained_enc_dict)
        self.decoder.load_state_dict(pretrained_dec_dict)


class DAE(nn.Module):
    def __init__(self, d_channel, dropout):    
        super(DAE, self).__init__()
        self.dropout = dropout
        self.encoder = nn.Sequential(
                            nn.Linear(d_channel, 500),
                            nn.ReLU(),
                            nn.Dropout(self.dropout),
                            nn.Linear(500, 500),
                            nn.ReLU(),
                            nn.Dropout(self.dropout),
                            nn.Linear(500, 2000),
                            nn.ReLU(),
                            nn.Dropout(self.dropout),
                            nn.Linear(2000, 10)
        )

        self.decoder = nn.Sequential(
                            nn.Linear(10, 2000),
                            nn.ReLU(),
                            nn.Dropout(self.dropout),
                            nn.Linear(2000, 500),
                            nn.ReLU(),
                            nn.Dropout(self.dropout),
                            nn.Linear(500, 500),
                            nn.ReLU(),
                            nn.Dropout(self.dropout),
                            nn.Linear(500, d_channel)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y, z

class IDEC(nn.Module):
    def __init__(self, d_channel, n_cluster):
        super(IDEC, self).__init__()
        self.ae = AutoEncoder(d_channel)
        self.cluster_layer = nn.Parameter(torch.Tensor(n_cluster, 10))
        # torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x):
        y, z = self.ae(x)
        q = soft_assign(z, self.cluster_layer)

        return y, q, z