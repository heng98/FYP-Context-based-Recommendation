from distutils.command.config import config
import torch
from torch import dropout
import torch.nn as nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear

from utils import soft_assign

class AutoEncoder(nn.Module):
    def __init__(self, 
                d_channel, 
                channel_1=500, 
                channel_2=500, 
                channel_3=2000, 
                channel_bottleneck=10):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
                            nn.Linear(d_channel, channel_1),
                            nn.ReLU(),
                            nn.Linear(channel_1, channel_2),
                            nn.ReLU(),
                            nn.Linear(channel_2, channel_3),
                            nn.ReLU(),
                            nn.Linear(channel_3, channel_bottleneck)
        )

        self.decoder = nn.Sequential(
                            nn.Linear(channel_bottleneck, channel_3),
                            nn.ReLU(),
                            nn.Linear(channel_3, channel_2),
                            nn.ReLU(),
                            nn.Linear(channel_2, channel_1),
                            nn.ReLU(),
                            nn.Linear(channel_1, d_channel)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y, z
    
    def from_pretrained(self, path):
        state_dict = torch.load(path)
        # print(state_dict.keys())
        # print(self.state_dict().keys())
        # enc_dict = self.encoder.state_dict()
        # dec_dict = self.decoder.state_dict()

        # pretrained_enc_dict = {k: v for k, v in state_dict.items() if k in enc_dict}
        # pretrained_dec_dict = {k: v for k, v in state_dict.items() if k in dec_dict}

        # print(state_dict.keys())
        # print(pretrained_enc_dict)
        # print(pretrained_dec_dict)

        # self.encoder.load_state_dict(pretrained_enc_dict)
        # self.decoder.load_state_dict(pretrained_dec_dict)

        self.load_state_dict(state_dict)


class DAE(nn.Module):
    def __init__(self, 
            d_channel, 
            channel_1=500, 
            channel_2=500, 
            channel_3=2000, 
            channel_bottleneck=10,
            dropout=0.5):   
        super(DAE, self).__init__()
        self.dropout = dropout
        self.encoder = nn.Sequential(
                            nn.Linear(d_channel, channel_1),
                            nn.ReLU(),
                            nn.Dropout(self.dropout),
                            nn.Linear(channel_1, channel_2),
                            nn.ReLU(),
                            nn.Dropout(self.dropout),
                            nn.Linear(channel_2, channel_3),
                            nn.ReLU(),
                            nn.Dropout(self.dropout),
                            nn.Linear(channel_3, channel_bottleneck)
        )

        self.decoder = nn.Sequential(
                            nn.Linear(channel_bottleneck, channel_3),
                            nn.ReLU(),
                            nn.Dropout(self.dropout),
                            nn.Linear(channel_3, channel_2),
                            nn.ReLU(),
                            nn.Dropout(self.dropout),
                            nn.Linear(channel_2, channel_1),
                            nn.ReLU(),
                            nn.Dropout(self.dropout),
                            nn.Linear(channel_1, d_channel)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)

        return y, z

class IDEC(nn.Module):
    def __init__(self, config):
        super(IDEC, self).__init__()
        self.config = config

        self.config.channel = [int(i) for i in self.config.channel.split(',')]
        print(self.config.channel)

        self.ae = AutoEncoder(self.config.channel[0], self.config.channel[1], self.config.channel[2], self.config.channel[3], self.config.channel[4])
        self.cluster_layer = nn.Parameter(torch.Tensor(self.config.n_cluster, self.config.channel[4]))
        # torch.nn.init.xavier_normal_(self.cluster_layer.data)
    
    def from_pretrained(self, path):
        state_dict = torch.load(path)
        self.ae.load_state_dict(state_dict)

    def forward(self, x):
        y, z = self.ae(x)
        q = soft_assign(z, self.cluster_layer)

        return y, q, z