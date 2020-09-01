from IDEC.utils import cluster_acc, target_dist
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

class Trainer():
    def __init__(self, model: nn.Module, data: Dataset, config):
        self.model = model
        self.config = config

        self.optimizer = optim.Adam()
        
        # Loss function
        self.clustering_loss_f = nn.KLDivLoss()
        self.rec_loss_f = nn.MSELoss()

        self.kmean = KMeans(n_cluster=self.config.n_cluster)

    def step(self, x, idx):
        y, q = self.model(x)

        self.rec_loss = self.rec_loss_f(y, x)
        self.clustering_loss = self.clustering_loss_f(q, self.p)

    def optimize(self):
        self.loss = self.rec_loss + self.config.gamma * self.clustering_loss

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def update_p(self):
        _, tmp_q = self.model(self.dataset)
        self.p = target_dist(tmp_q)

        y_pred = tmp_q.cpu().numpy().argmax(1)

        acc = cluster_acc(y, y_pred)
        nmi = nmi_score(y, y_pred)
        ari = ari_score(y, y_pred)

    def train(self):
        for epoch in range(self.config.epoch):
            if epoch % self.config.update_interval == 0:
                self.update_p()

            for x, idx in 

