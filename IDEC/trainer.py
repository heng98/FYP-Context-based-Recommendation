from utils import cluster_acc, target_dist, tsne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from tqdm import tqdm

class Trainer():
    def __init__(self, model: nn.Module, dataset: Dataset, config):
        # self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)


        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        
        # Loss function
        self.clustering_loss_f = nn.KLDivLoss(reduction='batchmean')
        self.rec_loss_f = nn.MSELoss()

        self.kmeans = KMeans(n_clusters=self.config.n_cluster, n_init=30)

        self.dataset = dataset
        self.X = torch.Tensor(self.dataset.X)
        self.Y = self.dataset.Y
        self.dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

        self.y_prev = None

    def step(self, x, idx):
        """Forward function of each iteration

        Parameters:

            x: One batch of data
            idx: Index of one batch of data
        """
        y, q, _ = self.model(x)

        self.rec_loss = self.rec_loss_f(y, x)
        # print(self.p[idx].size())
        self.clustering_loss = self.clustering_loss_f(q.log(), self.p[idx])

    def optimize(self):
        """Calculate loss and backward propagation"""

        self.loss = self.rec_loss + self.config.gamma * self.clustering_loss

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def update_p(self):
        """Update target distribution P that serves as soft label"""
        print('Update_P')
        with torch.no_grad():
            _, tmp_q, z = self.model(self.X.to(self.device))

        self.p = target_dist(tmp_q)
        z = z.cpu().numpy()
        tsne(z, self.Y, './plot.png')



        y_pred = tmp_q.cpu().numpy().argmax(1)

        acc = cluster_acc(self.Y, y_pred)
        nmi = nmi_score(self.Y, y_pred)
        ari = ari_score(self.Y, y_pred)

        print("Acc: {}, NMI: {}, ARI: {}".format(acc, nmi, ari))

        delta_label = np.sum(y_pred != self.y_prev.astype(np.float32) / y_pred.shape[0])
        self.y_prev = y_pred

        if delta_label < self.config.tol:
            print(delta_label)
            print('Reached tolerance threshold. Stopping training.')
            torch.save(self.model.state_dict(), './weights/weight.pth')
            exit()


    def init_cluster_center(self):
        """Initialize cluster center by employing k-means on the embedded points z"""

        print("Initializing cluster centers with kmeans")
        self.model.eval()
    
        encoded = []
        with torch.no_grad():
            for x, _, _ in tqdm(self.dataloader):
                x = x.to(self.device)
                _, _, z = self.model(x)
                encoded.append(z)

        encoded = torch.cat(encoded, dim=0)
        y_pred = self.kmeans.fit_predict(encoded.cpu().numpy())
        self.y_prev = y_pred
        self.model.cluster_layer.data.copy_(torch.Tensor(self.kmeans.cluster_centers_))

    def train(self):
        self.init_cluster_center()
        self.model.train()
        for epoch in range(self.config.epoch):
            if epoch % self.config.update_interval == 0:
                self.update_p()

            for x, _, idx in tqdm(self.dataloader):
                x = x.to(self.device)
                idx = idx.to(self.device)

                self.step(x, idx)
                self.optimize()




