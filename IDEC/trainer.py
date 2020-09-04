from torch.utils.data import data, dataloader
from IDEC.utils import cluster_acc, target_dist
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from tqdm import tqdm

class Trainer():
    def __init__(self, model: nn.Module, dataset: Dataset, config):
        self.model = model
        self.config = config

        self.optimizer = optim.Adam()
        
        # Loss function
        self.clustering_loss_f = nn.KLDivLoss()
        self.rec_loss_f = nn.MSELoss()

        self.kmeans = KMeans(n_cluster=self.config.n_cluster, n_init=20)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dataset = dataset
        self.dataloader = DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)

    def pretrain(self):
        
    
    def step(self, x, idx):
        """Forward function of each iteration

        Parameters:

            x: One batch of data
            idx: Index of one batch of data
        """
        y, q, _ = self.model(x)

        self.rec_loss = self.rec_loss_f(y, x)
        self.clustering_loss = self.clustering_loss_f(q.log(), self.p[idx])

    def optimize(self):
        """Calculate loss and backward propagation"""

        self.loss = self.rec_loss + self.config.gamma * self.clustering_loss

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def update_p(self):
        """Update target distribution P that serves as soft label"""

        _, tmp_q, _ = self.model(self.dataset)
        self.p = target_dist(tmp_q)

        y_pred = tmp_q.cpu().numpy().argmax(1)

        acc = cluster_acc(y, y_pred)
        nmi = nmi_score(y, y_pred)
        ari = ari_score(y, y_pred)

    def init_cluster_center(self):
        """Initialize cluster center by employing k-means on the embedded points z"""

        print("Initializing cluster centers with kmeans")
        self.model.eval()
    
        encoded = []
        for x, _ in tqdm(dataloader):
            x = x.to(self.device)
            _, _, z = self.model(x)
            encoded.append(z)

        encoded = torch.cat(encoded, dim=0)
        y_pred = self.kmeans.fit_predict(encoded.cpu().numpy())
        self.model.cluster_layer.data.copy_(torch.Tensor(self.kmeans.cluster_centers_))

    def train(self):
        self.init_cluster_center()
        self.model.train()
        for epoch in range(self.config.epoch):
            if epoch % self.config.update_interval == 0:
                self.update_p()

            for x, idx in tqdm(dataloader):
                x = x.to(self.device)
                idx = idx.to(self.device)

                self.step(x, idx)
                self.optimize()




