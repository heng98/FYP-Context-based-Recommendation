import torch
from torch.utils.data import Dataset

from numpy.lib.function_base import vectorize
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# print(X.shape)

class NewsGroupDataset(Dataset):
    def __init__(self):
        super(NewsGroupDataset, self).__init__()
        data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'), subset='all')
        vectorizer = TfidfVectorizer(max_features=2000, dtype=np.float32)

        self.X = vectorizer.fit_transform(data['data']).toarray()
        self.Y = data['target']

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], torch.LongTensor([index])   