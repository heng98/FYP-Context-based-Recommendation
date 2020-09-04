import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import DAE
from dataset import NewsGroupDataset
from tqdm import tqdm

model = DAE(2000, 0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = NewsGroupDataset()
dataloader = DataLoader(data, batch_size=256, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(5):
    epoch_loss = 0
    for x, _ in tqdm(dataloader):
        x.to(device)

        y, _ = model(x)
        loss = criterion(y, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    print(epoch_loss)

torch.save(model.state_dict(), './weights/pretrain_weight.pth')


        
