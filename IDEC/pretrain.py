import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from collections import OrderedDict

from model import DAE
from dataset import NewsGroupDataset
from tqdm import tqdm

model = DAE(2000, 4000, 2000, 500, 50, 0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = model.to(device)

data = NewsGroupDataset()
j = int(len(data) * 0.8)
train, test = random_split(data, [j, len(data)- j])
train_dataloader = DataLoader(train, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test, batch_size=256, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(10):
    epoch_loss = 0
    model.train()
    for x, _, _ in tqdm(train_dataloader):
        x = x.to(device)

        y, _ = model(x)
        loss = criterion(y, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_dataloader)
    print(epoch_loss)

    model.eval()
    with torch.no_grad():
        for x, _, _ in tqdm(test_dataloader):
            x = x.to(device)

            y, _ = model(x)
            loss = criterion(y, x)

            epoch_loss += loss.item()
    epoch_loss /= len(test_dataloader)
    print(epoch_loss)
    print()

model.eval()
state_dict = model.state_dict()
# new_state_dict = OrderedDict()
# Rename key of state dict to fit the AutoEncoder
state_dict['encoder.2.weight'] = state_dict.pop('encoder.3.weight')
state_dict['encoder.2.bias'] = state_dict.pop('encoder.3.bias')
state_dict['encoder.4.weight'] = state_dict.pop('encoder.6.weight')
state_dict['encoder.4.bias'] = state_dict.pop('encoder.6.bias')
state_dict['encoder.6.weight'] = state_dict.pop('encoder.9.weight')
state_dict['encoder.6.bias'] = state_dict.pop('encoder.9.bias')

state_dict['decoder.2.weight'] = state_dict.pop('decoder.3.weight')
state_dict['decoder.2.bias'] = state_dict.pop('decoder.3.bias')
state_dict['decoder.4.weight'] = state_dict.pop('decoder.6.weight')
state_dict['decoder.4.bias'] = state_dict.pop('decoder.6.bias')
state_dict['decoder.6.weight'] = state_dict.pop('decoder.9.weight')
state_dict['decoder.6.bias'] = state_dict.pop('decoder.9.bias')
    
torch.save(state_dict, './weights/pretrain_weight.pth')


        
