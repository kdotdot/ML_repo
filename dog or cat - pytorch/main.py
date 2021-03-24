# TODO: LOSS DOESNT GO DOWN PLEASE FIX
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from einops import rearrange, reduce, repeat

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,16,3)
        self.conv2 = nn.Conv2d(16,32,3,3)
        self.conv3 = nn.Conv2d(32,64,5,3)

        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(p=0.15)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512,32)
        self.fc3 = nn.Linear(32,1)


    # Add dropout
    # Add pooling
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = rearrange(x,'n c h w -> n (c h w)')
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1) 

if __name__ == '__main__':
    transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder('./train', transform=transformations)
    test_dataset = datasets.ImageFolder('./test1', transform=transformations)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=128,
                                          shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=128,
                                          shuffle=True)

    model = Net()
    optimizer = torch.optim.Adam(model.parameters())
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)
    loss_fct = nn.BCEWithLogitsLoss()
    for epoch in range(10):
        model.train()
        for batch in train_loader:
            X = batch[0]
            y = batch[1]
            optimizer.zero_grad()
            out = model(X)
            out = out.double() 
            y = y.double()
            loss= loss_fct(out,y)
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch} Train loss: {loss.item()}')
    model.eval()
    losses = []
    for batch in test_loader:
        X = batch[0]
        y = batch[1]
        out = model(X)
        out = out.double() 
        y = y.double()
        loss= loss_fct(out,y)
        losses.append(loss.item())
    print(f'Test loss: {np.array(losses).mean()}')
