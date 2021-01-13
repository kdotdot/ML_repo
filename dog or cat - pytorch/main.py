# TODO: LOSS DOESNT GO DOWN PLEASE FIX
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from einops import rearrange, reduce, repeat

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,16,7)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(16,32,7,3)
        self.pool2 = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(32 * 17 * 17,120)
        self.fc2 = nn.Linear(120,100)
        self.fc3 = nn.Linear(100,2)

        self.finalact = nn.LogSoftmax(dim=1)

    # Add dropout
    # Add pooling
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = rearrange(x,'n c h w -> n (c h w)')
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.finalact(x)

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
    #optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)
    loss_fct = nn.NLLLoss()
    for epoch in range(3):
        model.train()
        for batch in train_loader:
            X = batch[0]
            y = batch[1].long()
            optimizer.zero_grad()
            out = model(X)
            loss= loss_fct(out,y)
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch} Loss: {loss.item()}')
