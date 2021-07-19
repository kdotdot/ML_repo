# TODO: LOSS DOESNT GO DOWN PLEASE FIX
# May be over parameterized
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from einops import rearrange, reduce, repeat

import matplotlib.pyplot as plt 

# https://www.kaggle.com/reukki/pytorch-cnn-tutorial-with-cats-and-dogs
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn,self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3, padding=0,stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        
        self.fc1 = nn.Linear(3*3*64,10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10,2)
        self.relu = nn.ReLU()
        
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out



def accuracy_metric(outputs, targets):
    pred = torch.argmax(outputs,-1)
    return torch.sum(pred==targets).item() / targets.shape[0]

if __name__ == '__main__':
    transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(255),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 0: Cat 1:Dog
    train_dataset = datasets.ImageFolder('./train', transform=transformations,target_transform = None)

    indices = np.arange(len(train_dataset))
    np.random.shuffle(indices)
    train_idx, valid_idx,test_idx = indices[:1000], indices[1000:2000], indices[2000:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=512,sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(train_dataset,
                                      batch_size=512,sampler=valid_sampler)

    test_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=512,sampler=test_sampler)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    model = Cnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)
    loss_fct = nn.CrossEntropyLoss()
    for epoch in range(10):
        model.train()
        losses, outputs, targets = [], [] , []
        for batch in train_loader:
            optimizer.zero_grad()
            X,Y = batch[0].to(device),batch[1].to(device)
            out = model(X)
            loss= loss_fct(out,Y)
            losses.append(loss.detach().item())
            outputs.append(out.detach())
            targets.append(Y)
            loss.backward()
            optimizer.step()
        outputs, targets = torch.cat(outputs), torch.cat(targets)
        print(f'Epoch: {epoch} Train loss: {np.array(losses).mean()} Train acc: {accuracy_metric(outputs, targets)}')

        model.eval()
        losses_v, outputs_v, targets_v = [], [] , []
        for batch in val_loader:
            X,Y = batch[0].to(device),batch[1].to(device)
            out = model(X)
            loss= loss_fct(out,Y)
            losses_v.append(loss.detach().item())
            outputs_v.append(out.detach())
            targets_v.append(Y)
        outputs_v, targets_v = torch.cat(outputs_v), torch.cat(targets_v)
        print(f'Epoch: {epoch} Val loss: {np.array(losses_v).mean()} Val acc: {accuracy_metric(outputs_v, targets_v)}')
        
    model.eval()
    losses, outputs, targets = [], [] , []
    for batch in test_loader:
        X,Y = batch[0].to(device),batch[1].to(device)
        out = model(X)
        loss= loss_fct(out,Y)
        losses.append(loss.detach().item())
        outputs.append(out.detach())
        targets.append(Y)
    outputs, targets = torch.cat(outputs), torch.cat(targets)
    print(f'Epoch: {epoch} Test loss: {np.array(losses).mean()} Test acc: {accuracy_metric(outputs, targets)}')
