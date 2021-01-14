#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.datasets import KarateClub

class GNN(nn.Module):
    def __init__(self,input_channels,hidden_channels,n_classes):
        super(GNN,self).__init__()
        self.conv1 = gnn.GCNConv(input_channels,hidden_channels)
        self.conv2 = gnn.GCNConv(hidden_channels,hidden_channels)
        self.conv3 = gnn.GCNConv(hidden_channels,hidden_channels)

        self.fc1 = nn.Linear(hidden_channels,n_classes)

    def forward(self,x,edge_index):
        h = self.conv1(x,edge_index)
        h = F.relu(h)
        h = self.conv2(h,edge_index)
        h = F.relu(h)
        h = self.conv3(h,edge_index)
        h = F.relu(h)
            
        out = self.fc1(h)
        return h, out
    
    def train_epoch(self,criterion,optimizer,data):
        optimizer.zero_grad()
        h, out = self(data.x,data.edge_index)
        loss = criterion(out[data.train_mask],data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss 


if __name__ == '__main__':
    dataset = KarateClub()
    model = GNN(dataset.num_features,16,dataset.num_classes)
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.  
    
    data = dataset[0]
    for epoch in range(1,41):
        model.train()
        loss = model.train_epoch(criterion,optimizer,data)
        print(f'Epoch: {epoch:03d} Train loss: {loss.item():.6f}')
