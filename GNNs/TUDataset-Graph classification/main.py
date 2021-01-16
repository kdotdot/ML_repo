#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

class GNN(nn.Module):
    def __init__(self,input_channels,hidden_channels,n_classes):
        super(GNN,self).__init__()
        self.conv1 = gnn.GraphConv(input_channels,hidden_channels)
        self.conv2 = gnn.GraphConv(hidden_channels,hidden_channels)
        self.conv3 = gnn.GraphConv(hidden_channels,hidden_channels)

        self.lin = nn.Linear(hidden_channels,n_classes)


    def forward(self,x,edge_index,batch):
        x = self.conv1(x,edge_index)
        x = F.relu(x)
        x = self.conv2(x,edge_index)
        x = F.relu(x)
        x = self.conv3(x,edge_index)

        #READOUT
        x = gnn.global_mean_pool(x,batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

    def train_epoch(self,criterion,optimizer,loader):
        losses = []
        for data in loader:
            optimizer.zero_grad()
            out = self(data.x,data.edge_index,data.batch)
            loss = criterion(out,data.y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        return np.array(losses).mean()

    def val_test_epoch(self,loader):
        correct = 0 
        for data in loader:
            out = self(data.x,data.edge_index,data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
            val_correct = pred == data.y
        return correct/len(loader.dataset)

if __name__ == '__main__':
    # Transform makes all node features sum up to 1 (Row wise)
    dataset = TUDataset(root='data/TUDataset',name ='MUTAG')

    train_dataset = dataset[:120]
    val_dataset = dataset[120:150]
    test_dataset = dataset[150:]

    train_loader = DataLoader(train_dataset,batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=64, shuffle=True)

    model = GNN(dataset.num_features,64,dataset.num_classes)
    criterion = torch.nn.CrossEntropyLoss()  # Define criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.  

    best_val_acc = float('-inf')
    patience, max_patience = 0,20
    for epoch in range(1,201):
        model.train()
        train_loss = model.train_epoch(criterion,optimizer,train_loader)
        model.eval()
        val_acc = model.val_test_epoch(val_loader)
        print(f'Epoch: {epoch:03d} Train loss: {train_loss:.3f} Val acc: {val_acc*100:.1f}%')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),'./best_params')
            patience=0
        else:
            if best_val_acc-val_acc < 0.1:
                patience +=1
                if patience > max_patience:
                    break
    model.load_state_dict(torch.load('./best_params'))
    val_acc = model.val_test_epoch(val_loader)
    test_acc = model.val_test_epoch(test_loader)
    print(f'Best val accuracy: {val_acc*100:.1f}% Test accuracy: {test_acc*100:.1f}%')
