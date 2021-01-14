#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

class GNN(nn.Module):
    def __init__(self,input_channels,hidden_channels,n_classes):
        super(GNN,self).__init__()
        self.conv1 = gnn.GCNConv(input_channels,hidden_channels)
        self.conv2 = gnn.GCNConv(hidden_channels,hidden_channels)
        self.conv3 = gnn.GCNConv(hidden_channels,n_classes)


    def forward(self,x,edge_index):
        x = self.conv1(x,edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # Less is more
        # x = self.conv2(x,edge_index)
        # x = F.relu(x)
        x = self.conv3(x,edge_index)
            
        return x
    
    def train_epoch(self,criterion,optimizer,data):
        optimizer.zero_grad()
        out = self(data.x,data.edge_index)
        loss = criterion(out[data.train_mask],data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss 

    def val_epoch(self,data):
        out = self(data.x,data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        val_correct = pred[data.val_mask] == data.y[data.val_mask]  # Check against ground-truth labels.
        val_acc = (val_correct.sum()) / int(data.val_mask.sum())  # Derive ratio of correct predictions.
        return val_acc

    def test_epoch(self,data):
        out = self(data.x,data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
        test_acc = (test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
        return test_acc


if __name__ == '__main__':
    # Transform makes all node features sum up to 1 (Row wise)
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    model = GNN(dataset.num_features,64,dataset.num_classes)
    criterion = torch.nn.CrossEntropyLoss()  # Define criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.  
    
    data = dataset[0]
    best_val_acc = float('-inf')
    patience, max_patience = 0,20
    for epoch in range(1,201):
        model.train()
        train_loss = model.train_epoch(criterion,optimizer,data)
        model.eval()
        val_acc = model.val_epoch(data)
        print(f'Epoch: {epoch:03d} Train loss: {train_loss.item():.3f} Val acc: {val_acc*100:.1f}%')
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),'./best_params')
            patience=0
        else:
            patience +=1
            if patience > max_patience:
                break
    model.load_state_dict(torch.load('./best_params'))
    val_acc = model.val_epoch(data)
    test_acc = model.test_epoch(data)
    print(f'Best val accuracy: {val_acc*100:.1f}% Test accuracy: {test_acc*100:.1f}%')
