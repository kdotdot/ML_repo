import math
import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor

# Xavier uniform initializer
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

# Zero initializer
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class GCNConv(MessagePassing):
    def __init__(self,in_channels:int,out_channels:int,add_self_loops:bool=False,
             normalize:bool=True,bias:bool=True, **kwargs):

        kwargs.setdefault('aggr','add')
        super(GCNConv,self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        # Add self-loops and apply symetric normalization
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(in_channels,out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias',None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self,x:torch.Tensor, edge_index:Adj, edge_weight:OptTensor=None) ->torch.Tensor:
        if self.normalize:
            pass #TODO
        x = torch.matmul(x,self.weight)
        
        out = self.propagate(edge_index,x=x,edge_weight=edge_weight,size=None)

        if self.bias is not None:
            out += self.bias

        return out



if __name__ == '__main__':
    conv = GCNConv(2,4)
