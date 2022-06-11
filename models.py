from dgl.nn import GraphConv
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import dgl
import torch
from dgl.nn.pytorch.conv import GATConv


class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hid_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hid_feats, hid_feats*2, allow_zero_in_degree=True)
        self.conv3 = GraphConv(hid_feats*2, hid_feats*2, allow_zero_in_degree=True)
        self.conv4 = GraphConv(hid_feats*2, hid_feats, allow_zero_in_degree=True)
        self.conv5 = GraphConv(hid_feats, hid_feats, allow_zero_in_degree=True)
        self.conv6 = GraphConv(hid_feats, hid_feats, allow_zero_in_degree=True)
        self.conv7 = GraphConv(hid_feats, 1, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        h = F.relu(h)
        h = self.conv4(g, h)
        h = F.relu(h)
        h = self.conv5(g, h)
        h = F.relu(h)
        h = self.conv6(g, h)
        h = F.relu(h)
        h = self.conv7(g, h)

        return h
    


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv3 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=1, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        h = F.relu(h)
        h = self.conv3(graph, h)
        return h



class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = GATConv(in_dim, hidden_dim, num_heads, allow_zero_in_degree=True)
        self.layer2 = GATConv(hidden_dim * num_heads, out_dim, 1, allow_zero_in_degree=True)
    def forward(self, g, h):
        h = self.layer1(g, h)
        h = h.view(-1, h.size(1) * h.size(2))
        h = F.elu(h)
        h = self.layer2(g, h)
        h = h.squeeze() 
        return h