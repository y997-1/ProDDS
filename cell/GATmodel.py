import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
from dgl.nn.pytorch import GATConv



class GATEncoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation=F.relu, k: int = 2, num_heads: int = 4):
        super(GATEncoder, self).__init__()
        assert k >= 2, "Number of GAT conv layers must be at least 2"
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.layers = nn.ModuleList()

        # 第一层
        self.layers.append(GATConv(in_features, 2 * out_features // num_heads, num_heads=num_heads))

        # 中间层
        for _ in range(k - 2):
            self.layers.append(GATConv(2 * out_features, 2 * out_features // num_heads, num_heads))

        # 最后一层
        self.layers.append(GATConv(2 * out_features, out_features, num_heads=1))

    def forward(self, g, x):
        for layer in self.layers:
            x = layer(g, x)
            x = self.activation(x)
            x = x.flatten(1)
        return x

class Cell2Vec(nn.Module):

    def __init__(self, encoder: GATEncoder, n_cell, n_dim):
        super(Cell2Vec, self).__init__()
        self.encoder = encoder
        self.embeddings = nn.Embedding(n_cell, n_dim)
        self.projector = nn.Sequential(
            nn.Linear(encoder.out_features, n_dim),
            nn.Dropout()
        )

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor,
                x_indices: torch.LongTensor, c_indices: torch.LongTensor):
        encoded = self.encoder(g, x)
        encoded = encoded.index_select(0, x_indices)
        proj = self.projector(encoded).permute(1, 0)
        emb = self.embeddings(c_indices)
        out = torch.mm(emb, proj)
        return out


class RandomW(nn.Module):

    def __init__(self, n_node, n_node_dim, n_cell, n_dim):
        super(RandomW, self).__init__()
        self.encoder = nn.Embedding(n_node, n_node_dim)
        self.embeddings = nn.Embedding(n_cell, n_dim)
        self.projector = nn.Sequential(
            nn.Linear(n_node_dim, n_dim),
            nn.Dropout()
        )

    def forward(self, x_indices: torch.LongTensor, c_indices: torch.LongTensor):
        encoded = self.encoder(x_indices)
        proj = self.projector(encoded).permute(1, 0)
        emb = self.embeddings(c_indices)
        out = torch.mm(emb, proj)
        return out
