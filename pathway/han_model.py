import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv
import dgl.nn as dglnn


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=4):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        # print(self.project(z))
        # print(w)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        # print(beta)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        # print(beta[0])

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)  # 将meta-path转换成元组形式

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:  # 第一次，建立一张metapath下的异构图
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                    g, meta_path)  # 构建异构图的邻居;
        # self._cached_coalesced_graph 多个metapath下的异构图
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]  # meta-path下的节点邻居图
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))  # 图attention
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        # 确保 num_heads 列表的长度至少为2，以匹配两层所需的头数
        if len(num_heads) < 2:
            raise ValueError("num_heads must contain at least two elements for two layers")

        # 第一层使用10个注意力头
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))

        # 第二层使用 num_heads[1] 个注意力头
        self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[0],
                                     hidden_size, num_heads[1], dropout))

        # 修改全连接层以匹配第二层的输出尺寸
        self.predict = nn.Linear(hidden_size * num_heads[1], out_size)

    def forward(self, g, h):
        # 应用第一层图卷积
        h = self.layers[0](g, h)

        # 应用第二层图卷积
        h = self.layers[1](g, h)

        # 使用全连接层进行预测
        return self.predict(h)


# Define a Heterograph Conv model
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats1, hid_feats2, hid_feats3, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats1)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats1, hid_feats2)
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats2, hid_feats3)
            for rel in rel_names}, aggregate='sum')

        self.conv4 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats3, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}

        h = self.conv2(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}

        h = self.conv3(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}

        h = self.conv4(graph, h)

        return h


class MLPPredictor(nn.Module):
    def __init__(self, in_features, hind_feature, out_classes):
        super().__init__()
        self.W1 = nn.Linear(in_features, hind_feature)
        self.W2 = nn.Linear(hind_feature, out_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, dst, h):
        h_list = []
        '''
        for i in range(len(h)):
            with open('h.txt', 'a') as f:
                f.write(str(h[i]))
        '''
        for i in range(len(src)):
            h_u = h[src[i]]
            h_v = h[dst[i]]
            h_concat = h_u + h_v
            h_list.append(h_concat)

        # print(h_list)

        h_list = torch.stack((h_list), dim=0)
        # print(h_list)

        fc1 = self.W1(h_list)
        fc1 = self.relu(fc1)
        fc2 = self.W2(fc1)
        score = self.sigmoid(fc2)

        return score


class CNNPredictor(nn.Module):
    def __init__(self, in_feats, hid1_feats, hid2_feats, hid3_feats, hid4_feats, out_feats):
        super().__init__()
        self.hid4_feats = hid4_feats
        # 原有的两层卷积和批标准化层
        self.cnn1 = nn.Conv2d(in_channels=in_feats, out_channels=hid1_feats, kernel_size=(2, 2), padding='same')
        self.bn1 = nn.BatchNorm2d(hid1_feats)
        self.cnn2 = nn.Conv2d(in_channels=hid1_feats, out_channels=hid2_feats, kernel_size=(2, 2), padding='same')
        self.bn2 = nn.BatchNorm2d(hid2_feats)

        # 新增的两层卷积和批标准化层
        self.cnn3 = nn.Conv2d(in_channels=hid2_feats, out_channels=hid3_feats, kernel_size=(2, 2), padding='same')
        self.bn3 = nn.BatchNorm2d(hid3_feats)
        self.cnn4 = nn.Conv2d(in_channels=hid3_feats, out_channels=hid4_feats, kernel_size=(2, 2), padding='same')
        self.bn4 = nn.BatchNorm2d(hid4_feats)

        # 全连接层
        self.fc3 = nn.Linear(hid4_feats, out_feats)

        # Dropout层
        self.dropout = nn.Dropout(0.5)

        # 激活函数
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, dst, h):
        h_list = []

        for i in range(len(src)):
            h_u = h[src[i]]
            h_v = h[dst[i]]
            h_concat = h_u + h_v
            h_list.append(h_concat)

        h_list = torch.stack(h_list, dim=0)
        h_list = torch.reshape(h_list, (-1, 128, 1, 1))  # 假设128是in_feats的值

        h = self.cnn1(h_list)
        h = self.bn1(h)
        h = self.relu(h)

        h = self.cnn2(h)
        h = self.bn2(h)
        h = self.relu(h)

        # 新增的卷积层操作
        h = self.cnn3(h)
        h = self.bn3(h)
        h = self.relu(h)

        h = self.cnn4(h)
        h = self.bn4(h)
        h = self.relu(h)

        # 可能的dropout操作
        # h = self.dropout(h)

        # 展平层以适配全连接层
        h = h.view(-1, self.hid4_feats)
        h = self.fc3(h)
        h = self.sigmoid(h)

        return h


class Model(nn.Module):
    def __init__(self, meta_paths, in_features, hidden_features1, hidden_features2, hidden_features3, hidden_features4, hidden_features5, hidden_features6, out_features, rel_names):
        super().__init__()
        self.rgcn = RGCN(in_feats=in_features,
                         hid_feats1=hidden_features1,
                         hid_feats2=hidden_features2,
                         hid_feats3=hidden_features3,
                         out_feats=out_features,
                         rel_names=rel_names)
        self.han = HAN(meta_paths, in_features, hidden_features1, hidden_features2, num_heads=[10, 8], dropout=0.4)
        self.fc = CNNPredictor(
            in_feats=hidden_features2,
            hid1_feats=hidden_features3,
            hid2_feats=hidden_features4,
            hid3_feats=hidden_features5,
            hid4_feats=hidden_features6,
            out_feats=out_features
        )

    def forward(self, graph, feature_all, drug_feature, src, dst):
        h1 = self.rgcn(graph, feature_all)['drug']
        h_mp = self.han(graph, drug_feature)

        h = h1 + h_mp
        #  print(h)
        score_all = self.fc(src, dst, h)

        return score_all