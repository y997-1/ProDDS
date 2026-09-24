import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import itertools
import sklearn
from sklearn import svm
import torch.utils.data as Data
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pylab as plt
import random
import matplotlib
import pickle
import warnings

warnings.filterwarnings('ignore')
matplotlib.use('Agg')


def my_softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = [round(i / sum_exp_x, 2) for i in exp_x]
    return y


# Model


# Define a Heterograph Conv model
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # 输入是节点的特征字典

        h = self.conv1(graph, inputs)

        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)

        '''
        for item in h.items():
            for i in range(len(item)):
                str1 = item[i]
                for j in range(len(str1)):
                    with open('h_rgcn.txt', 'a') as f:
                        f.write(str(str1[j]))
       '''
        return h


class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()

        self.conv1 = dglnn.DenseGraphConv(
            in_feats=in_feats, out_feats=hid_feats)
        self.conv2 = dglnn.DenseGraphConv(
            in_feats=hid_feats, out_feats=out_feats)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, g_adj, feature):
        h = self.conv1(g_adj, feature)
        h = self.relu(h)
        h = self.conv2(g_adj, h)
        h = self.relu(h)

        return h


class GCN_MP(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()

        self.conv1_dtd = dglnn.DenseGraphConv(
            in_feats=in_feats, out_feats=hid_feats)
        self.conv2_dtd = dglnn.DenseGraphConv(
            in_feats=hid_feats, out_feats=out_feats)
        self.conv1_dttd = dglnn.DenseGraphConv(
            in_feats=in_feats, out_feats=hid_feats)
        self.conv2_dttd = dglnn.DenseGraphConv(
            in_feats=hid_feats, out_feats=out_feats)
        self.conv1_dtptd = dglnn.DenseGraphConv(
            in_feats=in_feats, out_feats=hid_feats)
        self.conv2_dtptd = dglnn.DenseGraphConv(
            in_feats=hid_feats, out_feats=out_feats)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()

    def forward(self, dtd, dttd, dtptd, feature):
        h_dtd = self.conv1_dtd(dtd, feature)
        h_dttd = self.conv1_dttd(dttd, feature)
        h_dtptd = self.conv1_dtptd(dtptd, feature)

        h_dtd = self.relu(h_dtd)
        h_dtd = self.conv2_dtd(dtd, h_dtd)
        h_dtd = self.relu(h_dtd)

        h_dttd = self.relu(h_dttd)
        h_dttd = self.conv2_dttd(dttd, h_dttd)
        h_dttd = self.relu(h_dttd)

        h_dtptd = self.relu(h_dtptd)
        h_dtptd = self.conv2_dtptd(dtptd, h_dtptd)
        h_dtptd = self.relu(h_dtptd)

        return h_dtd, h_dttd, h_dtptd


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
        # print('score: ',score)
        # score = fc1

        return score


class CNNPredictor(nn.Module):
    def __init__(self, in_feats, hid1_feats, hid2_feats, out_feats):
        super().__init__()
        self.hid2_feats = hid2_feats
        self.cnn1 = nn.Conv2d(in_channels=in_feats, out_channels=hid1_feats, kernel_size=(2, 2), padding='same')
        self.bn1 = nn.BatchNorm2d(hid1_feats)
        self.cnn2 = nn.Conv2d(in_channels=hid1_feats, out_channels=hid2_feats, kernel_size=(2, 2), padding='same')
        self.bn2 = nn.BatchNorm2d(hid2_feats)
        self.fc3 = nn.Linear(hid2_feats, out_feats)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
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

        h = self.cnn1(h_list)
        h = self.bn1(h)
        h = self.tanh(h)

        # h = self.dropout(h)

        h = self.cnn2(h)
        h = self.bn2(h)
        h = self.tanh(h)

        # print(h.shape)
        # h = self.dropout(h)

        # h = self.fc3(h)
        # h = torch.sigmoid(h)
        h = h.view(-1, self.hid2_feats)
        h = self.fc3(h)
        # print(h.shape)
        h = self.sigmoid(h)
        # print(h.shape)

        return h


class Model(nn.Module):
    def __init__(self, in_features, hidden_features1, hidden_features2, hidden_features3, out_features, rel_names):
        super().__init__()
        self.rgcn = RGCN(in_features, hidden_features1, hidden_features2, rel_names)
        self.gcn = GCN_MP(in_features, hidden_features1, hidden_features2)
        self.fc = MLPPredictor(hidden_features2, hidden_features3, out_features)
        self.rel_names = rel_names

    def my_attention(self, h_list):

        h_list_shape1 = h_list.shape[1]
        h_list_shape2 = h_list.shape[2]

        pool_1 = nn.AvgPool2d(kernel_size=(1, h_list.shape[2]), stride=1, padding=0)
        relu = nn.ReLU()
        sigmoid = nn.Sigmoid()
        softmax = nn.Softmax(dim=0)
        h_pool = pool_1(h_list)
        # print(h_pool)
        h_pool = h_pool.squeeze(-2)
        h_pool = h_pool.view(-1, h_pool.shape[1])
        # print('view: ', h_pool)

        fc1 = nn.Linear(h_pool.shape[1], int(h_pool.shape[1] / 2))
        fc2 = nn.Linear(int(h_pool.shape[1] / 2), h_pool.shape[1])
        h_pool = fc1(h_pool)
        # print('fc1: ', h_pool)
        h_pool = relu(h_pool)

        h_pool = fc2(h_pool)

        h_pool = h_pool.squeeze(-1)
        h_pool = sigmoid(h_pool)
        h_pool_t = h_pool.T
        h_list_new = []

        for i in range(h_pool_t.shape[0]):

            h_list_add = 0
            for j in range(len(h_pool_t[i])):
                h_list[j][i] = h_pool_t[i][j] * h_list[j][i]

                h_list_add += h_list[j][i]
            h_list_new.append(h_list_add)

        h_list_all = torch.stack((h_list_new), dim=0)
        h_list_all = torch.reshape(h_list_all, (h_list_shape1, h_list_shape2))

        return h_list_all, h_pool_t

    def forward(self, graph, DTD, DTTD, DTPTD, feature_all, drug_feature, src, dst, mark=None):

        h1 = self.rgcn(graph, feature_all)['drug']

        h_DTD, h_DTTD, h_dtptd = self.gcn(DTD, DTTD, DTPTD, drug_feature)

        if mark == 'A':
            h_list = [h1, h_DTD, h_DTTD]
            h_list = torch.cat((h_list), 0)
            h_list = torch.reshape(h_list, (-1, h1.shape[0], h1.shape[1]))
            h, att_w = self.my_attention(h_list)
            # print(att_w)

        else:
            h = h1 + h_DTTD + h_DTD + h_dtptd

        score_all = self.fc(src, dst, h)

        return score_all


class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h是从5.1节中对每种类型的边所计算的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h  # 一次性为所有节点类型的 'h'赋值
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


class Model_1(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names):
        super().__init__()
        self.rgcn = RGCN(in_features, hidden_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()

    def forward(self, g, x, etype):
        h = self.rgcn(g, x)
        print(h)
        return self.pred(g, h, etype)


class CNN(nn.Module):
    def __init__(self, in_feats, hid1_feats, hid2_feats, out_feats):
        super().__init__()
        self.hid2_feats = hid2_feats
        self.cnn1 = nn.Conv2d(in_channels=in_feats, out_channels=hid1_feats, kernel_size=(2, 2), padding='same')
        self.bn1 = nn.BatchNorm2d(hid1_feats)
        self.cnn2 = nn.Conv2d(in_channels=hid1_feats, out_channels=hid2_feats, kernel_size=(2, 2), padding='same')
        self.bn2 = nn.BatchNorm2d(hid2_feats)
        self.fc3 = nn.Linear(hid2_feats, out_feats)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.cnn1(x)
        h = self.bn1(h)
        h = self.tanh(h)

        # h = self.dropout(h)

        h = self.cnn2(h)
        h = self.bn2(h)
        h = self.tanh(h)

        # print(h.shape)
        # h = self.dropout(h)

        # h = self.fc3(h)
        # h = torch.sigmoid(h)
        h = h.view(-1, self.hid2_feats)
        h = self.fc3(h)
        # print(h.shape)
        h = self.sigmoid(h)
        # print(h.shape)

        return h


class InnerProductDecoder(nn.Module):
    def forward(self, inputs):
        x = inputs.T
        x = torch.mm(inputs, x)
        x = torch.reshape(x, [-1])
        outputs = torch.sigmoid(x)
        return outputs


class GAE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.encoder = GCN(in_features, hidden_features, out_features)
        self.decoder = InnerProductDecoder()

    def forward(self, adj, feature):
        h = self.encoder(adj, feature)
        # h_noise = h + (0.1**0.5)*torch.randn(187,128)
        h_noise = self.decoder(h)
        # print('h: ',h)
        return h_noise, h


class MLP_P(nn.Module):
    def __init__(self, in_feats, out_feats, hid1_feats=1024, hid2_feats=512):
        super().__init__()

        self.mlp1 = nn.Linear(in_feats, hid1_feats)
        self.mlp2 = nn.Linear(hid1_feats, hid2_feats)
        self.mlp3 = nn.Linear(hid2_feats, out_feats)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.in_feats = in_feats

    def forward(self, x, method):

        x = x.view(-1, self.in_feats)
        out = self.mlp1(x)
        out = self.dropout(out)
        out = self.sigmoid(out)

        out = self.mlp2(out)
        out = self.dropout(out)
        out = self.sigmoid(out)
        out = self.mlp3(out)
        if method == 'relu':
            out = self.relu(out)
        elif method == 'sigmoid':
            out = self.sigmoid(out)

        return out


class MLP_T(nn.Module):
    def __init__(self, in_feats, out_feats, hid1_feats=256):
        super().__init__()

        self.mlp1 = nn.Linear(in_feats, hid1_feats)
        self.mlp2 = nn.Linear(hid1_feats, out_feats)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.in_feats = in_feats

    def forward(self, x, method):

        x = x.view(-1, self.in_feats)
        out = self.mlp1(x)
        out = self.dropout(out)
        out = self.sigmoid(out)

        out = self.mlp2(out)

        if method == 'relu':
            out = self.relu(out)
        elif method == 'sigmoid':
            out = self.sigmoid(out)

        return out


def permutation_importances(model, X, y, metric):
    baseline = metric(model, X, y)
    imp = []
    for col in X.columns:
        save = X[col].copy()
        X[col] = np.random.permutation(X[col])
        m = metric(model, X, y)
        X[col] = save
        imp.append(baseline - m)
    return np.array(imp)


def my_attention(h_list):
    h_list_save = h_list
    h_list = torch.tensor([item.cpu().detach().numpy() for item in h_list])

    pool_1 = nn.AvgPool2d(kernel_size=h_list.shape[2], stride=h_list.shape[2], padding=1)
    relu = nn.ReLU()
    # sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=0)
    h_pool = pool_1(h_list)
    h_pool = h_pool.squeeze(-2)
    h_pool = relu(h_pool)

    # print(h_pool)
    h_pool = h_pool.squeeze(-1)
    h_pool = softmax(h_pool)
    h_list_new = 0

    for i in range(len(h_pool)):
        # print(h_pool[i])
        h_list[i] = h_pool[i] * h_list[i]
        # print(h_list[i])
        h_list_new += h_list[i]
    # print(h_list_new)

    return h_list_new


def my_attention_single(h_list):
    h_list = torch.tensor([item.cpu().detach().numpy() for item in h_list])
    h_list_shape1 = h_list.shape[1]
    h_list_shape2 = h_list.shape[2]

    pool_1 = nn.AvgPool2d(kernel_size=(1, h_list.shape[2]), stride=1, padding=0)
    relu = nn.ReLU()
    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=0)
    h_pool = pool_1(h_list)
    # print(h_pool)
    h_pool = h_pool.squeeze(-2)
    h_pool = relu(h_pool)

    h_pool = h_pool.squeeze(-1)
    h_pool = softmax(h_pool)
    h_pool_t = h_pool.T
    h_list_new = []

    for i in range(h_pool_t.shape[0]):

        h_list_add = 0
        for j in range(len(h_pool_t[i])):
            h_list[j][i] = h_pool_t[i][j] * h_list[j][i]

            h_list_add += h_list[j][i]
        h_list_new.append(h_list_add)

    h_list_new = torch.tensor([item.cpu().detach().numpy() for item in h_list_new])
    h_list_new = torch.reshape(h_list_new, (h_list_shape1, h_list_shape2))

    return h_list_new
