# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time  # 添加time模块
import os
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

import dgl
import torch.nn.functional as F
import sklearn
from sklearn import svm
import torch.utils.data as Data
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pylab as plt

from process import *
from han_model import Model
from model import MLP_P
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train():
    pass

def main():
    start_time = time.time()  # 开始计时

    # load data
    data_start_time = time.time()  # 数据加载开始计时
    drug_data = pd.read_csv('./data/modified_g_smiles.csv')
    drug_src = torch.tensor(np.array(drug_data['g_id1'])).to(device)
    drug_dst = torch.tensor(np.array(drug_data['g_id2'])).to(device)
    similar_label = torch.tensor(np.array(drug_data['simi'])).to(device)
    synergy_label = torch.tensor(np.array(drug_data['class'])).to(device)

    target_data = pd.read_csv('./data/drug_target.csv')
    target_list = torch.tensor(np.array(target_data['t_id'])).to(device)
    drug_list = torch.tensor(np.array(target_data['g_id'])).to(device)

    drug_feature = pd.read_csv('./data/drug_feature.csv')
    drug_feat = torch.tensor(np.array(drug_feature)).to(device)
    drug_feat = torch.reshape(drug_feat, (-1, 300))

    target_target_data = pd.read_csv('./data/target_target_id.csv')
    target1 = torch.tensor(np.array(target_target_data['t_id1'])).to(device)
    target2 = torch.tensor(np.array(target_target_data['t_id2'])).to(device)

    target_feature = pd.read_csv('./data/target_feature.csv')
    normalize = nn.Softmax(dim=0)
    target_feat = torch.tensor(np.array(target_feature)).to(device)
    target_feat = normalize(target_feat)
    target_feat = target_feat.to(torch.float32)
    target_feat = torch.reshape(target_feat, (-1, 300))

    target_pathway = pd.read_csv('./data/target_pathway_id.csv')
    target_id = torch.tensor(np.array(target_pathway['t_id'])).to(device)
    pathway_id = torch.tensor(np.array(target_pathway['p_id'])).to(device)

    pathway_feature = np.array(pd.read_csv('./data/p_feature.csv'))
    p_feature = torch.tensor(pathway_feature).to(device)
    p_feature = torch.reshape(p_feature, (-1, 9087))

    n_drug = 1493
    n_synergy = 21964
    n_target = 726
    n_pathway = 2201

    data_end_time = time.time()  # 数据加载结束计时
    print(f'Data loading time: {data_end_time - data_start_time:.2f} seconds')

    # Hetero graph
    graph_start_time = time.time()  # 图构建开始计时
    hetero_graph = dgl.heterograph({
        ('drug', 'ddi', 'drug'): (drug_src, drug_dst),
        ('drug', 'ddi', 'drug'): (drug_dst, drug_src),
        ('drug', 'dds', 'drug'): (drug_src, drug_dst),
        ('drug', 'dds', 'drug'): (drug_dst, drug_src),
        ('drug', 'dt', 'target'): (drug_list, target_list),
        ('target', 'td', 'drug'): (target_list, drug_list),
        ('target', 'tt', 'target'): (target1, target2),
        ('target', 'tt', 'target'): (target2, target1),
        ('target', 'tp', 'pathway'): (target_id, pathway_id),
        ('pathway', 'pt', 'target'): (pathway_id, target_id),
    }).to(device)  # 将图复制到同一个设备上

    graph_end_time = time.time()  # 图构建结束计时
    print(f'Graph construction time: {graph_end_time - graph_start_time:.2f} seconds')

    # Print device information for the graph
    print(f'hetero_graph device: {hetero_graph.device}')

    hetero_graph.edges['dds'].data['label'] = synergy_label
    hetero_graph.edges['ddi'].data['label'] = similar_label

    hetero_graph.nodes['drug'].data['feature'] = drug_feat
    hetero_graph.nodes['target'].data['feature'] = target_feat

    # model
    model_start_time = time.time()  # 模型构建开始计时
    meta_paths = [['dt', 'td'], ['dt', 'tt', 'td'], ['dt', 'tp', 'pt', 'td']]
    model = Model(meta_paths, 300, 256, 128, 64, 32, 16, 4, 1, hetero_graph.etypes).to(device)
    mlp = MLP_P(9087, 300).to(device)

    label = hetero_graph.edges['dds'].data['label']
    label = torch.reshape(label, (-1, 1))
    label = label.to(torch.float32)

    opt_model = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 0.0001},
        {'params': mlp.parameters(), 'lr': 0.0001},
    ])

    loss = nn.BCELoss()

    model_end_time = time.time()  # 模型构建结束计时
    print(f'Model construction time: {model_end_time - model_start_time:.2f} seconds')

    lenth = n_synergy
    pot = int(lenth / 5)
    print('lenth', lenth)
    print('pot', pot)

    random_num = random.sample(range(0, lenth), lenth)
    
    # 定义日志文件路径，保存在程序所在目录
    file_path = os.path.join(os.getcwd(), 'training_gpu.txt')

    # 保存各折测试指标，便于后续统计平均值
    all_fold_metrics = []

    for i_time in range(5):
        test_num = random_num[pot * i_time:pot * (i_time + 1)]
        train_num = random_num[:pot * i_time] + random_num[pot * (i_time + 1):]

        train_mask = bool_mask(n_synergy, train_num)
        test_mask = bool_mask(n_synergy, test_num)

        for epoch in range(100):
            epoch_start_time = time.time()  # 每个epoch开始计时

            # train
            model.train()
            mlp.train()

            len_train = len(train_num)
            pot_train = int(len_train / 62)
            for j in range(62):
                train_batch = train_num[pot_train * j:pot_train * (j + 1)]
                train_batch_mask = bool_mask(n_synergy, train_batch)

                p_feat = mlp(p_feature.to(torch.float32), 'relu')
                node_features = {'drug': drug_feat, 'target': target_feat, 'pathway': p_feat}

                pred = model(hetero_graph, node_features, drug_feat.to(torch.float32), drug_src, drug_dst)

                loss_net = loss(pred[train_batch_mask], label[train_batch_mask])

                opt_model.zero_grad()
                loss_net.backward()
                opt_model.step()
                acc_train_batch = acc(pred[train_batch_mask], label[train_batch_mask], 0.5)

                model.eval()
                mlp.eval()
                acc_test_0 = acc(pred[test_mask], label[test_mask], 0.5)

                with open(file_path, 'a') as f:  # 追加模式写入日志
                    f.write(f'| i_time: {i_time} | epoch: {epoch} | step: {j} | loss: {loss_net.item():.4f}\n')
                    f.write(f'| ACC_train: {acc_train_batch:.4f} | ACC_test: {acc_test_0:.4f}\n')
            epoch_end_time = time.time()
            with open(file_path, 'a') as f:
                f.write(f'Epoch {epoch} time: {epoch_end_time - epoch_start_time:.2f} seconds\n')
        
        # 每折结束后，对当前折测试集进行一次完整评估，计算各项指标
        model.eval()
        mlp.eval()
        with torch.no_grad():
            p_feat = mlp(p_feature.to(torch.float32), 'relu')
            node_features = {'drug': drug_feat, 'target': target_feat, 'pathway': p_feat}
            pred_fold = model(hetero_graph, node_features, drug_feat.to(torch.float32), drug_src, drug_dst)
            pred_test = pred_fold[test_mask]
            label_test = label[test_mask]
            # 转换为二值预测（阈值0.5）
            pred_test_binary = (pred_test >= 0.5).float().cpu().numpy().flatten()
            label_test_np = label_test.cpu().numpy().flatten()

        # 计算指标（使用sklearn的metrics）
        acc_val = metrics.accuracy_score(label_test_np, pred_test_binary)
        precision_val = metrics.precision_score(label_test_np, pred_test_binary)
        recall_val = metrics.recall_score(label_test_np, pred_test_binary)
        f1_val = metrics.f1_score(label_test_np, pred_test_binary)
        # ROC AUC计算时需要预测的概率值
        roc_auc_val = metrics.roc_auc_score(label_test_np, pred_test.cpu().numpy().flatten())
        kappa_val = metrics.cohen_kappa_score(label_test_np, pred_test_binary)

        # 将当前折的指标保存
        all_fold_metrics.append((acc_val, precision_val, recall_val, f1_val, roc_auc_val, kappa_val))
        with open(file_path, 'a') as f:
            f.write(f'\n--- Fold {i_time} Evaluation Metrics ---\n')
            f.write(f'Accuracy: {acc_val:.4f}\n')
            f.write(f'Precision: {precision_val:.4f}\n')
            f.write(f'Recall: {recall_val:.4f}\n')
            f.write(f'F1 Score: {f1_val:.4f}\n')
            f.write(f'ROC AUC: {roc_auc_val:.4f}\n')
            f.write(f'Kappa Coefficient: {kappa_val:.4f}\n')
            f.write('---------------------------------------\n')

    # 可选：计算所有折的平均指标
    all_fold_metrics = np.array(all_fold_metrics)
    avg_metrics = np.mean(all_fold_metrics, axis=0)
    with open(file_path, 'a') as f:
        f.write('\n=== Average Evaluation Metrics over all folds ===\n')
        f.write(f'Accuracy: {avg_metrics[0]:.4f}\n')
        f.write(f'Precision: {avg_metrics[1]:.4f}\n')
        f.write(f'Recall: {avg_metrics[2]:.4f}\n')
        f.write(f'F1 Score: {avg_metrics[3]:.4f}\n')
        f.write(f'ROC AUC: {avg_metrics[4]:.4f}\n')
        f.write(f'Kappa Coefficient: {avg_metrics[5]:.4f}\n')
        f.write('===============================================\n')

    end_time = time.time()
    with open(file_path, 'a') as f:
        f.write(f'Total execution time: {end_time - start_time:.2f} seconds\n')

if __name__ == '__main__':
    main()
