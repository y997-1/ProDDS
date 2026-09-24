import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn.functional as F
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
import os

from process import *
from han_model import Model
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train():
    pass


def main():
    # load data
    drug_data = pd.read_csv('./data/g_smiles.csv')
    drug_src = np.array(drug_data['g_id1'])
    drug_dst = np.array(drug_data['g_id2'])
    similar_label = np.array(drug_data['simi'])
    synergy_label = np.array(drug_data['class'])

    n_drug = 1493
    n_synergy = 30964
    n_target = 726
    n_pathway = 2201

    target_data = pd.read_csv('./data/drug_target.csv')
    target_list = np.array(target_data['t_id'])
    drug_list = np.array(target_data['g_id'])

    drug_feature = pd.read_csv('./data/drug_feature.csv')
    drug_feat = torch.tensor(np.array(drug_feature))
    drug_feat = torch.reshape(drug_feat, (-1, 300))

    target_target_data = pd.read_csv('./data/target_target_id.csv')
    target1 = np.array(target_target_data['t_id1'])
    target2 = np.array(target_target_data['t_id2'])

    target_feature = pd.read_csv('./data/target_feature.csv')
    normalize = nn.Softmax(dim=0)

    target_feat = torch.tensor(np.array(target_feature))
    target_feat = normalize(target_feat)

    target_feat = target_feat.to(torch.float32)
    target_feat = torch.reshape(target_feat, (-1, 300))
    # print(target_feat)

    # Adjacent metrics
    DD_metrics = adj_create_label(drug_src, drug_dst, synergy_label, n_drug)
    DT_metrics = adj_create_edge(drug_list, target_list, n_drug, n_target)
    TT_metrics = adj_create_edge(target1, target2, n_target, n_target)

    # Meta path
    DTD_metrics = torch.tensor(metapath_metric('DTD', DT=DT_metrics))
    DTTD_metrics = torch.tensor(metapath_metric('DTTD', DT=DT_metrics, TT=TT_metrics))

    # Hetero graph
    # 构建异构图保证代号同即可
    # drug 和 target feature维度需要相同
    hetero_graph = dgl.heterograph({
        ('drug', 'ddi', 'drug'): (drug_src, drug_dst),
        ('drug', 'ddi', 'drug'): (drug_dst, drug_src),
        ('drug', 'dds', 'drug'): (drug_src, drug_dst),
        ('drug', 'dds', 'drug'): (drug_dst, drug_src),
        ('drug', 'dt', 'target'): (drug_list, target_list),
        ('target', 'td', 'drug'): (target_list, drug_list),
        ('target', 'tt', 'target'): (target1, target2),
        ('target', 'tt', 'target'): (target2, target1),
    })

    hetero_graph.edges['dds'].data['label'] = torch.tensor(synergy_label)
    hetero_graph.edges['ddi'].data['label'] = torch.tensor(similar_label)

    hetero_graph.nodes['drug'].data['feature'] = drug_feat
    hetero_graph.nodes['target'].data['feature'] = target_feat

    # model
    meta_paths = [['dt', 'td'], ['dt', 'tt', 'td']]
    model = Model(meta_paths, 300, 256, 128, 64, 32, 1, hetero_graph.etypes)

    node_features = {'drug': drug_feat, 'target': target_feat}
    label = hetero_graph.edges['dds'].data['label']
    label = torch.reshape(label, (-1, 1))
    label = label.to(torch.float32)
    # print("label: ", label)

    opt = torch.optim.Adam(model.parameters(), lr=0.0001)

    # opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99, weight_decay=1e-4)
    loss = nn.BCELoss()

    # loss = nn.CrossEntropyLoss()
    # loss = nn.NLLLoss()

    lenth = n_synergy
    pot = int(lenth / 5)
    print('lenth', lenth)
    print('pot', pot)

    random_num = random.sample(range(0, lenth), lenth)

    # print(random_num)
    for i_time in range(5):
        test_num = random_num[pot * i_time:pot * (i_time + 1)]
        train_num = random_num[:pot * i_time] + random_num[pot * (i_time + 1):]

        train_mask = bool_mask(n_synergy, train_num)
        test_mask = bool_mask(n_synergy, test_num)

        for epoch in range(10):

            # train
            model.train()

            # pred = model(hetero_graph, DTD_metrics, DTTD_metrics, node_features, drug_feat, drug_src, drug_dst)

            len_train = len(train_num)
            pot_train = int(len_train / 100)
            for j in range(62):
                train_batch = train_num[pot_train * j:pot_train * (j + 1)]
                train_batch_mask = bool_mask(n_synergy, train_batch)

                pred = model(hetero_graph, node_features, drug_feat.to(torch.float32), drug_src, drug_dst)

                loss_net = loss(pred[train_batch_mask], label[train_batch_mask])

                opt.zero_grad()
                loss_net.requires_grad_(True)
                loss_net.backward()
                opt.step()
                acc_train_batch = acc(pred[train_batch_mask], label[train_batch_mask], 0.5)

                # print('step: ', j, 'Loss: ', loss_net.item(), 'ACC_train: ', acc_train_batch)

                model.eval()
                acc_test_0 = acc(pred[test_mask], label[test_mask], 0.5)

                #  print('| i_time: ', i_time, '| epoch: ', epoch, '| step:', j, '| loss: ', loss_net.item())
                #  print('| ACC_train: ',acc_train_batch,'| ACC_test: ', acc_test_0)

                desktop_path = os.path.join(os.path.expanduser("~"), 'Desktop')
                file_path = os.path.join(desktop_path, 'training_han_test.txt')

                with open(file_path, 'a') as f:  # 'a' 模式代表追加模式，如果文件不存在则创建
                    f.write(f'| i_time: {i_time} | epoch: {epoch} | step: {j} | loss: {loss_net.item():.4f}\n')
                    f.write(f'| ACC_train: {acc_train_batch:.4f} | ACC_test: {acc_test_0:.4f}\n')

            #  metrics_draw_binary(label[test_mask], pred[test_mask], 'five_fold_batch')
            '''
            if epoch % 100 == 0:
                # metrics_draw_binary(label[test_mask], pred[test_mask],'five_fold_batch')
                pred_all = pred[test_mask]
                label_test = label[test_mask]
                for i in range(len(pred[test_mask])):
                    with open('./output/Han_fivefold_pred_test_%s.txt' % epoch, 'a') as f:
                        f.write(str(pred_all[i]))
                        f.write('\t')
                        f.write(str(label_test[i]))
                        f.write('\n')
           '''

if __name__ == '__main__':
    main()
