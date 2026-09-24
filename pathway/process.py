import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
import itertools
import sklearn
from sklearn import svm
import torch.utils.data as Data
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, f1_score, recall_score, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pylab as plt
import random
import matplotlib
import csv

matplotlib.use('Agg')


def Importances(h1, h2, h3, mark):
    if mark == 'P':
        h_save = h1
        h1 = torch.tensor(np.random.permutation(h1))
        print(h1)
        h23 = h1 + h2 + h3
        h1 = h_save

        h_save = h2
        h2 = torch.tensor(np.random.permutation(h2))
        print(h2)
        h13 = h1 + h2 + h3
        h2 = h_save

        h_save = h3
        h3 = torch.tensor(np.random.permutation(h3))
        print(h3)
        h12 = h1 + h2 + h3
        h3 = h_save

    elif mark == 'M':

        h12 = h1 + h2
        h13 = h1 + h3
        h23 = h2 + h3

    return [h23, h13, h12]


def bool_reverse(bool):
    not_function = lambda item: not item
    bool_not = list(map(not_function, bool))

    return bool_not


def adj_create_label(src, dst, label, num):
    adj_x = np.zeros(shape=(num, num))

    for i in range(0, len(src)):
        if label[i]:
            src_i1 = int(src[i])
            dst_j1 = int(dst[i])

            adj_x[src_i1][dst_j1] = 1
            adj_x[dst_j1][src_i1] = 1

    return adj_x


def adj_create_edge(src, dst, num1, num2, mark=None):
    adj_x = np.zeros(shape=(num1, num2))

    for i in range(len(src)):

        src_i1 = int(src[i])
        dst_i1 = int(dst[i])

        adj_x[src_i1][dst_i1] = 1
        if mark == 'same':
            adj_x[dst_i1][src_i1] = 1
        else:
            pass

    return adj_x


def metapath_metric(meta_type, DT=None, TT=None, TP=None):
    if meta_type == 'DTD':
        DTD_metric = np.dot(DT, np.transpose(DT))
        return DTD_metric
    if meta_type == 'DTTD':
        DTTD_metric = np.dot(np.dot(DT, TT), np.transpose(DT))
        return DTTD_metric
    if meta_type == 'DTPTD':
        DTPTD_metric = np.dot(np.dot(DT, TP), np.dot(np.transpose(TP), np.transpose(DT)))
        return DTPTD_metric


def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = (data - minVals) / ranges
    return normData


def bool_mask(n_list, list_bool):
    list_mask = np.zeros(n_list, dtype=bool)
    for i in range(len(list_bool)):
        list_mask[list_bool[i]] = True

    return list_mask


'''
def acc(pred, labels, std, binary=False):
    count = 0

    if binary == True:
        pred_all = []
        pred_new = pred.detach().numpy()

        for i in range(len(pred_new)):
            pred_all.append(pred_new[i][1])
        pred_all = torch.tensor(pred_all)
    else:
        pred_all = pred

    a = torch.where(pred_all > std, 1, 0).type(torch.int32)
    b = labels.type(torch.int32)
    for i in torch.eq(a, b):
        if i:
            count += 1

    acc_all = count / len(pred)
    return acc_all
'''


def acc(pred, labels, threshold):  # acc调整
    pred_label = (pred > threshold).int()  # 将预测概率转换为类别标签
    correct = (pred_label == labels).float()  # 计算正确预测的浮点数
    acc_all = correct.sum() / len(pred)  # 计算准确率
    return acc_all


def print_pred(pred, num):
    pred_np1 = pred.detach().numpy()
    for i in range(len(pred_np1)):
        if pred_np1[i] >= 0.9:
            pred_np1[i] = 1
        else:
            pred_np1[i] = 0
    pred_tensor = torch.tensor(pred_np1)
    pred_new1 = torch.reshape(pred_tensor, [num, num])

    return pred_new1


def metrics_draw(true, pred, name):
    true = true.detach().numpy()
    pred = pred.detach().numpy()

    Truelist = []
    Problist = []
    for i in range(len(true)):
        Truelist.append(true[i][0])
        Problist.append(pred[i][0])

    Problist_int = []
    for i in range(len(Problist)):
        if Problist[i] >= 0.5:
            Problist_int.append(1)
        else:
            Problist_int.append(0)
    # print(Problist_int)

    precision_scores = metrics.precision_score(Truelist, Problist_int)
    recall_scores = metrics.recall_score(Truelist, Problist_int)
    f1_scores = metrics.f1_score(Truelist, Problist_int)
    print('f1_scores:', f1_scores)
    print('precision_scores:', precision_scores)
    print('recall_scores:', recall_scores)

    with open('./output/metrics.txt', 'a') as f:
        f.write('f1_scores:')
        f.write(str(f1_scores))
        f.write('\r\n')
        f.write('precision_scores:')
        f.write(str(precision_scores))
        f.write('\r\n')
        f.write('recall_scores:')
        f.write(str(recall_scores))
        f.write('\r\n')

    precision, recall, _ = metrics.precision_recall_curve(Truelist, Problist)
    pr_auc = metrics.auc(recall, precision)
    print('pr_auc:', pr_auc)

    plt.figure(1)
    plt.plot(recall, precision, 'g', label='AUPR = %0.4f' % pr_auc)
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('Recall')  # 横坐标是fpr
    plt.ylabel('Precision')  # 纵坐标是tpr
    plt.title('Precision Recall Curve')
    plt.savefig('./%sAUPR' % name + '.jpg')

    fpr, tpr, thresholds = metrics.roc_curve(Truelist, Problist, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)  # auc为Roc曲线下的面积
    print('roc_auc:', roc_auc)

    plt.figure(2)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.title('Receiver operating characteristic')
    plt.savefig('./%sAUC' % name + '.jpg')
    # plt.show()


def metrics_draw_binary(true, pred, name):
    true = true.detach().numpy()
    pred = pred.detach().numpy()

    Truelist = true
    Problist = pred
    '''
    for i in range(len(pred)):
        Problist.append(pred[i][1])
    '''

    Problist_int = []
    for i in range(len(Problist)):  # 遍历预测概率，并将大于或等于0.5的概率转换为整数标签1，小于0.5的转换为0
        if Problist[i] >= 0.5:
            Problist_int.append(1)
        else:
            Problist_int.append(0)

    # print(Truelist)
    # print(Problist_int)

    precision_scores = metrics.precision_score(Truelist, Problist_int)
    recall_scores = metrics.recall_score(Truelist, Problist_int)
    f1_scores = metrics.f1_score(Truelist, Problist_int)
    print('f1_scores:', f1_scores)
    print('precision_scores:', precision_scores)
    print('recall_scores:', recall_scores)

    precision, recall, _ = metrics.precision_recall_curve(Truelist, Problist)
    pr_auc = metrics.auc(recall, precision)
    print('pr_auc:', pr_auc)

    plt.figure(1)
    plt.plot(recall, precision, 'g', label='AUPR = %0.4f' % pr_auc)
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('Recall')  # 横坐标是fpr
    plt.ylabel('Precision')  # 纵坐标是tpr
    plt.title('Precision Recall Curve')
    plt.savefig('./%sAUPR' % name + '.jpg')

    fpr, tpr, thresholds = metrics.roc_curve(Truelist, Problist, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)  # auc为Roc曲线下的面积
    print('roc_auc:', roc_auc)

    plt.figure(2)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')  # 横坐标是fpr
    plt.ylabel('True Positive Rate')  # 纵坐标是tpr
    plt.title('Receiver operating characteristic')
    plt.savefig('./%sAUC' % name + '.jpg')
    # plt.show()


def metric_scores(y_values_all, probas_all, predictions_all):
    # print(y_values_all)
    # print(predictions_all)

    aucs = [roc_auc_score(y, proba) for y, proba in zip(y_values_all, probas_all)]
    accs = [accuracy_score(y, pred) for y, pred in zip(y_values_all, predictions_all)]

    for prob_i in probas_all:
        # print(prob_i)
        for i in range(len(prob_i)):
            # print(prob_i[i])
            if prob_i[i] > 0.5:
                prob_i[i] = 1
            else:
                prob_i[i] = 0

    precision_scores = [precision_score(y, proba) for y, proba in zip(y_values_all, probas_all)]
    recall_scores = [recall_score(y, proba) for y, proba in zip(y_values_all, probas_all)]
    f1_scores = [f1_score(y, proba) for y, proba in zip(y_values_all, probas_all)]

    pr_all = []
    for i in range(len(predictions_all)):
        precision, recall, _ = metrics.precision_recall_curve(y_values_all[i], predictions_all[i])
        pr_auc = metrics.auc(recall, precision)
        pr_all.append(pr_auc)

    # pr_auc = metrics.auc(np.mean(recall_scores), np.mean(precision_scores))

    print('accuracy: ', np.mean(accs), accs)
    print('roc_auc :', np.mean(aucs), aucs)
    print('pr_auc: ', np.mean(pr_auc), pr_all)
    print('precision_scores: ', np.mean(precision_scores), precision_scores)
    print('recall scores: ', np.mean(recall_scores), recall_scores)
    print('f1_scores: ', np.mean(f1_scores), f1_scores)


def multi_models_roc(names, sampling_methods, colors, X_test, y_test, save=True, dpin=100):
    """
    将多个机器模型的roc图输出到一张图上

    Args:
        names: list, 多个模型的名称
        sampling_methods: list, 多个模型的实例化对象
        save: 选择是否将结果保存（默认为png格式）

    Returns:
        返回图片对象plt
    """
    plt.figure(figsize=(20, 20), dpi=dpin)

    for (name, method, colorname) in zip(names, sampling_methods, colors):
        y_test_preds = method.predict(X_test)
        y_test_predprob = method.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)

        plt.plot(fpr, tpr, lw=5, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)
        plt.plot([0, 1], [0, 1], '--', lw=5, color='grey')
        plt.axis('square')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.title('ROC Curve', fontsize=25)
        plt.legend(loc='lower right', fontsize=20)

    if save:
        plt.savefig('multi_models_roc.png')

    return plt
