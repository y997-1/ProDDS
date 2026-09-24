# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, cohen_kappa_score
from model.datasets import FastSynergyDataset, FastTensorDataLoader
from model.models import MLP  
from model.utils import conf_inv, calc_stat, find_best_model
from const import SYNERGY_FILE, DRUG2ID_FILE, DRUG_FEAT_FILE, CELL_FEAT_FILE, CELL2ID_FILE, OUTPUT_DIR

# 定义模型和数据集
# 修改 create_model 函数，添加 test_mdl_dir 参数
def create_model(data, hidden_size, gpu_id=None, test_mdl_dir=None):
    input_size = 1600
    model = MLP(input_size, hidden_size)
    if gpu_id is not None:
        model = model.cuda(gpu_id)
    
    # 如果传递了 test_mdl_dir，则加载模型，忽略不匹配的键
    if test_mdl_dir:
        model.load_state_dict(torch.load(find_best_model(test_mdl_dir), map_location=torch.device('cpu')), strict=False)
    
    return model



def calc_metrics(out_dir):
    n_folds = 5
    n_delimiter = 60
    loss_func = nn.BCEWithLogitsLoss()  # 使用二分类损失函数
    test_losses = []
    pearson_coefs = []
    y_preds = []
    y_trues = []

    # 存储每个fold的评估指标
    accuracies = []
    precisions = []
    recalls = []
    roc_aucs = []
    f1s = []
    kappas = []

    # 初始化用于保存所有fold的y_true和y_pred
    all_y_true = []
    all_y_pred = []

    # 打开文件，准备写入输出
    with open(os.path.join(out_dir, 'GATevaluation_results.txt'), 'w') as f:
        # 重定向打印输出到文件
        def print_to_file(*args, **kwargs):
            print(*args, **kwargs)  # 打印到控制台
            print(*args, file=f, **kwargs)  # 打印到文件

        for test_fold in range(n_folds):
            test_data = FastSynergyDataset(
                cell2id_file=CELL2ID_FILE, 
                drug2id_file=DRUG2ID_FILE, 
                drug_feat_file=DRUG_FEAT_FILE, 
                cell_feat_file=CELL_FEAT_FILE,
                synergy_score_file=SYNERGY_FILE, 
                use_folds=[True]  # 或 False，根据数据集需求
            )
            
            test_mdl_dir = os.path.join(out_dir, str(test_fold))
            
            try:
                model = create_model(test_data, 4096, None, test_mdl_dir)
            except Exception:
                try:
                    model = create_model(test_data, 8192, None, test_mdl_dir)
                except Exception:
                    model = create_model(test_data, 2048, None, test_mdl_dir)

            test_loader = FastTensorDataLoader(*test_data.tensor_samples(), batch_size=len(test_data)) 
            model.eval()
            with torch.no_grad():
                for drug1_feats, drug2_feats, cell_feats, y_true in test_loader:
                    # 将连续值的标签转换为二分类标签
                    y_true = np.where(y_true >= 0, 1, 0)  # 大于等于 0 为 1，小于 0 为 0
                    
                    yp1 = model(drug1_feats, drug2_feats, cell_feats)
                    yp2 = model(drug2_feats, drug1_feats, cell_feats)
                    y_pred = (yp1 + yp2) / 2  # 计算预测值

                    # 确保 y_pred 和 y_true 是相同形状的张量
                    y_pred = y_pred.squeeze()  # 去掉多余的维度
                    y_true = torch.tensor(y_true, dtype=torch.float32).squeeze()  # 转换为张量并去除多余维度
                    
                    # 确保 y_true 和 y_pred 都是整数类型的数组
                    y_true = y_true.numpy().flatten()  # 转为数组
                    y_pred = y_pred.numpy().flatten()  # 转为数组

                    # 将预测值转换为二分类标签：大于 0 为 1，小于等于 0 为 0
                    y_pred = (y_pred > 0).astype(int)  # 如果 y_pred 大于 0，标签为 1，否则为 0
                    y_true = (y_true > 0).astype(int)  # 如果 y_true 大于 0，标签为 1，否则为 0

                    y_preds.extend(y_pred)  # 使用 extend 确保合并为一维数组
                    y_trues.extend(y_true)  # 使用 extend 确保合并为一维数组

                    # 将当前fold的y_true和y_pred添加到all_y_true和all_y_pred
                    all_y_true.extend(y_trues)
                    all_y_pred.extend(y_preds)

                    print_to_file(f"y_true shape: {np.shape(y_true)}")
                    print_to_file(f"y_pred shape: {np.shape(y_pred)}")
                    print_to_file("y_true:", y_true[:10])  # 打印前10个样本，避免输出过多
                    print_to_file("y_pred:", y_pred[:10])  # 打印前10个样本，避免输出过多

                    pc = np.corrcoef(y_pred, y_true)[0, 1]
                    pearson_coefs.append(pc)
                    test_loss = loss_func(torch.tensor(y_pred, dtype=torch.float32), torch.tensor(y_true, dtype=torch.float32)).item()  # 计算损失
                    test_loss /= len(y_true)
                    test_losses.append(test_loss)
                    print_to_file("Test fold: {} | Test loss: {:.2f} | Pearson Coef: {:.2f}".format(test_fold, test_loss, pc))

            # 计算各种指标
            accuracy = accuracy_score(y_trues, y_preds)
            precision = precision_score(y_trues, y_preds, average='binary')  # 二分类计算精确度
            recall = recall_score(y_trues, y_preds, average='binary')  # 二分类计算召回率
            roc_auc = roc_auc_score(y_trues, y_preds)
            f1 = f1_score(y_trues, y_preds, average='binary')  # 二分类计算F1分数
            kappa = cohen_kappa_score(y_trues, y_preds)

            # 将每个fold的评估结果添加到相应的列表
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            roc_aucs.append(roc_auc)
            f1s.append(f1)
            kappas.append(kappa)

            print_to_file(f"Accuracy: {accuracy:.2f}")
            print_to_file(f"Precision: {precision:.2f}")
            print_to_file(f"Recall: {recall:.2f}")
            print_to_file(f"ROC AUC: {roc_auc:.2f}")
            print_to_file(f"F1 Score: {f1:.2f}")
            print_to_file(f"Kappa Score: {kappa:.2f}")

        # 计算每个评估指标的平均值
        avg_accuracy = np.mean(accuracies)
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_roc_auc = np.mean(roc_aucs)
        avg_f1 = np.mean(f1s)
        avg_kappa = np.mean(kappas)

        # 输出每个指标的平均值
        print_to_file(f"\nAverage Accuracy: {avg_accuracy:.2f}")
        print_to_file(f"Average Precision: {avg_precision:.2f}")
        print_to_file(f"Average Recall: {avg_recall:.2f}")
        print_to_file(f"Average ROC AUC: {avg_roc_auc:.2f}")
        print_to_file(f"Average F1 Score: {avg_f1:.2f}")
        print_to_file(f"Average Kappa Score: {avg_kappa:.2f}")

    # 保存 y_true 和 y_pred 到文件
    with open(os.path.join(out_dir, 'y_true.pkl'), 'wb') as ft:
        pickle.dump(all_y_true, ft)
    with open(os.path.join(out_dir, 'y_pred.pkl'), 'wb') as fp:
        pickle.dump(all_y_pred, fp)
    print(f"\nSaved combined y_true to 'y_true.pkl' and y_pred to 'y_pred.pkl' in {out_dir}")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mdl_dir', type=str, help="模型目录")
    args = parser.parse_args()
    mdl_dir = os.path.join(OUTPUT_DIR, args.mdl_dir)
    calc_metrics(mdl_dir)

if __name__ == '__main__':
    main()



