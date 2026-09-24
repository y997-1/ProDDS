# -*- coding: utf-8 -*-
import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score

from model.datasets import FastSynergyDataset, FastTensorDataLoader
from model.models import CNN
from model.utils import find_best_model
from const import SYNERGY_FILE, DRUG2ID_FILE, DRUG_FEAT_FILE, CELL_FEAT_FILE, CELL2ID_FILE, OUTPUT_DIR

# 如果细胞特征已预处理，请更新 CELL_FEAT_FILE 路径（例如指向预处理后的文件）
CELL_FEAT_FILE = "/public/home/Yule1/Desktop/DeepLearning/PPI/cell/GraphSAGEcell_feat_processed.npy"

def create_model(data, hidden_size, device=None):
    """
    根据数据构造 CNN 模型。输入维度为：cell_feat_len + 2 * drug_feat_len
    """
    model = CNN(data.cell_feat_len() + 2 * data.drug_feat_len(), hidden_size)
    if device is not None:
        model = model.to(device)
    return model

def evaluate_cv_results(cv_dir):
    n_folds = 5
    log_path = os.path.join(cv_dir, 'evaluation_results.txt')
    with open(log_path, 'w') as f:
        def print_to_file(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=f)
        
        # 存储各折指标和所有预测
        accuracies, precisions, recalls, f1s, roc_aucs, kappas = [], [], [], [], [], []
        all_y_true, all_y_pred = [], []
        
        print_to_file("Evaluation results for each fold:")
        for fold in range(n_folds):
            fold_dir = os.path.join(cv_dir, str(fold))
            if not os.path.isdir(fold_dir):
                continue
            
            # 加载测试数据时使用更新后的 CELL_FEAT_FILE
            test_data = FastSynergyDataset(
                cell2id_file=CELL2ID_FILE,
                drug2id_file=DRUG2ID_FILE,
                drug_feat_file=DRUG_FEAT_FILE,
                cell_feat_file=CELL_FEAT_FILE,
                synergy_score_file=SYNERGY_FILE,
                use_folds=[fold],
                train=False
            )
            best_model_path = find_best_model(fold_dir)
            if not best_model_path or not os.path.isfile(best_model_path):
                print_to_file(f"Fold {fold}: ERROR - best model file not found.")
                continue
            try:
                state_dict = torch.load(best_model_path, map_location=torch.device('cpu'))
            except Exception as e:
                print_to_file(f"Fold {fold}: ERROR - failed to load state dict. Exception: {e}")
                continue

            # 尝试候选 hidden size
            candidate_hidden_sizes = [4096, 8192, 2048, 512, 1024]
            model_loaded = False
            inferred_hidden = None
            for hs in candidate_hidden_sizes:
                try:
                    model = create_model(test_data, hs, device=None)
                    model.load_state_dict(state_dict)
                    model.eval()
                    model_loaded = True
                    inferred_hidden = hs
                    break
                except Exception as e:
                    continue
            if not model_loaded:
                print_to_file(f"Fold {fold}: ERROR - could not load model state dict with candidate hidden sizes.")
                continue
            
            # 生成预测结果（双向输入取平均）
            test_loader = FastTensorDataLoader(*test_data.tensor_samples(), batch_size=len(test_data))
            y_true_fold, y_pred_fold = [], []
            with torch.no_grad():
                for drug1_feats, drug2_feats, cell_feats, y_true in test_loader:
                    y_pred1 = model(drug1_feats.cpu(), drug2_feats.cpu(), cell_feats.cpu())
                    y_pred2 = model(drug2_feats.cpu(), drug1_feats.cpu(), cell_feats.cpu())
                    y_pred = (y_pred1 + y_pred2) / 2.0
                    y_true_fold.extend(y_true.cpu().numpy().flatten().tolist())
                    y_pred_fold.extend(y_pred.cpu().numpy().flatten().tolist())
            
            all_y_true.extend(y_true_fold)
            all_y_pred.extend(y_pred_fold)
            
            # 二分类映射（阈值为 0）
            y_true_bin = (np.array(y_true_fold) > 0).astype(int)
            y_pred_bin = (np.array(y_pred_fold) > 0).astype(int)
            acc = accuracy_score(y_true_bin, y_pred_bin)
            precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            roc_auc = roc_auc_score(y_true_bin, np.array(y_pred_fold))
            kappa = cohen_kappa_score(y_true_bin, y_pred_bin)
            accuracies.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            roc_aucs.append(roc_auc)
            kappas.append(kappa)
            
            print_to_file(f"Fold {fold}:")
            print_to_file(f"  Hidden size (inferred): {inferred_hidden}")
            print_to_file(f"  Accuracy:  {acc:.4f}")
            print_to_file(f"  Precision: {precision:.4f}")
            print_to_file(f"  Recall:    {recall:.4f}")
            print_to_file(f"  F1-Score:  {f1:.4f}")
            print_to_file(f"  ROC AUC:   {roc_auc:.4f}")
            print_to_file(f"  Cohen's Kappa: {kappa:.4f}")
            print_to_file("-" * 40)
        
        if accuracies:
            avg_acc = np.mean(accuracies)
            avg_prec = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_f1 = np.mean(f1s)
            avg_roc_auc = np.mean(roc_aucs)
            avg_kappa = np.mean(kappas)
            print_to_file("Average performance across all folds:")
            print_to_file(f"  Avg Accuracy:  {avg_acc:.4f}")
            print_to_file(f"  Avg Precision: {avg_prec:.4f}")
            print_to_file(f"  Avg Recall:    {avg_recall:.4f}")
            print_to_file(f"  Avg F1-Score:  {avg_f1:.4f}")
            print_to_file(f"  Avg ROC AUC:   {avg_roc_auc:.4f}")
            print_to_file(f"  Avg Cohen's Kappa: {avg_kappa:.4f}")
        else:
            print_to_file("No fold results were collected. Please check the folder structure or data.")
    
    with open(os.path.join(cv_dir, 'y_true.pkl'), 'wb') as ft:
        pickle.dump(all_y_true, ft)
    with open(os.path.join(cv_dir, 'y_pred.pkl'), 'wb') as fp:
        pickle.dump(all_y_pred, fp)
    print(f"\nSaved combined y_true to 'y_true.pkl' and y_pred to 'y_pred.pkl' in {cv_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate cross-validation results")
    parser.add_argument("--mdl_dir", type=str, required=True, help="Cross-validation output directory (e.g., cv_XXXXXXXXXX)")
    args = parser.parse_args()
    cv_output_dir = os.path.join(OUTPUT_DIR, args.mdl_dir)
    if not os.path.isdir(cv_output_dir):
        raise FileNotFoundError(f"Directory {cv_output_dir} does not exist.")
    evaluate_cv_results(cv_output_dir)

if __name__ == "__main__":
    main()
