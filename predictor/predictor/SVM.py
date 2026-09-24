# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from model.datasets import FastSynergyDataset
import pandas as pd

# 加载数据集
dataset = FastSynergyDataset(
    drug2id_file="/public/home/Yule1/Desktop/DeepLearning/PPI/predictor/data/drug2id.tsv",
    cell2id_file="/public/home/Yule1/Desktop/DeepLearning/PPI/predictor/data/cell2id.tsv",
    drug_feat_file="/public/home/Yule1/Desktop/DeepLearning/PPI/predictor/data/drug_feat.npy",
    cell_feat_file="/public/home/Yule1/Desktop/DeepLearning/PPI/predictor/data/cell_feat.npy",
    synergy_score_file="/public/home/Yule1/Desktop/DeepLearning/PPI/predictor/data/synergy.tsv",
    use_folds={0, 1, 2, 3},
    train=True
)

# 提取数据
d1, d2, c, y = dataset.tensor_samples()
X = torch.cat([d1, d2, c], dim=1).numpy()
y = y.numpy().flatten()

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为二分类 (0 或 1)
y_train = np.where(y_train >= 0.5, 1, 0)
y_test = np.where(y_test >= 0.5, 1, 0)

# 定义 CNN
class CNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(CNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Conv1d(in_channels=hidden_size // 2, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], 1)
        out = self.network(x)
        return out.view(out.shape[0])

# 初始化 CNN
cnn_model = CNN(input_size=X_train.shape[1], hidden_size=128)
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# 训练 CNN
def train_cnn(model, X_train, y_train, epochs=10):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

train_cnn(cnn_model, X_train, y_train)

# CNN 预测
cnn_model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_pred_cnn = torch.sigmoid(cnn_model(X_test_tensor)).detach().numpy()
y_pred_cnn = (y_pred_cnn > 0.5).astype(int)

# 训练 SVM
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)

# SVM 预测
y_pred_svm = svm_model.predict(X_test)
y_pred_svm_prob = svm_model.predict_proba(X_test)[:, 1]

# 计算 CNN 指标
cnn_acc = accuracy_score(y_test, y_pred_cnn)
cnn_precision = precision_score(y_test, y_pred_cnn)
cnn_recall = recall_score(y_test, y_pred_cnn)
cnn_f1 = f1_score(y_test, y_pred_cnn)
cnn_auc = roc_auc_score(y_test, y_pred_cnn)
cnn_kappa = cohen_kappa_score(y_test, y_pred_cnn)

# 计算 SVM 指标
svm_acc = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)
svm_auc = roc_auc_score(y_test, y_pred_svm_prob)
svm_kappa = cohen_kappa_score(y_test, y_pred_svm)

# 显示结果
results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Kappa"],
    "CNN": [cnn_acc, cnn_precision, cnn_recall, cnn_f1, cnn_auc, cnn_kappa],
    "SVM": [svm_acc, svm_precision, svm_recall, svm_f1, svm_auc, svm_kappa]
})


# 保存 SVM 结果到 txt 文件
svm_results_path = "svm_results.txt"
with open(svm_results_path, "w") as f:
    f.write("SVM Model Performance Metrics\n")
    f.write("="*40 + "\n")
    f.write(f"Accuracy: {svm_acc:.4f}\n")
    f.write(f"Precision: {svm_precision:.4f}\n")
    f.write(f"Recall: {svm_recall:.4f}\n")
    f.write(f"F1 Score: {svm_f1:.4f}\n")
    f.write(f"ROC AUC: {svm_auc:.4f}\n")
    f.write(f"Kappa: {svm_kappa:.4f}\n")

print(f"SVM results saved to {svm_results_path}")

