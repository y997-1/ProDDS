# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from model.datasets import FastSynergyDataset
import pandas as pd

# 加载数据集（你的原始代码）
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

# 定义DNN（MLP）
class DNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64]):
        super(DNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Dropout(0.5),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Dropout(0.5),
            nn.Linear(hidden_sizes[1], 1)
        )

    def forward(self, x):
        return self.network(x).squeeze()

# 初始化DNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dnn_model = DNN(input_size=X_train.shape[1]).to(device)
optimizer = optim.Adam(dnn_model.parameters(), lr=0.0001)
criterion = nn.BCEWithLogitsLoss()

# 训练DNN
def train_dnn(model, X_train, y_train, epochs=20, batch_size=512):
    model.train()
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

train_dnn(dnn_model, X_train, y_train)

# DNN预测
dnn_model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_pred_prob = torch.sigmoid(dnn_model(X_test_tensor)).cpu().numpy()
    y_pred_dnn = (y_pred_prob > 0.5).astype(int)

# 计算DNN指标
dnn_acc = accuracy_score(y_test, y_pred_dnn)
dnn_precision = precision_score(y_test, y_pred_dnn)
dnn_recall = recall_score(y_test, y_pred_dnn)
dnn_f1 = f1_score(y_test, y_pred_dnn)
dnn_auc = roc_auc_score(y_test, y_pred_prob)
dnn_kappa = cohen_kappa_score(y_test, y_pred_dnn)

# 显示DNN结果
results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Kappa"],
    "DNN": [dnn_acc, dnn_precision, dnn_recall, dnn_f1, dnn_auc, dnn_kappa]
})

print(results)

# 保存DNN结果到txt文件
dnn_results_path = "dnn_results.txt"
with open(dnn_results_path, "w") as f:
    f.write("Deep Neural Network (DNN) Model Performance Metrics\n")
    f.write("="*50 + "\n")
    f.write(f"Accuracy: {dnn_acc:.4f}\n")
    f.write(f"Precision: {dnn_precision:.4f}\n")
    f.write(f"Recall: {dnn_recall:.4f}\n")
    f.write(f"F1 Score: {dnn_f1:.4f}\n")
    f.write(f"ROC AUC: {dnn_auc:.4f}\n")
    f.write(f"Kappa: {dnn_kappa:.4f}\n")

print(f"DNN results saved to {dnn_results_path}")


