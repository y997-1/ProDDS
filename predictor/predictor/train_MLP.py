import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.datasets import FastSynergyDataset, FastTensorDataLoader
from model.models import MLP
from model.utils import find_best_model
from const import DRUG_FEAT_FILE, CELL_FEAT_FILE, SYNERGY_FILE, DRUG2ID_FILE, CELL2ID_FILE, OUTPUT_DIR

def train_model(out_dir, input_size=1600, hidden_size=4096, epochs=10, batch_size=128, lr=0.001):
    os.makedirs(out_dir, exist_ok=True)
    
    dataset = FastSynergyDataset(
        cell2id_file=CELL2ID_FILE,
        drug2id_file=DRUG2ID_FILE,
        drug_feat_file=DRUG_FEAT_FILE,
        cell_feat_file=CELL_FEAT_FILE,
        synergy_score_file=SYNERGY_FILE,
        use_folds=[False]
    )
    
    model = MLP(input_size=input_size, hidden_size=hidden_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = FastTensorDataLoader(*dataset.tensor_samples(), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for drug1, drug2, cell, label in loader:
            label = torch.tensor((label >= 0).float())  # 二分类标签
            label = label.view(-1)  # 确保 label 的维度是 [batch_size]

            optimizer.zero_grad()

            # 确保拼接输入特征
            input_feat = torch.cat([drug1, drug2, cell], dim=1)  # 拼接特征，确保尺寸为 [batch_size, 1600 + 768 + 1600]
            pred1 = model(input_feat)
            pred2 = model(input_feat)  # 可以修改为其他组合方法，如果需要不同的输入组合

            pred = (pred1 + pred2) / 2
            pred = pred.view(-1)  # 确保 pred 的维度是 [batch_size]

            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))
    print(f"模型保存到 {os.path.join(out_dir, 'best_model.pt')}")


if __name__ == "__main__":
    save_dir = os.path.join(OUTPUT_DIR, "mlp_model")
    train_model(save_dir)



