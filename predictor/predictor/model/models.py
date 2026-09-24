 # -*- coding: utf-8 -*- 
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid激活函数
        self.batch_norm = nn.BatchNorm1d(hidden_size)
    def forward(self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor):
        # 拼接drug1、drug2和cell特征
        feat = torch.cat([drug1_feat, drug2_feat, cell_feat], dim=1)

        # 通过MLP层
        x = self.fc1(feat)
        x = self.sigmoid(x)  # 使用Sigmoid激活函数代替ReLU
        x = self.batch_norm(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)  # 使用Sigmoid激活函数代替ReLU
        out = self.fc3(x)
        return out
