# -*- coding: utf-8 -*- 
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(CNN, self).__init__()
        hidden_mid = max(hidden_size // 2, 1)

        self.network = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_mid, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_mid),
            nn.Conv1d(in_channels=hidden_mid, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor):
        # 如果细胞特征为 3D，则对最后一维求均值
        if cell_feat.dim() == 3:
            cell_feat = cell_feat.mean(dim=-1)
        
        # 确保所有输入都是 2D
        assert drug1_feat.dim() == 2, f"drug1_feat expected 2D, got {drug1_feat.dim()}D"
        assert drug2_feat.dim() == 2, f"drug2_feat expected 2D, got {drug2_feat.dim()}D"
        assert cell_feat.dim() == 2, f"cell_feat expected 2D, got {cell_feat.dim()}D"

        feat = torch.cat([drug1_feat, drug2_feat, cell_feat], dim=1)
        feat = feat.unsqueeze(2)
        out = self.network(feat)
        out = out.squeeze(2)
        return out
