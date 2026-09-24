import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(CNN, self).__init__()
        
        # Define the CNN layers
        self.network = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Conv1d(in_channels=hidden_size // 2, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, drug1_feat: torch.Tensor, drug2_feat: torch.Tensor, cell_feat: torch.Tensor):
        # Concatenate the features along the correct dimension
        feat = torch.cat([drug1_feat, drug2_feat, cell_feat], dim=1)
        
        # Ensure that the features have the correct shape for 1D convolution (batch_size, channels, length)
        feat = feat.unsqueeze(2)  # Add a dimension to simulate "sequence" length (e.g., [batch_size, channels, 1])
        
        out = self.network(feat)  # Forward pass through the CNN layers
        
        # After passing through the CNN layers, we flatten the output to match the final output dimension (batch_size, 1)
        out = out.squeeze(2)  # Remove the extra dimension created earlier
        
        return out
