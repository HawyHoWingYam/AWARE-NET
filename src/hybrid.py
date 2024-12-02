import torch
import torch.nn as nn
import timm
from pretrainedmodels import xception

class HybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Different architectures for hybrid approach
        self.xception = BaseModel('xception')  # Good at local features
        self.efficient_net = BaseModel('tf_efficientnet_b4')  # Efficient scaling
        self.swin = BaseModel('swin_base_patch4_window7_224')  # Global features
        
        # Learnable weights for hybrid combination
        self.weights = nn.Parameter(torch.ones(3))
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
    def forward(self, x):
        # Get predictions from different architectures
        x_pred = self.xception(x)
        e_pred = self.efficient_net(x)
        s_pred = self.swin(x)
        
        # Weighted combination of different architectures
        weights = torch.softmax(self.weights, dim=0)
        hybrid_pred = (weights[0] * x_pred + 
                      weights[1] * e_pred + 
                      weights[2] * s_pred)
        
        return hybrid_pred 