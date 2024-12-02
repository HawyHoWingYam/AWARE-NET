import torch
import torch.nn as nn
import timm
from pretrainedmodels import xception

class BaseModel(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model_name = model_name
        
        if model_name == 'xception':
            self.model = xception(pretrained='imagenet')
            self.model.last_linear = nn.Linear(2048, 1)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)

class HybridDeepfakeDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize architectures
        self.xception = BaseModel('xception')  # Local features
        self.efficient_net = BaseModel('tf_efficientnet_b4')  # Efficient scaling
        self.swin = BaseModel('swin_base_patch4_window7_224')  # Global features
        
        # Learnable weights for hybrid combination
        self.weights = nn.Parameter(torch.ones(3))
        
        # Feature fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(3, 3),  # Learn relationships between predictions
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE)
        )
        
    def forward(self, x):
        # Get predictions from different architectures
        x_pred = self.xception(x)
        e_pred = self.efficient_net(x)
        s_pred = self.swin(x)
        
        # Stack predictions
        preds = torch.stack([x_pred, e_pred, s_pred], dim=1)  # Shape: [batch_size, 3, 1]
        
        # Learn relationships and combine
        weights = torch.softmax(self.fusion(preds.squeeze(-1)), dim=1)
        hybrid_pred = (weights[:, 0].unsqueeze(1) * x_pred + 
                      weights[:, 1].unsqueeze(1) * e_pred + 
                      weights[:, 2].unsqueeze(1) * s_pred)
        
        return hybrid_pred
    
    def get_model_weights(self):
        with torch.no_grad():
            dummy_weights = self.fusion(torch.ones(1, 3))
            return torch.softmax(dummy_weights, dim=1).cpu().numpy() 