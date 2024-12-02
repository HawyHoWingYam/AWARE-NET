import torch
import torch.nn as nn
import timm
import logging

class BaseModel(nn.Module):
    def __init__(self, model_name, config, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        try:
            # Create model with 2 output classes (real/fake)
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=2,  # Binary classification
                drop_rate=config.DROPOUT_RATE   # Add dropout for regularization
            )
            
            # Log model size
            num_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Loaded {model_name} with {num_params:,} parameters")
            
        except Exception as e:
            self.logger.error(f"Error loading {model_name}: {str(e)}")
            raise
    
    def forward(self, x):
        return self.model(x)

class SingleModelDetector(nn.Module):
    def __init__(self, model_name, config):
        super().__init__()
        self.model = BaseModel(model_name, config)
        self.model_name = model_name
    
    def forward(self, x):
        return self.model(x)
    
    def get_model_weights(self):
        # For single models, return None or empty array since there are no ensemble weights
        return None

class EnsembleDeepfakeDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize architectures with config
        self.models = nn.ModuleList([
            BaseModel('xception', config),
            BaseModel('res2net101_26w_4s', config),
            BaseModel('tf_efficientnet_b7_ns', config)
        ])
        
        # Learnable weights for ensemble
        self.weights = nn.Parameter(torch.ones(3))
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE)
        )
        
    def forward(self, x):
        # Get predictions from each model
        predictions = []
        for model in self.models:
            pred = model(x)
            # Apply softmax to get probabilities
            pred = torch.softmax(pred, dim=1)[:, 1].unsqueeze(1)  # Get fake probability
            predictions.append(pred)
        
        # Stack predictions
        preds = torch.stack(predictions, dim=1)  # Shape: [batch_size, 3, 1]
        
        # Learn weights and combine
        weights = torch.softmax(self.fusion(preds.squeeze(-1)), dim=1)
        ensemble_pred = (weights[:, 0].unsqueeze(1) * predictions[0] + 
                        weights[:, 1].unsqueeze(1) * predictions[1] + 
                        weights[:, 2].unsqueeze(1) * predictions[2])
        
        return ensemble_pred
    
    def get_model_weights(self):
        with torch.no_grad():
            dummy_weights = self.fusion(torch.ones(1, 3))
            return torch.softmax(dummy_weights, dim=1).cpu().numpy() 