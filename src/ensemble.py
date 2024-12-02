import timm
import torch.nn as nn
import torch

class EnsembleModel(nn.Module):
    def __init__(self, config, base_architecture='xception', num_models=3):
        super().__init__()
        # Multiple instances of same architecture
        self.models = nn.ModuleList([
            BaseModel(base_architecture) 
            for _ in range(num_models)
        ])
        
        # Equal weights for ensemble (typically)
        self.weights = torch.ones(num_models) / num_models
        
    def forward(self, x):
        # Get predictions from same architecture with different initializations
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
            
        # Simple averaging of predictions
        ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        return ensemble_pred

# Train multiple models with different initializations/architectures
models = [LightDetector().cuda() for _ in range(3)]
for model in models:
    train_model(model, train_loader, val_loader)

ensemble = EnsembleModel() 