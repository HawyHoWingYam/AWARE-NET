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
        self.logger = logging.getLogger(__name__)
        
        # Base models configuration
        self.model_configs = {
            'legacy_xception': 'xception',  # timm_name: directory_name
            'res2net101_26w_4s': 'res2net101_26w_4s',
            'tf_efficientnet_b7_ns': 'tf_efficientnet_b7_ns'
        }
        
        # Initialize and load pre-trained models
        self.models = self._initialize_models(config)
        
        # Initialize ensemble weights (equal weights for each model)
        num_models = len(self.model_configs)
        self.model_weights = nn.Parameter(torch.ones(num_models) / num_models)
    
    def _initialize_models(self, config):
        """Initialize and load pre-trained models"""
        models = nn.ModuleList()
        
        for timm_name, dir_name in self.model_configs.items():
            # Create model
            model = BaseModel(timm_name, config, pretrained=False)
            
            # Load trained weights
            weights_path = self._get_weights_path(config, dir_name)
            self._load_model_weights(model, weights_path, timm_name)
            
            # Freeze model parameters
            for param in model.parameters():
                param.requires_grad = False
            
            models.append(model)
        
        return models
    
    def _get_weights_path(self, config, dir_name):
        """Get path to model weights"""
        weights_dir = config.RESULTS_DIR / 'weights' / 'ff++' / dir_name / 'no_aug'
        
        if not weights_dir.exists():
            raise FileNotFoundError(f"Missing weights directory: {weights_dir}")
        
        weight_files = list(weights_dir.glob('*.pth'))
        if not weight_files:
            raise FileNotFoundError(f"No weight files found in {weights_dir}")
        
        return weight_files[0]
    
    def _load_model_weights(self, model, weights_path, model_name):
        """Load pre-trained weights into model"""
        self.logger.info(f"Loading weights for {model_name} from {weights_path.name}")
        
        checkpoint = torch.load(weights_path)
        state_dict = checkpoint['model_state_dict']
        
        # Fix state dict keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.model.'):
                new_key = key.replace('model.model.', 'model.')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict)
        self.logger.info(f"Successfully loaded weights for {model_name}")
    
    def forward(self, x):
        """Forward pass using weighted average ensemble"""
        # Get predictions from each model
        predictions = []
        for model in self.models:
            pred = model(x)  # [batch_size, 2]
            pred = torch.softmax(pred, dim=1)  # Convert to probabilities
            predictions.append(pred)
        
        # Stack predictions: [batch_size, num_models, 2]
        stacked_preds = torch.stack(predictions, dim=1)
        
        # Apply learned weights
        weights = torch.softmax(self.model_weights, dim=0)  # Ensure weights sum to 1
        weighted_preds = stacked_preds * weights.view(1, -1, 1)
        
        # Average predictions
        ensemble_output = weighted_preds.sum(dim=1)  # [batch_size, 2]
        
        return ensemble_output
    
    def get_model_weights(self):
        """Get current model weights for visualization"""
        with torch.no_grad():
            weights = torch.softmax(self.model_weights, dim=0)
            return weights.cpu().numpy()