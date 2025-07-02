import torch
import torch.nn as nn
import timm
import logging
import os


class BaseModel(nn.Module):
    def __init__(self, timm_name, config, pretrained=True):
        super().__init__()
        self.timm_name = timm_name
        self.logger = logging.getLogger(__name__)

        try:
            # Create model with 2 output classes (real/fake)
            self.model = timm.create_model(
                timm_name,
                pretrained=pretrained,
                num_classes=2,  # Binary classification
                drop_rate=config.DROPOUT_RATE,  # Add dropout for regularization
            )

            # Log model size
            num_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Loaded {timm_name} with {num_params:,} parameters")

        except Exception as e:
            self.logger.error(f"Error loading {timm_name}: {str(e)}")
            raise

    def forward(self, x):
        return self.model(x)

    def load_pretrained_weights(self, weights_path):
        """Load pretrained weights, handling both full checkpoints and state dicts"""
        try:
            checkpoint = torch.load(weights_path)
            # If it's a full checkpoint with metadata
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            # If it's just the model state dict
            else:
                self.model.load_state_dict(checkpoint)
            return True
        except Exception as e:
            self.logger.warning(
                f"No pre-trained weights found for {self.timm_name}, using base initialization"
            )
            return False


class SingleModelDetector(nn.Module):
    def __init__(
        self, model_name, config, dataset=None, augment=False, variant_name=None
    ):
        super().__init__()
        # Initialize logger first
        self.logger = logging.getLogger(__name__)

        # Get the actual timm model name from config
        timm_model_name = config.MODELS[model_name]["timm_name"]
        self.model = BaseModel(timm_model_name, config)
        self.timm_name = model_name  # Store the config key instead of timm_name

    def forward(self, x):
        return self.model(x)

    def get_model_weights(self):
        # For single models, return None or empty array since there are no ensemble weights
        return None

    def load_weights(self, weights_dir):
        weights_files = list(weights_dir.glob("*.pth"))

        if weights_files:
            # Get the latest weights file
            weights_path = sorted(
                weights_files, key=lambda x: float(x.stem.split("_")[-1])
            )[0]
            self.logger.info(f"Found weights at: {weights_path}")

            try:
                # Load the checkpoint
                checkpoint = torch.load(weights_path, map_location="cpu")

                # Extract model state dict from checkpoint
                if isinstance(checkpoint, dict):
                    if "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                # Create new state dict with corrected keys
                new_state_dict = {}
                for key, value in state_dict.items():
                    # Remove the extra 'model.' prefix if present
                    if key.startswith("model.model."):
                        new_key = key.replace("model.model.", "model.")
                    elif key.startswith("model."):
                        new_key = key
                    else:
                        new_key = f"model.{key}"
                    new_state_dict[new_key] = value

                # Load the state dict
                try:
                    self.model.load_state_dict(new_state_dict)
                    self.logger.info(
                        f"Successfully loaded weights for {self.timm_name}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to load weights for {self.timm_name}: {str(e)}"
                    )
                    # Try loading with strict=False as fallback
                    self.model.load_state_dict(new_state_dict, strict=False)
                    self.logger.warning(
                        f"Loaded weights with strict=False for {self.timm_name}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error processing weights for {self.timm_name}: {str(e)}"
                )
                raise
        else:
            self.logger.warning(f"No pretrained weights found for {self.timm_name}")


class EnsembleDeepfakeDetector(nn.Module):
    def __init__(self, config, dataset=None, augment=False, variant_name=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.augment = augment
        self.dataset = dataset
        self.variant_name = variant_name
<<<<<<< HEAD
        
        # *** FIX: Initialize three instances for each architecture family ***
        self.models = self._initialize_and_freeze_models(config)
        
        # Learnable weights for each ARCHITECTURE FAMILY
        self.architecture_weights = nn.Parameter(torch.ones(len(self.models)))

        ensemble_weights_dir = config.get_model_weights_dir('ensemble', dataset, 'with_aug' if augment else 'no_aug')
        ensemble_weights_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_and_freeze_models(self, config):
        """
        Initializes three instances for each base model, loads their weights, and freezes them.
        """
        model_families = nn.ModuleList()
        base_model_keys = ['xception', 'res2net101_26w_4s', 'tf_efficientnet_b7_ns']

        for model_key in base_model_keys:
            family_instances = nn.ModuleList()
            # *** Create 3 instances for the current model family ***
            for i in range(3):
                self.logger.info(f"Initializing instance {i+1} for {model_key}")
                # Note: The paper implies 3 separately trained instances. 
                # Here, we load the same pre-trained base model weights into 3 instances.
                # For a true implementation, you would need to train each of the 9 base models
                # with different random seeds and save them separately.
                # This implementation provides the structural basis.
                model_instance = SingleModelDetector(
                    model_name=model_key,
                    config=config,
                    dataset=self.dataset,
                    augment=self.augment
                )
                
                weights_dir = config.get_model_weights_dir(model_key, self.dataset, 'with_aug' if self.augment else 'no_aug')
                if weights_dir.exists() and any(weights_dir.iterdir()):
                    model_instance.load_weights(weights_dir)
                else:
                    self.logger.warning(f"No weights found for {model_key}. Instance {i+1} will use its initial weights.")

                # Freeze all parameters in the instance
                for param in model_instance.parameters():
                    param.requires_grad = False
                
                family_instances.append(model_instance)
                
            model_families.append(family_instances)
            self.logger.info(f"Initialized and froze 3 instances for architecture family: {model_key}")
            
        return model_families

    def forward(self, x):
        """
        Forward pass with two-tier ensemble.
        """
        # Tier 1: Intra-Architecture Averaging
        architecture_predictions = []
        for family_instances in self.models:
            # Get predictions from each of the 3 instances in the family
            instance_preds = [instance(x) for instance in family_instances]
            instance_preds_softmax = [torch.softmax(p, dim=1) for p in instance_preds]
            
            # Average the predictions for the family
            avg_pred = torch.mean(torch.stack(instance_preds_softmax, dim=0), dim=0)
            architecture_predictions.append(avg_pred)
            
        # [batch_size, num_architectures, 2]
        stacked_preds = torch.stack(architecture_predictions, dim=1)
        
        # Tier 2: Inter-Architecture Adaptive Weighting
        # Apply learned weights to the averaged predictions from each family
        weights = torch.softmax(self.architecture_weights, dim=0)
        weighted_preds = stacked_preds * weights.view(1, -1, 1)
        
        # Final prediction
        ensemble_output = weighted_preds.sum(dim=1)
        
=======

        # *** FIX: Initialize three instances for each architecture family ***
        self.models = self._initialize_and_freeze_models(config)

        # Learnable weights for each ARCHITECTURE FAMILY
        self.architecture_weights = nn.Parameter(torch.ones(len(self.models)))

        ensemble_weights_dir = config.get_model_weights_dir(
            "ensemble", dataset, "with_aug" if augment else "no_aug"
        )
        ensemble_weights_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_and_freeze_models(self, config):
        """
        Initializes three instances for each base model, loads their weights, and freezes them.
        """
        model_families = nn.ModuleList()
        base_model_keys = ["xception", "res2net101_26w_4s", "tf_efficientnet_b7_ns"]

        for model_key in base_model_keys:
            family_instances = nn.ModuleList()
            # *** Create 3 instances for the current model family ***
            for i in range(3):
                self.logger.info(f"Initializing instance {i+1} for {model_key}")
                # Note: The paper implies 3 separately trained instances.
                # Here, we load the same pre-trained base model weights into 3 instances.
                # For a true implementation, you would need to train each of the 9 base models
                # with different random seeds and save them separately.
                # This implementation provides the structural basis.
                model_instance = SingleModelDetector(
                    model_name=model_key,
                    config=config,
                    dataset=self.dataset,
                    augment=self.augment,
                )

                weights_dir = config.get_model_weights_dir(
                    model_key, self.dataset, "with_aug" if self.augment else "no_aug"
                )
                if weights_dir.exists() and any(weights_dir.iterdir()):
                    model_instance.load_weights(weights_dir)
                else:
                    self.logger.warning(
                        f"No weights found for {model_key}. Instance {i+1} will use its initial weights."
                    )

                # Freeze all parameters in the instance
                for param in model_instance.parameters():
                    param.requires_grad = False

                family_instances.append(model_instance)

            model_families.append(family_instances)
            self.logger.info(
                f"Initialized and froze 3 instances for architecture family: {model_key}"
            )

        return model_families

    def forward(self, x):
        """
        Forward pass with two-tier ensemble.
        """
        # Tier 1: Intra-Architecture Averaging
        architecture_predictions = []
        for family_instances in self.models:
            # Get predictions from each of the 3 instances in the family
            instance_preds = [instance(x) for instance in family_instances]
            instance_preds_softmax = [torch.softmax(p, dim=1) for p in instance_preds]

            # Average the predictions for the family
            avg_pred = torch.mean(torch.stack(instance_preds_softmax, dim=0), dim=0)
            architecture_predictions.append(avg_pred)

        # [batch_size, num_architectures, 2]
        stacked_preds = torch.stack(architecture_predictions, dim=1)

        # Tier 2: Inter-Architecture Adaptive Weighting
        # Apply learned weights to the averaged predictions from each family
        weights = torch.softmax(self.architecture_weights, dim=0)
        weighted_preds = stacked_preds * weights.view(1, -1, 1)

        # Final prediction
        ensemble_output = weighted_preds.sum(dim=1)

        return ensemble_output

    def get_model_weights(self):
        with torch.no_grad():
            weights = torch.softmax(self.architecture_weights, dim=0)
            return weights.cpu().numpy()

    # class EnsembleDeepfakeDetector(nn.Module):
    def __init__(self, config, dataset=None, augment=False, variant_name=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.augment = augment
        self.dataset = dataset
        self.variant_name = variant_name

        # Initialize individual models
        self.models = self._initialize_models(config)

        # Initialize model weights (equal weighting by default)
        num_models = len(self.models)
        if num_models > 0:
            self.model_weights = nn.Parameter(torch.ones(num_models) / num_models)
        else:
            self.logger.error("No models were initialized for the ensemble")
            raise ValueError("No models were initialized for the ensemble")

        # Create ensemble weights directory
        ensemble_weights_dir = config.get_model_weights_dir(
            "ensemble", dataset, "with_aug" if augment else "no_aug"
        )
        ensemble_weights_dir.mkdir(parents=True, exist_ok=True)

    def _get_weights_path(self, config, dir_name):
        """Get path to model weights based on augmentation setting and dataset"""
        # Use the new config helper method
        weights_path = config.get_latest_weights_path(
            model_key=next(
                k for k, v in config.MODELS.items() if v["weights_dir"] == dir_name
            ),
            dataset=self.dataset,
            variant="with_aug" if self.augment else "no_aug",
        )

        if weights_path is None:
            self.logger.warning(f"No weights found for {dir_name} in {self.dataset}")
            return None

        self.logger.info(f"Found weights at: {weights_path}")
        return weights_path

    def _initialize_models(self, config):
        """Initialize individual models for the ensemble."""
        models = nn.ModuleList()

        for model_name, model_config in config.MODELS.items():
            try:
                # Skip ensemble and disabled models
                if model_name == "ensemble" or not model_config.get("enabled", True):
                    continue

                # Create single model detector
                model = SingleModelDetector(
                    model_name=model_name,  # Pass the model name correctly
                    config=config,
                    dataset=self.dataset,
                    augment=self.augment,
                    variant_name=self.variant_name,
                )

                # Load pre-trained weights if available
                weights_dir = config.get_model_weights_dir(
                    model_name, self.dataset, "with_aug" if self.augment else "no_aug"
                )
                if weights_dir.exists():
                    try:
                        model.load_weights(weights_dir)
                        self.logger.info(
                            f"Loaded pre-trained weights for {model_name} from {weights_dir}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Failed to load weights for {model_name}: {str(e)}"
                        )
                else:
                    self.logger.warning(
                        f"No weights directory found for {model_name} at {weights_dir}"
                    )

                models.append(model)
                self.logger.info(f"Added {model_name} to ensemble")

            except Exception as e:
                self.logger.error(f"Failed to initialize {model_name}: {str(e)}")
                continue

        if len(models) == 0:
            raise ValueError("No models were successfully initialized for the ensemble")

        return models

    def _load_model_weights(self, model, weights_path, model_name):
        """Load pre-trained weights into model"""
        self.logger.info(f"Loading weights for {model_name} from {weights_path.name}")

        try:
            checkpoint = torch.load(weights_path, map_location="cpu")

            # Handle both old and new checkpoint formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                self.logger.info(
                    f"Loading checkpoint from epoch {checkpoint['epoch']} "
                    f"with validation loss {checkpoint['val_loss']:.4f}"
                )
            else:
                state_dict = checkpoint

            # Fix state dict keys if needed
            if any(k.startswith("model.model.") for k in state_dict.keys()):
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("model.model."):
                        new_key = key.replace("model.model.", "model.")
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict

            # Load weights
            model.load_state_dict(state_dict)
            self.logger.info(f"Successfully loaded weights for {model_name}")

        except Exception as e:
            self.logger.error(f"Error loading weights for {model_name}: {str(e)}")
            raise

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
        weights = torch.softmax(self.model_weights, dim=0)  # Ensures weights sum to 1
        weighted_preds = stacked_preds * weights.view(1, -1, 1)

        # Average predictions
        ensemble_output = weighted_preds.sum(dim=1)  # [batch_size, 2]

>>>>>>> main
        return ensemble_output

    def get_model_weights(self):
        with torch.no_grad():
<<<<<<< HEAD
            weights = torch.softmax(self.architecture_weights, dim=0)
            return weights.cpu().numpy()
        
# class EnsembleDeepfakeDetector(nn.Module):
#     def __init__(self, config, dataset=None, augment=False, variant_name=None):
#         super().__init__()
#         self.logger = logging.getLogger(__name__)
#         self.augment = augment
#         self.dataset = dataset
#         self.variant_name = variant_name
        
#         # Initialize individual models and freeze them
#         self.models = self._initialize_and_freeze_models(config)
        
#         # Initialize model weights (these will be the only trainable parameters)
#         num_models = len(self.models)
#         if num_models > 0:
#             self.model_weights = nn.Parameter(torch.ones(num_models)) # Trainable
#         else:
#             self.logger.error("No models were initialized for the ensemble")
#             raise ValueError("No models were initialized for the ensemble")
        
#         # Create ensemble weights directory
#         ensemble_weights_dir = config.get_model_weights_dir('ensemble', dataset, 'with_aug' if augment else 'no_aug')
#         ensemble_weights_dir.mkdir(parents=True, exist_ok=True)
    
#     def _initialize_and_freeze_models(self, config):
#         """Initialize individual models for the ensemble, load their weights, and freeze them."""
#         models = nn.ModuleList()
        
#         # Define the base models to be used in the ensemble
#         base_model_keys = ['xception', 'res2net101_26w_4s', 'tf_efficientnet_b7_ns']

#         for model_key in base_model_keys:
#             try:
#                 # Create a single model detector instance for each base architecture
#                 model = SingleModelDetector(
#                     model_name=model_key,
#                     config=config,
#                     dataset=self.dataset,
#                     augment=self.augment,
#                     variant_name=self.variant_name
#                 )
                
#                 # Load pre-trained weights for this model
#                 weights_dir = config.get_model_weights_dir(model_key, self.dataset, 'with_aug' if self.augment else 'no_aug')
#                 if weights_dir.exists() and any(weights_dir.iterdir()):
#                     model.load_weights(weights_dir)
#                     self.logger.info(f"Successfully loaded pre-trained weights for {model_key}")
#                 else:
#                     self.logger.warning(f"No pre-trained weights found for {model_key} at {weights_dir}. Using initial weights.")

#                 # *** FREEZE THE MODEL ***
#                 for param in model.parameters():
#                     param.requires_grad = False
                
#                 models.append(model)
#                 self.logger.info(f"Initialized and froze model: {model_key}")
                
#             except Exception as e:
#                 self.logger.error(f"Failed to initialize or load {model_key}: {str(e)}")
#                 continue
        
#         if len(models) == 0:
#             raise ValueError("No models were successfully initialized for the ensemble.")
        
#         return models

#     def forward(self, x):
#         """Forward pass using weighted average ensemble"""
#         predictions = []
#         for model in self.models:
#             pred = model(x)
#             pred = torch.softmax(pred, dim=1)
#             predictions.append(pred)
        
#         stacked_preds = torch.stack(predictions, dim=1)
        
#         weights = torch.softmax(self.model_weights, dim=0)
#         weighted_preds = stacked_preds * weights.view(1, -1, 1)
        
#         ensemble_output = weighted_preds.sum(dim=1)
        
#         return ensemble_output
    
#     def get_model_weights(self):
#         """Get normalized weights for visualization"""
#         with torch.no_grad():
#             weights = torch.softmax(self.model_weights, dim=0)
#             return weights.cpu().numpy()
=======
            weights = torch.softmax(self.model_weights, dim=0)
            return weights.cpu().numpy()
>>>>>>> main
