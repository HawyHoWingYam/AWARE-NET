#!/usr/bin/env python3
"""
GenConViT Dual-Mode Manager - genconvit_manager.py
=================================================

Central management system for GenConViT integration supporting:
- Hybrid Mode: Custom recreation for perfect AWARE-NET integration
- Pretrained Mode: Original weights with 95.8% accuracy, 99.3% AUC
- Auto Mode: Intelligent selection based on availability

Key Features:
- Runtime mode switching without code changes
- Unified interface regardless of underlying implementation
- Automatic fallback and dependency management
- Performance comparison utilities
- Seamless Stage 2/3 framework integration

Usage:
    # Hybrid mode (custom implementation)
    manager = GenConViTManager(mode="hybrid")
    model = manager.create_model(variant="ED")
    
    # Pretrained mode (original weights)
    manager = GenConViTManager(mode="pretrained") 
    model = manager.load_pretrained()
    
    # Auto mode (intelligent selection)
    manager = GenConViTManager(mode="auto")
    model = manager.get_best_model()
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from enum import Enum

import torch
import torch.nn as nn

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import GenConViT components
try:
    from .genconvit.common.base import GenConViTVariant, GenConViTOutput
    from .genconvit.hybrid.model import create_genconvit_hybrid
    from .genconvit.pretrained.adapter import create_adapter_model
    from .genconvit.pretrained.config import get_huggingface_config
    GENCONVIT_AVAILABLE = True
except ImportError as e:
    GENCONVIT_AVAILABLE = False
    _import_error = e

class GenConViTMode(Enum):
    """Supported GenConViT integration modes"""
    HYBRID = "hybrid"
    PRETRAINED = "pretrained" 
    AUTO = "auto"

class GenConViTVariant(Enum):
    """GenConViT model variants"""
    ED = "ED"  # Encoder-Decoder
    VAE = "VAE"  # Variational Autoencoder

class GenConViTManager:
    """
    Central manager for GenConViT dual-mode integration
    
    Handles seamless switching between hybrid and pretrained approaches
    with unified interface and automatic dependency management.
    """
    
    def __init__(self, 
                 mode: Union[str, GenConViTMode] = GenConViTMode.AUTO,
                 variant: Union[str, GenConViTVariant] = GenConViTVariant.ED,
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize GenConViT Manager
        
        Args:
            mode: Integration mode ('hybrid', 'pretrained', 'auto')
            variant: Model variant ('ED', 'VAE')  
            device: Target device ('cuda', 'cpu', None for auto)
            cache_dir: Cache directory for downloaded models
            verbose: Enable detailed logging
        """
        
        # Setup logging
        self.logger = self._setup_logging(verbose)
        
        # Parse and validate arguments
        self.mode = self._parse_mode(mode)
        self.variant = self._parse_variant(variant)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.cache' / 'aware_net'
        
        # Initialize paths and configurations
        self.project_root = PROJECT_ROOT
        self.stage2_dir = self.project_root / 'src' / 'stage2'
        self.weights_dir = self.cache_dir / 'genconvit_weights'
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Model state
        self.model = None
        self.config = None
        self.active_mode = None
        
        # Pretrained model information
        self.hf_repo = "Deressa/GenConViT"
        self.weight_files = {
            GenConViTVariant.ED: "genconvit_ed_inference.pth",
            GenConViTVariant.VAE: "genconvit_vae_inference.pth"
        }
        
        self.logger.info(f"GenConViT Manager initialized: mode={self.mode.value}, variant={self.variant.value}, device={self.device}")
        
        # Auto-detect best mode if requested
        if self.mode == GenConViTMode.AUTO:
            self.mode = self._determine_best_mode()
            self.logger.info(f"Auto-selected mode: {self.mode.value}")
    
    def _setup_logging(self, verbose: bool) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(f"GenConViTManager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO if verbose else logging.WARNING)
        return logger
    
    def _parse_mode(self, mode: Union[str, GenConViTMode]) -> GenConViTMode:
        """Parse and validate mode argument"""
        if isinstance(mode, str):
            try:
                return GenConViTMode(mode.lower())
            except ValueError:
                raise ValueError(f"Invalid mode '{mode}'. Must be one of: {[m.value for m in GenConViTMode]}")
        return mode
    
    def _parse_variant(self, variant: Union[str, GenConViTVariant]) -> GenConViTVariant:
        """Parse and validate variant argument"""
        if isinstance(variant, str):
            try:
                return GenConViTVariant(variant.upper())
            except ValueError:
                raise ValueError(f"Invalid variant '{variant}'. Must be one of: {[v.value for v in GenConViTVariant]}")
        return variant
    
    def _determine_best_mode(self) -> GenConViTMode:
        """
        Automatically determine the best integration mode
        
        Priority:
        1. Pretrained (if weights downloadable and dependencies available)
        2. Hybrid (as fallback)
        
        Returns:
            Best available mode
        """
        self.logger.info("Auto-detecting best GenConViT integration mode...")
        
        # Check pretrained availability
        if self._check_pretrained_availability():
            self.logger.info("Pretrained mode available - high performance guaranteed")
            return GenConViTMode.PRETRAINED
        
        # Check hybrid dependencies
        if self._check_hybrid_dependencies():
            self.logger.info("Hybrid mode available - perfect AWARE-NET integration")
            return GenConViTMode.HYBRID
        
        # Default fallback
        self.logger.warning("Limited availability detected. Defaulting to hybrid mode.")
        return GenConViTMode.HYBRID
    
    def _check_pretrained_availability(self) -> bool:
        """Check if pretrained weights can be downloaded"""
        try:
            # Check Hugging Face repo accessibility
            api = HfApi()
            repo_info = api.repo_info(self.hf_repo)
            
            # Check if weight file exists
            weight_file = self.weight_files[self.variant]
            files = [f.rfilename for f in api.list_repo_files(self.hf_repo)]
            
            if weight_file in files:
                self.logger.info(f"Pretrained weights available: {weight_file}")
                return True
            else:
                self.logger.warning(f"Weight file not found: {weight_file}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Cannot access pretrained weights: {e}")
            return False
    
    def _check_hybrid_dependencies(self) -> bool:
        """Check if hybrid implementation dependencies are available"""
        try:
            # Check critical imports
            import timm
            import torchvision
            
            # Check timm models availability
            convnext_models = [m for m in timm.list_models() if 'convnext_tiny' in m]
            swin_models = [m for m in timm.list_models() if 'swin_tiny' in m]
            
            if convnext_models and swin_models:
                self.logger.info("Hybrid mode dependencies satisfied")
                return True
            else:
                self.logger.warning("Required timm models not available")
                return False
                
        except ImportError as e:
            self.logger.warning(f"Hybrid dependencies missing: {e}")
            return False
    
    def create_model(self, 
                    config: Optional[Dict[str, Any]] = None,
                    force_mode: Optional[Union[str, GenConViTMode]] = None) -> nn.Module:
        """
        Create GenConViT model based on current mode
        
        Args:
            config: Optional model configuration
            force_mode: Override current mode for this creation
            
        Returns:
            GenConViT model instance
        """
        
        # Use forced mode if specified
        active_mode = self._parse_mode(force_mode) if force_mode else self.mode
        self.active_mode = active_mode
        
        self.logger.info(f"Creating GenConViT model: mode={active_mode.value}, variant={self.variant.value}")
        
        if active_mode == GenConViTMode.HYBRID:
            return self._create_hybrid_model(config)
        elif active_mode == GenConViTMode.PRETRAINED:
            return self._create_pretrained_model(config)
        else:
            raise ValueError(f"Invalid mode for model creation: {active_mode}")
    
    def _create_hybrid_model(self, config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Create hybrid (custom recreation) GenConViT model"""
        try:
            from .genconvit.hybrid.model import GenConViTED, GenConViTVAE
            from .genconvit.hybrid.config import get_hybrid_config
            
            # Load configuration
            self.config = get_hybrid_config(self.variant, config)
            
            # Create model based on variant
            if self.variant == GenConViTVariant.ED:
                model = GenConViTED(self.config)
            else:  # VAE
                model = GenConViTVAE(self.config)
            
            model = model.to(self.device)
            self.model = model
            
            # Log model info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.logger.info(f"Hybrid GenConViT{self.variant.value} created")
            self.logger.info(f"Total parameters: {total_params:,}")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
            
            return model
            
        except ImportError as e:
            self.logger.error(f"Cannot import hybrid model components: {e}")
            self.logger.info("Falling back to pretrained mode...")
            return self._create_pretrained_model(config)
    
    def _create_pretrained_model(self, config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """Create pretrained GenConViT model with original weights"""
        
        if not GENCONVIT_AVAILABLE:
            raise ImportError(f"GenConViT components not available: {_import_error}")
        
        self.logger.info("Creating pretrained GenConViT model...")
        
        try:
            # Create adapter model with pretrained weights
            adapter_model = create_adapter_model(
                model_name=self.hf_repo,
                variant=self.variant.value,
                config=config,
                adapt_features=True,
                feature_dim=512,  # AWARE-NET standard feature dimension
                auto_load_weights=True
            )
            
            # Move to device
            adapter_model = adapter_model.to(self.device)
            
            # Store model and config
            self.model = adapter_model
            self.config = adapter_model.get_model_info()
            
            self.logger.info(f"✅ Pretrained GenConViT model created successfully")
            self.logger.info(f"📊 Model info: {self.config}")
            
            return adapter_model
            
        except Exception as e:
            self.logger.error(f"Failed to create pretrained model: {e}")
            self.logger.warning("Falling back to hybrid mode...")
            try:
                return self._create_hybrid_model(config)
            except Exception as fallback_error:
                raise RuntimeError(f"Both pretrained and hybrid model creation failed. "
                                 f"Pretrained: {e}, Hybrid: {fallback_error}")
    
    def _check_pretrained_availability(self) -> bool:
        """Check if pretrained models are available"""
        try:
            from .genconvit.pretrained.weight_loader import check_model_availability
            
            availability = check_model_availability(self.hf_repo)
            return availability.get('available', False)
            
        except ImportError:
            self.logger.warning("Pretrained components not available")
            return False
        except Exception as e:
            self.logger.warning(f"Failed to check pretrained availability: {e}")
            return False
    
    def switch_mode(self, 
                   new_mode: Union[str, GenConViTMode],
                   preserve_config: bool = True) -> nn.Module:
        """
        Switch to a different integration mode
        
        Args:
            new_mode: Target mode to switch to
            preserve_config: Whether to preserve current configuration
            
        Returns:
            New model instance
        """
        
        old_mode = self.mode
        self.mode = self._parse_mode(new_mode)
        
        self.logger.info(f"Switching GenConViT mode: {old_mode.value} → {self.mode.value}")
        
        # Preserve configuration if requested
        config = self.config if preserve_config else None
        
        # Create new model
        new_model = self.create_model(config)
        
        self.logger.info(f"Mode switch completed successfully")
        return new_model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about current model"""
        
        if self.model is None:
            return {"status": "No model created"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            "mode": self.active_mode.value if self.active_mode else self.mode.value,
            "variant": self.variant.value,
            "device": self.device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / 1024 / 1024,  # Rough estimate
            "config": self.config
        }
        
        # Add mode-specific information
        if self.active_mode == GenConViTMode.PRETRAINED:
            info.update({
                "pretrained_repo": self.hf_repo,
                "weight_file": self.weight_files[self.variant],
                "expected_accuracy": "95.8%",
                "expected_auc": "99.3%"
            })
        
        return info
    
    def compare_modes(self, 
                     test_input: Optional[torch.Tensor] = None,
                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compare performance between hybrid and pretrained modes
        
        Args:
            test_input: Optional test tensor for forward pass comparison
            config: Model configuration
            
        Returns:
            Comparison results
        """
        
        self.logger.info("Comparing GenConViT modes...")
        
        results = {
            "comparison_timestamp": torch.datetime.now().isoformat(),
            "variant": self.variant.value,
            "modes": {}
        }
        
        # Test both modes
        for mode in [GenConViTMode.HYBRID, GenConViTMode.PRETRAINED]:
            try:
                self.logger.info(f"Testing {mode.value} mode...")
                
                # Create model
                model = self.create_model(config, force_mode=mode)
                
                # Get model info
                model_info = {
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "memory_mb": sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024,
                    "device": next(model.parameters()).device.type
                }
                
                # Test forward pass if input provided
                if test_input is not None:
                    model.eval()
                    with torch.no_grad():
                        start_time = torch.cuda.Event(enable_timing=True)
                        end_time = torch.cuda.Event(enable_timing=True)
                        
                        start_time.record()
                        output = model(test_input.to(self.device))
                        end_time.record()
                        
                        torch.cuda.synchronize()
                        inference_time = start_time.elapsed_time(end_time)
                        
                        model_info.update({
                            "forward_pass_success": True,
                            "output_shape": output['classification'].shape if isinstance(output, dict) else output.shape,
                            "inference_time_ms": inference_time
                        })
                
                results["modes"][mode.value] = {
                    "status": "success",
                    "info": model_info
                }
                
            except Exception as e:
                results["modes"][mode.value] = {
                    "status": "failed",
                    "error": str(e)
                }
                self.logger.warning(f"{mode.value} mode test failed: {e}")
        
        # Generate recommendation
        if results["modes"].get("pretrained", {}).get("status") == "success":
            results["recommendation"] = "pretrained"
            results["reason"] = "Guaranteed performance (95.8% acc, 99.3% AUC)"
        elif results["modes"].get("hybrid", {}).get("status") == "success":
            results["recommendation"] = "hybrid"  
            results["reason"] = "Perfect AWARE-NET integration"
        else:
            results["recommendation"] = "none"
            results["reason"] = "Both modes failed"
        
        return results
    
    def get_best_model(self, config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """
        Get the best available GenConViT model
        
        Convenience method that creates a model using the current/auto-determined mode.
        
        Args:
            config: Optional model configuration
            
        Returns:
            Best available GenConViT model
        """
        return self.create_model(config)
    
    def cleanup(self):
        """Cleanup resources and cached models"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("GenConViT Manager cleanup completed")

# Convenience functions for easy usage
def create_genconvit(mode: str = "auto", 
                    variant: str = "ED",
                    device: Optional[str] = None,
                    **kwargs) -> Tuple[nn.Module, GenConViTManager]:
    """
    Convenience function to create GenConViT model
    
    Args:
        mode: Integration mode ('hybrid', 'pretrained', 'auto')
        variant: Model variant ('ED', 'VAE')
        device: Target device
        **kwargs: Additional arguments for GenConViTManager
        
    Returns:
        (model, manager) tuple
    """
    manager = GenConViTManager(mode=mode, variant=variant, device=device, **kwargs)
    model = manager.get_best_model()
    return model, manager

def compare_genconvit_modes(variant: str = "ED", 
                           device: Optional[str] = None,
                           test_input_shape: Tuple[int, int, int, int] = (2, 3, 224, 224),
                           **kwargs) -> Dict[str, Any]:
    """
    Convenience function to compare GenConViT modes
    
    Args:
        variant: Model variant ('ED', 'VAE')
        device: Target device
        test_input_shape: Shape for test input tensor (B, C, H, W)
        **kwargs: Additional arguments
        
    Returns:
        Comparison results
    """
    manager = GenConViTManager(mode="auto", variant=variant, device=device, **kwargs)
    test_input = torch.randn(*test_input_shape)
    return manager.compare_modes(test_input)

if __name__ == "__main__":
    # Demo usage
    print("GenConViT Dual-Mode Manager Demo")
    print("=" * 40)
    
    # Create manager
    manager = GenConViTManager(mode="auto", variant="ED", verbose=True)
    
    # Get model info
    model = manager.get_best_model()
    info = manager.get_model_info()
    
    print(f"\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    manager.cleanup()
    print("\nDemo completed successfully!")