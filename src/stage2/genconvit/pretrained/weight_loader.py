"""
GenConViT Weight Loader
======================

Utilities for downloading and loading pretrained GenConViT weights from
Hugging Face Hub and other sources.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import json
import warnings
from urllib.parse import urlparse

from .config import get_huggingface_config, get_model_download_info, list_available_models

class HuggingFaceWeightLoader:
    """
    Loader for GenConViT weights from Hugging Face Hub
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir: Directory for caching downloaded models
        """
        self.cache_dir = Path(cache_dir or "~/.cache/genconvit").expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to import huggingface_hub
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            self.hf_hub_download = hf_hub_download
            self.snapshot_download = snapshot_download
            self.hf_available = True
        except ImportError:
            self.hf_available = False
            warnings.warn(
                "huggingface_hub not available. Install with: pip install huggingface_hub",
                ImportWarning
            )
    
    def download_model(self, 
                      model_name: str = 'Deressa/GenConViT',
                      force_download: bool = False) -> Dict[str, Path]:
        """
        Download GenConViT model from Hugging Face
        
        Args:
            model_name: HuggingFace model identifier
            force_download: Whether to force re-download
            
        Returns:
            Dictionary of downloaded file paths
        """
        
        if not self.hf_available:
            raise ImportError("huggingface_hub required for downloading models")
        
        # Get download info
        download_info = get_model_download_info(model_name)
        repo_id = download_info['repo_id']
        
        print(f"Downloading GenConViT model from {repo_id}...")
        
        try:
            # Download entire repository
            local_dir = self.cache_dir / model_name.replace('/', '_')
            
            downloaded_path = self.snapshot_download(
                repo_id=repo_id,
                cache_dir=str(self.cache_dir),
                local_dir=str(local_dir),
                force_download=force_download,
                revision=download_info.get('revision', 'main')
            )
            
            downloaded_path = Path(downloaded_path)
            
            # Map file types to paths
            file_paths = {}
            
            # Look for common model files
            for file_type, filename in [
                ('config', 'config.json'),
                ('weights', 'pytorch_model.bin'),
                ('weights_safetensors', 'model.safetensors'),
                ('tokenizer', 'tokenizer.json')
            ]:
                file_path = downloaded_path / filename
                if file_path.exists():
                    file_paths[file_type] = file_path
            
            # Alternative weight file names
            weight_files = list(downloaded_path.glob('*.bin')) + list(downloaded_path.glob('*.pt'))
            if weight_files and 'weights' not in file_paths:
                file_paths['weights'] = weight_files[0]
            
            print(f"✅ Model downloaded to: {downloaded_path}")
            print(f"📁 Available files: {list(file_paths.keys())}")
            
            return file_paths
            
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            raise RuntimeError(f"Model download failed: {e}")
    
    def load_weights(self, 
                    model: nn.Module,
                    model_name: str = 'Deressa/GenConViT',
                    strict: bool = True,
                    device: Optional[str] = None) -> Dict[str, Any]:
        """
        Load pretrained weights into model
        
        Args:
            model: Model to load weights into
            model_name: Model identifier
            strict: Whether to strictly enforce state dict keys
            device: Target device for loading
            
        Returns:
            Loading information dictionary
        """
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Download model if not cached
        file_paths = self.download_model(model_name)
        
        if 'weights' not in file_paths:
            raise FileNotFoundError("No weight files found in downloaded model")
        
        weight_path = file_paths['weights']
        
        print(f"Loading weights from: {weight_path}")
        
        try:
            # Load state dict
            if weight_path.suffix == '.safetensors':
                # Handle safetensors format
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(str(weight_path), device=device)
                except ImportError:
                    raise ImportError("safetensors required for .safetensors files")\
            else:
                # Standard PyTorch format
                state_dict = torch.load(weight_path, map_location=device)
            
            # Handle nested state dicts
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Load into model
            loading_info = model.load_state_dict(state_dict, strict=strict)
            
            # Move model to device
            model = model.to(device)
            
            print(f"✅ Weights loaded successfully")
            if loading_info.missing_keys:
                print(f"⚠️  Missing keys: {loading_info.missing_keys}")
            if loading_info.unexpected_keys:
                print(f"⚠️  Unexpected keys: {loading_info.unexpected_keys}")
            
            return {
                'success': True,
                'weight_path': str(weight_path),
                'missing_keys': loading_info.missing_keys,
                'unexpected_keys': loading_info.unexpected_keys,
                'device': device
            }
            
        except Exception as e:
            print(f"❌ Failed to load weights: {e}")
            return {
                'success': False,
                'error': str(e),
                'weight_path': str(weight_path)
            }
    
    def load_config(self, model_name: str = 'Deressa/GenConViT') -> Dict[str, Any]:
        """
        Load model configuration from downloaded files
        
        Args:
            model_name: Model identifier
            
        Returns:
            Model configuration dictionary
        """
        
        file_paths = self.download_model(model_name)
        
        config = {}
        
        # Try to load config.json if available
        if 'config' in file_paths:
            try:
                with open(file_paths['config'], 'r') as f:
                    hf_config = json.load(f)
                config.update(hf_config)
                print(f"✅ Loaded config from: {file_paths['config']}")
            except Exception as e:
                print(f"⚠️  Failed to load config.json: {e}")
        
        # Merge with our predefined config
        try:
            predefined_config = get_huggingface_config(model_name)
            config.update(predefined_config)
        except Exception as e:
            print(f"⚠️  Failed to load predefined config: {e}")
        
        return config
    
    def list_cached_models(self) -> List[Dict[str, Any]]:
        """
        List models available in cache
        
        Returns:
            List of cached model information
        """
        
        cached_models = []
        
        if not self.cache_dir.exists():
            return cached_models
        
        for model_dir in self.cache_dir.iterdir():
            if model_dir.is_dir():
                # Check for model files
                has_weights = bool(list(model_dir.glob('*.bin')) + list(model_dir.glob('*.pt')))
                has_config = (model_dir / 'config.json').exists()
                
                cached_models.append({
                    'name': model_dir.name,
                    'path': str(model_dir),
                    'has_weights': has_weights,
                    'has_config': has_config,
                    'size_mb': sum(f.stat().st_size for f in model_dir.glob('*') if f.is_file()) / 1024 / 1024
                })
        
        return cached_models
    
    def clear_cache(self, model_name: Optional[str] = None):
        """
        Clear cached models
        
        Args:
            model_name: Specific model to clear, or None for all models
        """
        
        if model_name:
            model_dir = self.cache_dir / model_name.replace('/', '_')
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
                print(f"✅ Cleared cache for: {model_name}")
        else:
            if self.cache_dir.exists():
                import shutil
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                print(f"✅ Cleared all model cache")

# Convenience functions
def load_pretrained_weights(model: nn.Module,
                          model_name: str = 'Deressa/GenConViT',
                          cache_dir: Optional[str] = None,
                          device: Optional[str] = None,
                          strict: bool = True) -> Dict[str, Any]:
    """
    Convenience function to load pretrained weights
    
    Args:
        model: Model to load weights into
        model_name: HuggingFace model identifier  
        cache_dir: Cache directory for downloads
        device: Target device
        strict: Strict loading mode
        
    Returns:
        Loading result information
    """
    
    loader = HuggingFaceWeightLoader(cache_dir)
    return loader.load_weights(model, model_name, strict, device)

def get_available_models() -> List[Dict[str, Any]]:
    """
    Get list of available pretrained models
    
    Returns:
        List of available models
    """
    return list_available_models()

def download_model_weights(model_name: str = 'Deressa/GenConViT',
                          cache_dir: Optional[str] = None,
                          force_download: bool = False) -> Dict[str, Path]:
    """
    Download model weights without loading
    
    Args:
        model_name: Model identifier
        cache_dir: Cache directory
        force_download: Force re-download
        
    Returns:
        Downloaded file paths
    """
    
    loader = HuggingFaceWeightLoader(cache_dir)
    return loader.download_model(model_name, force_download)

def check_model_availability(model_name: str = 'Deressa/GenConViT') -> Dict[str, Any]:
    """
    Check if a model is available for download
    
    Args:
        model_name: Model identifier
        
    Returns:
        Availability information
    """
    
    try:
        download_info = get_model_download_info(model_name)
        
        # Try to ping the repository
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            repo_info = api.repo_info(download_info['repo_id'])
            
            return {
                'available': True,
                'repo_id': download_info['repo_id'],
                'last_modified': repo_info.lastModified,
                'size': getattr(repo_info, 'size', 'Unknown'),
                'files': len(getattr(repo_info, 'siblings', []))
            }
            
        except ImportError:
            return {
                'available': True,  # Assume available if we can't check
                'repo_id': download_info['repo_id'],
                'note': 'huggingface_hub not available for detailed checking'
            }
            
    except Exception as e:
        return {
            'available': False,
            'error': str(e)
        }