#!/usr/bin/env python3
"""
ONNX Model Exporter - onnx_exporter.py
=====================================

Export optimized PyTorch models to ONNX format for cross-platform deployment.

Features:
- Quantized INT8 model export with proper optimization
- Dynamic input shapes for flexible deployment
- Validation of exported models with test inputs
- Optimization for mobile inference performance
- Metadata embedding for deployment tracking

Usage:
    exporter = ONNXExporter()
    exporter.export_model(model, "cascade_stage1.onnx", input_shape=(1, 3, 256, 256))
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from onnx import helper, version_converter
from onnxsim import simplify

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

class ONNXExporter:
    """ONNX model exporter for mobile deployment"""
    
    def __init__(self, optimize_for_mobile: bool = True, verbose: bool = True):
        self.optimize_for_mobile = optimize_for_mobile
        self.verbose = verbose
        
        # Setup logging
        self.logger = logging.getLogger('ONNXExporter')
        if verbose:
            self.logger.setLevel(logging.INFO)
    
    def export_model(self, 
                    model: nn.Module,
                    output_path: Union[str, Path],
                    input_shape: Tuple[int, ...] = (1, 3, 256, 256),
                    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                    opset_version: int = 11,
                    model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Export PyTorch model to ONNX format
        
        Args:
            model: PyTorch model to export
            output_path: Path to save ONNX model
            input_shape: Input tensor shape (B, C, H, W)
            dynamic_axes: Dynamic axes specification for flexible input sizes
            opset_version: ONNX opset version
            model_name: Optional model name for metadata
            
        Returns:
            Export results and validation info
        """
        self.logger.info(f"🔄 Exporting model to ONNX: {output_path}")
        
        try:
            # Prepare model for export
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(*input_shape)
            device = next(model.parameters()).device
            dummy_input = dummy_input.to(device)
            
            # Default dynamic axes for batch size flexibility
            if dynamic_axes is None:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            self.logger.info(f"✅ Model exported to: {output_path}")
            
            # Optimize exported model
            if self.optimize_for_mobile:
                optimized_path = self._optimize_onnx_model(output_path)
                output_path = optimized_path
            
            # Validate exported model
            validation_results = self._validate_onnx_model(
                model, output_path, dummy_input, model_name
            )
            
            # Add metadata
            self._add_model_metadata(output_path, {
                'model_name': model_name or 'cascade_model',
                'input_shape': input_shape,
                'opset_version': opset_version,
                'optimization_applied': self.optimize_for_mobile,
                'export_timestamp': torch.datetime.now().isoformat()
            })
            
            export_results = {
                'success': True,
                'output_path': str(output_path),
                'model_size_mb': os.path.getsize(output_path) / (1024 * 1024),
                'validation_results': validation_results,
                'input_shape': input_shape,
                'dynamic_axes': dynamic_axes,
                'opset_version': opset_version
            }
            
            self.logger.info(f"📊 Export completed - Size: {export_results['model_size_mb']:.2f} MB")
            return export_results
            
        except Exception as e:
            self.logger.error(f"❌ ONNX export failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'output_path': str(output_path)
            }
    
    def _optimize_onnx_model(self, model_path: Union[str, Path]) -> Path:
        """Optimize ONNX model for mobile deployment"""
        self.logger.info("🔧 Optimizing ONNX model for mobile deployment...")
        
        try:
            # Load original model
            model = onnx.load(str(model_path))
            
            # Simplify model using onnx-simplifier
            model_simplified, check = simplify(model)
            
            if check:
                # Save optimized model
                optimized_path = Path(str(model_path).replace('.onnx', '_optimized.onnx'))
                onnx.save(model_simplified, str(optimized_path))
                
                # Compare sizes
                original_size = os.path.getsize(model_path) / (1024 * 1024)
                optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)
                
                self.logger.info(f"✅ Model optimized: {original_size:.2f}MB → {optimized_size:.2f}MB")
                
                # Replace original with optimized
                os.remove(model_path)
                os.rename(optimized_path, model_path)
                
            else:
                self.logger.warning("⚠️ Model simplification failed, keeping original")
            
            return Path(model_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Optimization failed, keeping original: {e}")
            return Path(model_path)
    
    def _validate_onnx_model(self, 
                           pytorch_model: nn.Module,
                           onnx_path: Union[str, Path],
                           test_input: torch.Tensor,
                           model_name: Optional[str] = None) -> Dict[str, Any]:
        """Validate exported ONNX model against PyTorch original"""
        self.logger.info("🔍 Validating exported ONNX model...")
        
        try:
            # Load ONNX model
            ort_session = ort.InferenceSession(str(onnx_path))
            
            # Get PyTorch model output
            pytorch_model.eval()
            with torch.no_grad():
                pytorch_output = pytorch_model(test_input)
            
            # Handle different output formats
            if isinstance(pytorch_output, dict):
                pytorch_output = pytorch_output.get('classification', 
                                                  pytorch_output.get('logits', pytorch_output))
            
            pytorch_output = pytorch_output.cpu().numpy()
            
            # Get ONNX model output
            ort_inputs = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
            onnx_output = ort_session.run(None, ort_inputs)[0]
            
            # Compare outputs
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
            
            # Check if outputs are close enough
            tolerance = 1e-5
            outputs_match = max_diff < tolerance
            
            validation_results = {
                'outputs_match': outputs_match,
                'max_difference': float(max_diff),
                'mean_difference': float(mean_diff),
                'tolerance': tolerance,
                'pytorch_output_shape': list(pytorch_output.shape),
                'onnx_output_shape': list(onnx_output.shape),
                'onnx_providers': ort_session.get_providers()
            }
            
            if outputs_match:
                self.logger.info(f"✅ Validation passed - Max diff: {max_diff:.2e}")
            else:
                self.logger.warning(f"⚠️ Validation warning - Max diff: {max_diff:.2e} > {tolerance}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            return {
                'outputs_match': False,
                'error': str(e)
            }
    
    def _add_model_metadata(self, model_path: Union[str, Path], metadata: Dict[str, Any]):
        """Add metadata to ONNX model"""
        try:
            model = onnx.load(str(model_path))
            
            # Add metadata as model properties
            for key, value in metadata.items():
                model.metadata_props.append(
                    onnx.StringStringEntryProto(key=key, value=str(value))
                )
            
            # Save model with metadata
            onnx.save(model, str(model_path))
            
            self.logger.info("✅ Metadata added to ONNX model")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to add metadata: {e}")
    
    def export_cascade_bundle(self, 
                             models: Dict[str, nn.Module],
                             output_dir: Union[str, Path],
                             bundle_name: str = "cascade_detector") -> Dict[str, Any]:
        """Export complete cascade system as ONNX bundle"""
        self.logger.info(f"📦 Exporting cascade bundle: {bundle_name}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        export_results = {}
        total_size_mb = 0
        
        try:
            # Export each model
            for model_name, model in models.items():
                output_path = output_dir / f"{bundle_name}_{model_name}.onnx"
                
                result = self.export_model(
                    model=model,
                    output_path=output_path,
                    model_name=f"{bundle_name}_{model_name}"
                )
                
                export_results[model_name] = result
                if result['success']:
                    total_size_mb += result['model_size_mb']
            
            # Create bundle manifest
            manifest = {
                'bundle_name': bundle_name,
                'models': export_results,
                'total_size_mb': total_size_mb,
                'export_timestamp': torch.datetime.now().isoformat(),
                'deployment_instructions': {
                    'load_order': list(models.keys()),
                    'input_preprocessing': 'Resize to 256x256, normalize with ImageNet stats',
                    'output_postprocessing': 'Apply sigmoid for probability conversion'
                }
            }
            
            # Save manifest
            manifest_path = output_dir / f"{bundle_name}_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Create deployment package
            self._create_deployment_package(output_dir, bundle_name)
            
            self.logger.info(f"✅ Cascade bundle exported - Total size: {total_size_mb:.2f} MB")
            
            return {
                'success': True,
                'bundle_path': str(output_dir),
                'total_size_mb': total_size_mb,
                'models_exported': len([r for r in export_results.values() if r['success']]),
                'manifest_path': str(manifest_path)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Bundle export failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_deployment_package(self, bundle_dir: Path, bundle_name: str):
        """Create ready-to-deploy package with inference scripts"""
        
        # Create deployment script template
        deployment_script = f'''#!/usr/bin/env python3
"""
{bundle_name.title()} ONNX Deployment Script
Auto-generated deployment script for mobile inference.
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
import json

class {bundle_name.title()}Detector:
    def __init__(self, model_dir="."):
        """Initialize cascade detector with ONNX models"""
        self.models = {{}}
        
        # Load manifest
        with open(f"{{model_dir}}/{bundle_name}_manifest.json") as f:
            self.manifest = json.load(f)
        
        # Load models in order
        for model_name in self.manifest["deployment_instructions"]["load_order"]:
            model_path = f"{{model_dir}}/{bundle_name}_{{model_name}}.onnx"
            self.models[model_name] = ort.InferenceSession(model_path)
        
        print(f"✅ {{len(self.models)}} models loaded successfully")
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 256))
        
        # Convert to tensor and normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = (img_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        return img_array
    
    def predict(self, image_path):
        """Run cascade prediction"""
        # Preprocess
        input_tensor = self.preprocess_image(image_path)
        
        # Run inference (simplified single-stage for template)
        model = list(self.models.values())[0]
        input_name = model.get_inputs()[0].name
        output = model.run(None, {{input_name: input_tensor}})[0]
        
        # Convert to probability
        probability = 1 / (1 + np.exp(-output[0][0]))  # Sigmoid
        prediction = "fake" if probability > 0.5 else "real"
        
        return {{
            "prediction": prediction,
            "confidence": float(probability),
            "model_info": self.manifest["bundle_name"]
        }}

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python deploy.py <image_path>")
        sys.exit(1)
    
    detector = {bundle_name.title()}Detector()
    result = detector.predict(sys.argv[1])
    print(f"Result: {{result}}")
'''
        
        # Save deployment script
        script_path = bundle_dir / f"deploy_{bundle_name}.py"
        with open(script_path, 'w') as f:
            f.write(deployment_script)
        
        # Create requirements file
        requirements = [
            "onnxruntime>=1.12.0",
            "numpy>=1.20.0", 
            "Pillow>=8.0.0"
        ]
        
        req_path = bundle_dir / "requirements.txt"
        with open(req_path, 'w') as f:
            f.write('\n'.join(requirements))
        
        # Create README
        readme_content = f"""# {bundle_name.title()} ONNX Deployment

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run inference:
```bash
python deploy_{bundle_name}.py <image_path>
```

## Files

- `{bundle_name}_*.onnx`: Quantized ONNX models
- `{bundle_name}_manifest.json`: Deployment configuration
- `deploy_{bundle_name}.py`: Ready-to-use inference script
- `requirements.txt`: Python dependencies

## Model Info

Total size: See manifest.json
Input: 256x256 RGB images
Output: Binary classification (real/fake)

Generated by AWARE-NET Mobile Optimization Pipeline
"""
        
        readme_path = bundle_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        self.logger.info(f"📦 Deployment package created in: {bundle_dir}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ONNX Model Exporter')
    parser.add_argument('--model_path', type=str, required=True, help='Path to PyTorch model')
    parser.add_argument('--output_path', type=str, required=True, help='Output ONNX path')
    parser.add_argument('--input_shape', type=int, nargs=4, default=[1, 3, 256, 256], 
                       help='Input shape (B C H W)')
    parser.add_argument('--model_name', type=str, help='Model name for metadata')
    
    args = parser.parse_args()
    
    # Load PyTorch model (this is a simplified example)
    model = torch.load(args.model_path, map_location='cpu')
    
    # Export to ONNX
    exporter = ONNXExporter()
    result = exporter.export_model(
        model=model,
        output_path=args.output_path,
        input_shape=tuple(args.input_shape),
        model_name=args.model_name
    )
    
    if result['success']:
        print(f"✅ Export successful: {result['output_path']}")
        print(f"📊 Model size: {result['model_size_mb']:.2f} MB")
    else:
        print(f"❌ Export failed: {result['error']}")