#!/usr/bin/env python3
"""
Read PNG dimensions from file header without PIL
"""

import struct
import os

def get_png_dimensions(file_path):
    """Get PNG image dimensions by reading file header"""
    try:
        with open(file_path, 'rb') as f:
            # PNG signature
            signature = f.read(8)
            if signature != b'\x89PNG\r\n\x1a\n':
                return None, None
            
            # Read IHDR chunk
            chunk_length = struct.unpack('>I', f.read(4))[0]
            chunk_type = f.read(4)
            
            if chunk_type != b'IHDR':
                return None, None
            
            # Read width and height
            width = struct.unpack('>I', f.read(4))[0]
            height = struct.unpack('>I', f.read(4))[0]
            
            return width, height
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def analyze_df40_images():
    """Analyze DF40 image specifications"""
    base_path = "D:/work/AWARE-NET/dataset/DF40"
    
    print("=== DF40 Image Specifications Analysis ===")
    
    # Check root frames folder
    frames_dir = os.path.join(base_path, "frames")
    
    image_specs = []
    
    if os.path.exists(frames_dir):
        subfolders = [f for f in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, f))]
        
        # Check first few subfolders
        for subfolder in subfolders[:5]:
            subfolder_path = os.path.join(frames_dir, subfolder)
            images = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
            
            if images:
                # Check first image in each folder
                img_path = os.path.join(subfolder_path, images[0])
                width, height = get_png_dimensions(img_path)
                
                if width and height:
                    file_size = os.path.getsize(img_path)
                    image_specs.append({
                        'width': width,
                        'height': height,
                        'file_size_kb': file_size / 1024,
                        'path': f"{subfolder}/{images[0]}"
                    })
                    print(f"{subfolder}/{images[0]}: {width}x{height}, {file_size/1024:.1f}KB")
    
    # Check method-specific folders
    method_folders = ['blendface', 'e4s', 'facedancer', 'faceswap']
    
    for method in method_folders[:2]:  # Check first 2 methods
        method_frames_dir = os.path.join(base_path, method, "frames")
        if os.path.exists(method_frames_dir):
            subfolders = [f for f in os.listdir(method_frames_dir) if os.path.isdir(os.path.join(method_frames_dir, f))]
            
            if subfolders:
                # Check first subfolder
                subfolder_path = os.path.join(method_frames_dir, subfolders[0])
                images = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
                
                if images:
                    img_path = os.path.join(subfolder_path, images[0])
                    width, height = get_png_dimensions(img_path)
                    
                    if width and height:
                        file_size = os.path.getsize(img_path)
                        image_specs.append({
                            'width': width,
                            'height': height,
                            'file_size_kb': file_size / 1024,
                            'path': f"{method}/{subfolders[0]}/{images[0]}"
                        })
                        print(f"{method}/{subfolders[0]}/{images[0]}: {width}x{height}, {file_size/1024:.1f}KB")
    
    # Analyze results
    if image_specs:
        print(f"\n=== Analysis Results ===")
        
        widths = [spec['width'] for spec in image_specs]
        heights = [spec['height'] for spec in image_specs]
        file_sizes = [spec['file_size_kb'] for spec in image_specs]
        
        print(f"Samples analyzed: {len(image_specs)}")
        print(f"Width range: {min(widths)} - {max(widths)}")
        print(f"Height range: {min(heights)} - {max(heights)}")
        print(f"File size range: {min(file_sizes):.1f}KB - {max(file_sizes):.1f}KB")
        
        # Check consistency
        unique_widths = set(widths)
        unique_heights = set(heights)
        
        if len(unique_widths) == 1 and len(unique_heights) == 1:
            print(f"CONSISTENT: All images are {widths[0]}x{heights[0]}")
            
            print(f"\n=== RECOMMENDED CONFIG FOR OTHER DATASETS ===")
            print(f'"image_size": [{widths[0]}, {heights[0]}]')
            print(f'"format": "PNG"')
            
            return {
                'width': widths[0],
                'height': heights[0],
                'format': 'PNG'
            }
        else:
            print(f"INCONSISTENT: Multiple sizes found")
            print(f"Unique widths: {unique_widths}")
            print(f"Unique heights: {unique_heights}")
    
    else:
        print("No valid images found or analyzed")
    
    return None

if __name__ == "__main__":
    analyze_df40_images()