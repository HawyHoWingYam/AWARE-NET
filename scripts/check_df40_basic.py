#!/usr/bin/env python3
"""
Basic DF40 dataset analysis
"""

import os

def analyze_df40_basic():
    """Basic analysis of DF40 dataset"""
    base_path = "D:/work/AWARE-NET/dataset/DF40"
    
    print("=== DF40 Dataset Analysis ===")
    
    # Check root frames folder
    frames_dir = os.path.join(base_path, "frames")
    
    if os.path.exists(frames_dir):
        print(f"Root frames folder exists: {frames_dir}")
        
        subfolders = [f for f in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, f))]
        print(f"Found {len(subfolders)} subfolders")
        
        # Check first subfolder
        if subfolders:
            first_subfolder = subfolders[0]
            subfolder_path = os.path.join(frames_dir, first_subfolder)
            images = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
            print(f"Sample folder {first_subfolder}: {len(images)} PNG images")
            
            if images:
                # Get file size of first image
                sample_path = os.path.join(subfolder_path, images[0])
                file_size = os.path.getsize(sample_path)
                print(f"Sample image: {images[0]}, Size: {file_size/1024:.1f} KB")
                
                # Try to get image dimensions
                try:
                    from PIL import Image
                    with Image.open(sample_path) as img:
                        width, height = img.size
                        mode = img.mode
                        print(f"Image dimensions: {width} x {height}")
                        print(f"Color mode: {mode}")
                        
                        # Check a few more images for consistency
                        print("Checking consistency...")
                        for i, img_file in enumerate(images[1:6]):
                            img_path = os.path.join(subfolder_path, img_file)
                            with Image.open(img_path) as img2:
                                if img2.size != (width, height):
                                    print(f"WARNING: Different size found: {img_file} = {img2.size}")
                                else:
                                    print(f"OK: {img_file} = {img2.size}")
                        
                        print(f"\n=== RECOMMENDED PREPROCESSING CONFIG ===")
                        print(f"image_size: [{width}, {height}]")
                        print(f"Format: PNG")
                        print(f"Color mode: {mode}")
                        
                        return {
                            'width': width,
                            'height': height,
                            'format': 'PNG',
                            'mode': mode
                        }
                        
                except ImportError:
                    print("PIL not available - cannot determine image dimensions")
                except Exception as e:
                    print(f"Error reading image: {e}")
    
    # Check method folders
    method_folders = ['blendface', 'e4s', 'facedancer', 'faceswap', 'fsgan', 'inswap', 'mobileswap', 'simswap']
    
    for method in method_folders:
        method_path = os.path.join(base_path, method)
        if os.path.exists(method_path):
            frames_path = os.path.join(method_path, "frames")
            landmarks_path = os.path.join(method_path, "landmarks")
            
            has_frames = os.path.exists(frames_path)
            has_landmarks = os.path.exists(landmarks_path)
            
            print(f"{method}: frames={has_frames}, landmarks={has_landmarks}")
            
            if has_frames:
                try:
                    subfolders = [f for f in os.listdir(frames_path) if os.path.isdir(os.path.join(frames_path, f))]
                    print(f"  {len(subfolders)} subfolders")
                except Exception as e:
                    print(f"  Error reading {method}: {e}")
    
    return None

if __name__ == "__main__":
    analyze_df40_basic()