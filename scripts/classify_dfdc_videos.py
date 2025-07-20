#!/usr/bin/env python3
"""
DFDC Video Classification Script

This script classifies DFDC videos into real/fake categories based on metadata.json files
and organizes them into appropriate folder structures.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_metadata(metadata_path: str) -> Dict:
    """Load metadata.json file."""
    with open(metadata_path, 'r') as f:
        return json.load(f)


def create_output_directories(output_base: str) -> Tuple[str, str]:
    """Create output directories for real and fake videos."""
    real_dir = os.path.join(output_base, 'real')
    fake_dir = os.path.join(output_base, 'fake')
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    return real_dir, fake_dir


def get_all_metadata_files(base_dir: str) -> List[Tuple[str, str]]:
    """Find all metadata.json files in subdirectories."""
    metadata_files = []
    base_path = Path(base_dir)
    
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            metadata_file = subdir / 'metadata.json'
            if metadata_file.exists():
                metadata_files.append((str(subdir), str(metadata_file)))
    
    return metadata_files


def classify_videos(dfdc_dir: str, output_dir: str, copy_files: bool = True, 
                   max_videos_per_class: int = None) -> Dict[str, int]:
    """
    Classify DFDC videos based on metadata labels.
    
    Args:
        dfdc_dir: Path to DFDC dataset directory
        output_dir: Path to output directory for classified videos
        copy_files: Whether to copy files (True) or create symlinks (False)
        max_videos_per_class: Maximum number of videos per class (None for all)
    
    Returns:
        Dictionary with classification statistics
    """
    # Create output directories
    real_dir, fake_dir = create_output_directories(output_dir)
    
    # Find all metadata files
    metadata_files = get_all_metadata_files(dfdc_dir)
    print(f"Found {len(metadata_files)} metadata files")
    
    stats = {'real': 0, 'fake': 0, 'errors': 0}
    real_count = 0
    fake_count = 0
    
    # Process each metadata file
    for i, (folder_path, metadata_file) in enumerate(metadata_files):
        if i % 10 == 0:
            print(f"Processing folder {i+1}/{len(metadata_files)}: {os.path.basename(folder_path)}")
        try:
            metadata = load_metadata(metadata_file)
            folder_name = os.path.basename(folder_path)
            
            # Process each video in the metadata
            for video_name, video_info in metadata.items():
                label = video_info.get('label', '').upper()
                
                # Skip if max count reached
                if max_videos_per_class:
                    if label == 'REAL' and real_count >= max_videos_per_class:
                        continue
                    if label == 'FAKE' and fake_count >= max_videos_per_class:
                        continue
                
                # Source video path
                src_path = os.path.join(folder_path, video_name)
                
                if not os.path.exists(src_path):
                    print(f"Warning: Video not found: {src_path}")
                    stats['errors'] += 1
                    continue
                
                # Determine destination based on label
                if label == 'REAL':
                    dst_path = os.path.join(real_dir, f"{folder_name}_{video_name}")
                    real_count += 1
                    stats['real'] += 1
                elif label == 'FAKE':
                    dst_path = os.path.join(fake_dir, f"{folder_name}_{video_name}")
                    fake_count += 1
                    stats['fake'] += 1
                else:
                    print(f"Warning: Unknown label '{label}' for {video_name}")
                    stats['errors'] += 1
                    continue
                
                # Copy or link the file
                try:
                    if copy_files:
                        shutil.copy2(src_path, dst_path)
                    else:
                        # Create symlink (relative path)
                        rel_src = os.path.relpath(src_path, os.path.dirname(dst_path))
                        os.symlink(rel_src, dst_path)
                except Exception as e:
                    print(f"Error processing {src_path}: {e}")
                    stats['errors'] += 1
                    
                    # Adjust counts if file operation failed
                    if label == 'REAL':
                        stats['real'] -= 1
                        real_count -= 1
                    elif label == 'FAKE':
                        stats['fake'] -= 1
                        fake_count -= 1
        
        except Exception as e:
            print(f"Error processing {metadata_file}: {e}")
            stats['errors'] += 1
    
    return stats


def print_statistics(stats: Dict[str, int], output_dir: str):
    """Print classification statistics."""
    print("\n" + "="*50)
    print("DFDC Video Classification Complete!")
    print("="*50)
    print(f"Real videos: {stats['real']}")
    print(f"Fake videos: {stats['fake']}")
    print(f"Total processed: {stats['real'] + stats['fake']}")
    print(f"Errors: {stats['errors']}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Real videos folder: {os.path.join(output_dir, 'real')}")
    print(f"Fake videos folder: {os.path.join(output_dir, 'fake')}")


def analyze_metadata_only(dfdc_dir: str):
    """Analyze metadata without copying files."""
    metadata_files = get_all_metadata_files(dfdc_dir)
    print(f"Found {len(metadata_files)} metadata files")
    
    stats = {'real': 0, 'fake': 0, 'total_files': 0, 'folders': len(metadata_files)}
    
    for i, (folder_path, metadata_file) in enumerate(metadata_files):
        if i % 10 == 0:
            print(f"Analyzing folder {i+1}/{len(metadata_files)}: {os.path.basename(folder_path)}")
        try:
            metadata = load_metadata(metadata_file)
            stats['total_files'] += len(metadata)
            
            for video_name, video_info in metadata.items():
                label = video_info.get('label', '').upper()
                if label == 'REAL':
                    stats['real'] += 1
                elif label == 'FAKE':
                    stats['fake'] += 1
        except Exception as e:
            print(f"Error reading {metadata_file}: {e}")
    
    print("\n" + "="*50)
    print("DFDC Dataset Analysis")
    print("="*50)
    print(f"Total folders: {stats['folders']}")
    print(f"Total videos: {stats['total_files']}")
    print(f"Real videos: {stats['real']}")
    print(f"Fake videos: {stats['fake']}")
    print(f"Real/Fake ratio: {stats['real'] / stats['fake']:.2f}" if stats['fake'] > 0 else "N/A")


def main():
    parser = argparse.ArgumentParser(description='Classify DFDC videos based on metadata')
    parser.add_argument('dfdc_dir', help='Path to DFDC dataset directory')
    parser.add_argument('--output', '-o', help='Output directory for classified videos')
    parser.add_argument('--analyze-only', action='store_true', 
                       help='Only analyze metadata without copying files')
    parser.add_argument('--symlink', action='store_true', 
                       help='Create symlinks instead of copying files')
    parser.add_argument('--max-per-class', type=int, 
                       help='Maximum number of videos per class')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dfdc_dir):
        print(f"Error: DFDC directory not found: {args.dfdc_dir}")
        return
    
    if args.analyze_only:
        analyze_metadata_only(args.dfdc_dir)
    else:
        if not args.output:
            args.output = os.path.join(args.dfdc_dir, 'classified')
        
        print(f"Classifying videos from: {args.dfdc_dir}")
        print(f"Output directory: {args.output}")
        print(f"Copy method: {'Symlinks' if args.symlink else 'Copy files'}")
        if args.max_per_class:
            print(f"Max videos per class: {args.max_per_class}")
        
        stats = classify_videos(
            args.dfdc_dir, 
            args.output, 
            copy_files=not args.symlink,
            max_videos_per_class=args.max_per_class
        )
        
        print_statistics(stats, args.output)


if __name__ == "__main__":
    main()