#!/usr/bin/env python3
"""
Deep exploration of BOBSL dataset structure.
Saves output to a text file for easy sharing.
"""

import json
from pathlib import Path
from pprint import pformat
from collections import defaultdict
from datetime import datetime


class OutputWriter:
    """Write output to both file and console."""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.file = open(self.filepath, 'w', encoding='utf-8')
    
    def write(self, text: str = "") -> None:
        print(text)
        self.file.write(text + "\n")
    
    def close(self) -> None:
        self.file.close()


def explore_bobsl_dataset(base_dir: str, output_file: str) -> None:
    """Explore the complete BOBSL dataset structure."""
    base_dir = Path(base_dir)
    out = OutputWriter(output_file)
    
    out.write("=" * 70)
    out.write("BOBSL DATASET STRUCTURE REPORT")
    out.write("=" * 70)
    out.write(f"Base directory: {base_dir}")
    out.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find all files by extension
    files_by_ext = defaultdict(list)
    
    for filepath in base_dir.rglob('*'):
        if filepath.is_file():
            ext = filepath.suffix.lower()
            files_by_ext[ext].append(filepath)
    
    # Summary
    out.write("\n--- FILE SUMMARY ---")
    for ext, files in sorted(files_by_ext.items()):
        out.write(f"  {ext or '(no ext)'}: {len(files)} files")
    
    # Explore JSON files (annotations)
    json_files = files_by_ext.get('.json', [])
    if json_files:
        out.write("\n" + "=" * 70)
        out.write(f"JSON FILES ({len(json_files)} total)")
        out.write("=" * 70)
        
        # Group by parent directory
        json_by_dir = defaultdict(list)
        for f in json_files:
            json_by_dir[f.parent].append(f)
        
        out.write("\nLocations:")
        for dir_path, files in json_by_dir.items():
            rel_path = dir_path.relative_to(base_dir)
            out.write(f"  {rel_path}/  ({len(files)} files)")
        
        # Explore first JSON file from each directory
        for dir_path, files in list(json_by_dir.items())[:5]:
            sample_file = files[0]
            out.write(f"\n--- SAMPLE: {sample_file.relative_to(base_dir)} ---")
            out.write(f"Size: {sample_file.stat().st_size / 1024:.1f} KB")
            
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    keys = list(data.keys())
                    out.write(f"Type: dict with {len(keys)} keys")
                    out.write(f"Keys (first 10): {keys[:10]}")
                    
                    # Show first entry
                    first_key = keys[0]
                    first_value = data[first_key]
                    out.write(f"\nFirst entry key: '{first_key}'")
                    out.write(f"First entry value:")
                    out.write(pformat(first_value, width=100, depth=4))
                    
                    # Show second entry if different structure
                    if len(keys) > 1:
                        second_key = keys[1]
                        second_value = data[second_key]
                        out.write(f"\nSecond entry key: '{second_key}'")
                        out.write(f"Second entry value:")
                        out.write(pformat(second_value, width=100, depth=4))
                    
                elif isinstance(data, list):
                    out.write(f"Type: list with {len(data)} items")
                    if data:
                        out.write(f"First item:")
                        out.write(pformat(data[0], width=100, depth=4))
                        if len(data) > 1:
                            out.write(f"\nSecond item:")
                            out.write(pformat(data[1], width=100, depth=4))
                else:
                    out.write(f"Type: {type(data).__name__}")
                    
            except Exception as e:
                out.write(f"Error reading: {e}")
    
    # Explore NPY/NPZ files (features)
    npy_files = files_by_ext.get('.npy', []) + files_by_ext.get('.npz', [])
    if npy_files:
        out.write("\n" + "=" * 70)
        out.write(f"FEATURE FILES ({len(npy_files)} total)")
        out.write("=" * 70)
        
        # Group by parent directory
        npy_by_dir = defaultdict(list)
        for f in npy_files:
            npy_by_dir[f.parent].append(f)
        
        out.write("\nLocations:")
        for dir_path, files in npy_by_dir.items():
            rel_path = dir_path.relative_to(base_dir)
            out.write(f"  {rel_path}/  ({len(files)} files)")
            
            # Show sample filenames
            out.write(f"    Sample files: {[f.name for f in files[:5]]}")
        
        # Sample feature files
        try:
            import numpy as np
            out.write("\nSample feature analysis:")
            for sample_file in npy_files[:3]:
                out.write(f"\n  File: {sample_file.name}")
                data = np.load(sample_file, allow_pickle=True)
                if isinstance(data, np.lib.npyio.NpzFile):
                    out.write(f"    Format: NPZ")
                    out.write(f"    Arrays: {list(data.keys())}")
                    for key in list(data.keys())[:3]:
                        out.write(f"      {key}: shape={data[key].shape}, dtype={data[key].dtype}")
                else:
                    out.write(f"    Format: NPY")
                    out.write(f"    Shape: {data.shape}")
                    out.write(f"    Dtype: {data.dtype}")
        except ImportError:
            out.write("\n  numpy not available for feature inspection")
        except Exception as e:
            out.write(f"\n  Error loading features: {e}")
    
    # Explore VTT files (subtitles)
    vtt_files = files_by_ext.get('.vtt', [])
    if vtt_files:
        out.write("\n" + "=" * 70)
        out.write(f"SUBTITLE FILES ({len(vtt_files)} total)")
        out.write("=" * 70)
        
        # Group by parent directory
        vtt_by_dir = defaultdict(list)
        for f in vtt_files:
            vtt_by_dir[f.parent].append(f)
        
        out.write("\nLocations:")
        for dir_path, files in vtt_by_dir.items():
            rel_path = dir_path.relative_to(base_dir)
            out.write(f"  {rel_path}/  ({len(files)} files)")
        
        # Sample VTT content
        sample_file = vtt_files[0]
        out.write(f"\nSample subtitle: {sample_file.name}")
        try:
            with open(sample_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:20]
            out.write("Content (first 20 lines):")
            for line in lines:
                out.write(f"  {line.rstrip()}")
        except Exception as e:
            out.write(f"  Error: {e}")
    
    # Show full directory tree
    out.write("\n" + "=" * 70)
    out.write("FULL DIRECTORY TREE")
    out.write("=" * 70)
    
    show_tree(base_dir, base_dir, out, max_depth=6)
    
    out.write("\n" + "=" * 70)
    out.write("EXPLORATION COMPLETE")
    out.write("=" * 70)
    out.write(f"\nOutput saved to: {output_file}")
    
    out.close()


def show_tree(path: Path, base: Path, out: OutputWriter, prefix: str = "", 
              max_depth: int = 4, current_depth: int = 0) -> None:
    """Display directory tree."""
    if current_depth >= max_depth:
        return
    
    try:
        items = sorted(path.iterdir())
    except PermissionError:
        return
    
    dirs = [i for i in items if i.is_dir()]
    files = [i for i in items if i.is_file()]
    
    for i, d in enumerate(dirs):
        is_last = (i == len(dirs) - 1) and not files
        connector = "`-- " if is_last else "|-- "
        out.write(f"{prefix}{connector}{d.name}/")
        extension = "    " if is_last else "|   "
        show_tree(d, base, out, prefix + extension, max_depth, current_depth + 1)
    
    for i, f in enumerate(files[:10]):
        is_last = i == min(len(files), 10) - 1
        connector = "`-- " if is_last else "|-- "
        size_kb = f.stat().st_size / 1024
        out.write(f"{prefix}{connector}{f.name} ({size_kb:.1f} KB)")
    
    if len(files) > 10:
        out.write(f"{prefix}    ... and {len(files) - 10} more files")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Explore BOBSL dataset structure')
    parser.add_argument('--dir', type=str, 
                        default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output', type=str,
                        default='bobsl_structure_report.txt',
                        help='Output file path')
    
    args = parser.parse_args()
    explore_bobsl_dataset(args.dir, args.output)
