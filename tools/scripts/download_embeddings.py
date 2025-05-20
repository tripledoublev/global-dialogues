#!/usr/bin/env python
"""
Script to download embedding files for Global Dialogues analysis.
This allows users to selectively download only the embeddings they need.
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error
from urllib.parse import urlparse
import re
import hashlib
import time

# File URLs - Update these when hosting changes
# Format: GD number -> (file_size_bytes, direct_download_url, gdrive_url)
EMBEDDING_FILES = {
    1: (
        800000000,  # Approximate file size in bytes (800MB)
        "https://drive.usercontent.google.com/download?id=YOUR_GD1_ID_HERE&export=download",
        "https://drive.google.com/file/d/YOUR_GD1_ID_HERE/view"
    ),
    2: (
        800000000,  # Approximate file size in bytes (800MB)
        "https://drive.usercontent.google.com/download?id=YOUR_GD2_ID_HERE&export=download",
        "https://drive.google.com/file/d/YOUR_GD2_ID_HERE/view"
    ),
    3: (
        800000000,  # Approximate file size in bytes (800MB)
        "https://drive.usercontent.google.com/download?id=17Mwnr2_IX2xx0C3VMn8mS2RDjNMkuabI&export=download",
        "https://drive.google.com/file/d/17Mwnr2_IX2xx0C3VMn8mS2RDjNMkuabI/view"
    )
}

# File paths to save embeddings to
def get_embedding_path(gd_number):
    """Generate file path for a specific GD embedding file."""
    return os.path.join("Data", f"GD{gd_number}", f"GD{gd_number}_embeddings.json")

# Progress bar for downloads
def show_progress(block_num, block_size, total_size):
    """Display download progress."""
    if total_size > 0:
        percent = min(100, block_num * block_size * 100 / total_size)
        sys.stdout.write(f"\r[{'#' * int(percent // 2)}{'.' * (50 - int(percent // 2))}] {percent:.1f}% ")
        sys.stdout.flush()

def validate_file(file_path, expected_size):
    """
    Basic validation that the file exists and is approximately the expected size.
    Returns True if valid, False otherwise.
    """
    if not os.path.exists(file_path):
        return False
    
    actual_size = os.path.getsize(file_path)
    # Allow 10% size difference to account for approximation
    size_min = expected_size * 0.9
    size_max = expected_size * 1.1
    
    return size_min <= actual_size <= size_max

def download_embedding(gd_number, force=False):
    """
    Download embeddings file for the specified Global Dialogue number.
    
    Args:
        gd_number (int): Global Dialogue number (1, 2, or 3)
        force (bool): Force download even if file exists
        
    Returns:
        bool: True if download successful, False otherwise
    """
    if gd_number not in EMBEDDING_FILES:
        print(f"Error: No embedding file defined for GD{gd_number}")
        return False
    
    file_size, direct_url, gdrive_url = EMBEDDING_FILES[gd_number]
    file_path = get_embedding_path(gd_number)
    
    # Check if directory exists
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(file_path) and not force:
        if validate_file(file_path, file_size):
            print(f"Embedding file for GD{gd_number} already exists at {file_path}")
            return True
        else:
            print(f"Warning: Existing file {file_path} has unexpected size. Use --force to replace it.")
            return False
    
    # Download the file
    print(f"Downloading embedding file for GD{gd_number}...")
    print(f"File will be saved to: {file_path}")
    print(f"Expected file size: ~{file_size / 1024 / 1024:.1f} MB")
    
    try:
        print(f"Starting download from: {direct_url[:60]}...")
        start_time = time.time()
        
        urllib.request.urlretrieve(direct_url, file_path, show_progress)
        
        elapsed = time.time() - start_time
        print(f"\nDownload completed in {elapsed:.1f} seconds!")
        
        # Validate downloaded file
        if validate_file(file_path, file_size):
            print(f"Validation successful! File saved to {file_path}")
            return True
        else:
            print(f"Warning: Downloaded file has unexpected size. It may be incomplete.")
            return False
    
    except urllib.error.URLError as e:
        print(f"\nError downloading file: {e}")
        print("\nIf the direct download link has expired, please:")
        print(f"1. Visit the Google Drive link: {gdrive_url}")
        print(f"2. Click 'Download' and save the file manually to: {file_path}")
        return False

def validate_embeddings_json(file_path, verbose=False):
    """
    Validate that the downloaded JSON file has the expected format.
    Returns True if valid, False otherwise.
    """
    print(f"Validating JSON format of {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("Error: File does not contain a JSON list.")
            return False
        
        if len(data) == 0:
            print("Error: JSON list is empty.")
            return False
        
        if verbose:
            print(f"JSON contains {len(data)} items")
            
            # Check first item for expected structure
            first_item = data[0]
            if not isinstance(first_item, dict):
                print("Error: First item is not a dictionary.")
                return False
                
            if 'embedding' not in first_item:
                print("Error: No 'embedding' field found in first item.")
                return False
                
            if not isinstance(first_item['embedding'], list):
                print("Error: 'embedding' is not a list.")
                return False
                
            embedding_dim = len(first_item['embedding'])
            print(f"Embedding dimension: {embedding_dim}")
        
        print(f"JSON validation successful!")
        return True
    
    except json.JSONDecodeError:
        print("Error: The file is not valid JSON.")
        return False
    except Exception as e:
        print(f"Error validating file: {e}")
        return False

def list_available_embeddings():
    """List available embedding files and their status."""
    print("\nAvailable Embedding Files:")
    print("=" * 50)
    print(f"{'GD#':<5} {'Status':<15} {'Size':<15} {'Path':<30}")
    print("-" * 50)
    
    for gd_num in sorted(EMBEDDING_FILES.keys()):
        file_path = get_embedding_path(gd_num)
        expected_size, _, _ = EMBEDDING_FILES[gd_num]
        
        if os.path.exists(file_path):
            actual_size = os.path.getsize(file_path)
            size_str = f"{actual_size / 1024 / 1024:.1f} MB"
            
            if validate_file(file_path, expected_size):
                status = "✓ Downloaded"
            else:
                status = "! Invalid Size"
        else:
            status = "Not Downloaded"
            size_str = f"~{expected_size / 1024 / 1024:.1f} MB"
        
        print(f"GD{gd_num:<4} {status:<15} {size_str:<15} {file_path}")
    
    print("=" * 50)

def main():
    parser = argparse.ArgumentParser(
        description="Download embedding files for Global Dialogues analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download embeddings for GD3
  python download_embeddings.py 3
  
  # Download embeddings for GD1 and GD2
  python download_embeddings.py 1 2
  
  # Download all available embeddings
  python download_embeddings.py --all
  
  # Force re-download even if file exists
  python download_embeddings.py 3 --force
  
  # List available embedding files and their status
  python download_embeddings.py --list
"""
    )
    
    parser.add_argument('gd_numbers', type=int, nargs='*', choices=list(EMBEDDING_FILES.keys()),
                      help='Global Dialogue number(s) to download embeddings for')
    parser.add_argument('--all', action='store_true', 
                      help='Download embeddings for all available GD numbers')
    parser.add_argument('--force', action='store_true',
                      help='Force download even if file already exists')
    parser.add_argument('--list', action='store_true',
                      help='List available embedding files and their status')
    parser.add_argument('--validate', action='store_true',
                      help='Validate JSON format of already downloaded files')
    
    args = parser.parse_args()
    
    # Handle --list flag
    if args.list:
        list_available_embeddings()
        return
    
    # Handle --all flag
    if args.all:
        gd_numbers = list(EMBEDDING_FILES.keys())
    else:
        gd_numbers = args.gd_numbers
    
    # If no GD numbers provided and not using --all, show usage
    if not gd_numbers and not args.validate:
        parser.print_help()
        return
    
    # Handle --validate flag
    if args.validate:
        for gd_num in EMBEDDING_FILES.keys():
            file_path = get_embedding_path(gd_num)
            if os.path.exists(file_path):
                validate_embeddings_json(file_path, verbose=True)
        return
    
    # Download each requested embedding file
    for gd_num in gd_numbers:
        print(f"\nProcessing GD{gd_num} embeddings...")
        success = download_embedding(gd_num, force=args.force)
        
        if success and args.validate:
            validate_embeddings_json(get_embedding_path(gd_num))

if __name__ == "__main__":
    main()