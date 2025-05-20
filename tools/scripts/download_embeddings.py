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
        530000000,  # Approximate file size in bytes (530MB)
        "https://drive.usercontent.google.com/download?id=17Mwnr2_IX2xx0C3VMn8mS2RDjNMkuabI&export=download",
        "https://drive.google.com/file/d/17Mwnr2_IX2xx0C3VMn8mS2RDjNMkuabI/view"
    ),
    2: (
        555000000,  # Approximate file size in bytes (555MB)
        "https://drive.usercontent.google.com/download?id=1gFWgiSu-csqVfuTn4Clfjxctd0mna9Wp&export=download&confirm=t&uuid=2a40be5a-446e-44dc-9ad7-ce170fdfc3e1",
        "https://drive.google.com/file/d/1gFWgiSu-csqVfuTn4Clfjxctd0mna9Wp/view"
    ),
    3: (
        771000000,  # Approximate file size in bytes (771MB)
        "https://drive.usercontent.google.com/download?id=1R1ijVWoCtoclUWP_gevxsTiGoT8Cp4h1&export=download&confirm=t&uuid=16d23370-b34e-48ce-815e-1e38f528e3fa",
        "https://drive.google.com/file/d/1R1ijVWoCtoclUWP_gevxsTiGoT8Cp4h1/view"
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

def get_gdrive_confirmation_id(file_id):
    """
    Get the confirmation ID needed to bypass Google Drive's virus scan warning.
    
    Args:
        file_id (str): Google Drive file ID
        
    Returns:
        str: URL with confirmation parameters
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        # Create a session to handle cookies
        opener = urllib.request.build_opener()
        response = opener.open(url)
        
        # Sometimes small files don't need confirmation
        if response.geturl() != url:
            # Check if this is a download link
            if 'download' in response.geturl():
                return response.geturl()
        
        # Get the HTML response
        try:
            html = response.read().decode('utf-8')
        except UnicodeDecodeError:
            # If we can't decode, we're likely already getting the file
            return url
            
        # Extract the confirmation code
        confirm_match = re.search(r'confirm=([0-9A-Za-z]+)', html)
        if confirm_match:
            confirm_code = confirm_match.group(1)
            return f"https://drive.google.com/uc?export=download&confirm={confirm_code}&id={file_id}"
        
        # Look for the form action in case the pattern changes
        form_match = re.search(r'action="([^"]*)"', html)
        if form_match:
            form_action = form_match.group(1)
            if 'download' in form_action:
                return f"https://drive.google.com{form_action}"
                
        # If no confirmation needed or pattern changed, use the original URL
        return url
    except Exception as e:
        print(f"Warning: Could not get confirmation ID: {e}")
        return url

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
    
    # Get file ID from the Google Drive URL
    file_id_match = re.search(r'id=([^&]+)', direct_url)
    if not file_id_match:
        print("Could not extract file ID from URL. Using direct URL as fallback.")
        download_url = direct_url
    else:
        file_id = file_id_match.group(1)
        print(f"File ID: {file_id}")
        
        # Get confirmed download URL
        print("Getting download URL with confirmation...")
        download_url = get_gdrive_confirmation_id(file_id)
    
    # Create a request with a custom User-Agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        print(f"Starting download with confirmed URL...")
        start_time = time.time()
        
        # Create a request with headers
        req = urllib.request.Request(download_url, headers=headers)
        
        # Open with the request
        with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
            # Get content length for progress if available
            content_length = response.getheader('Content-Length')
            if content_length:
                total_size = int(content_length)
            else:
                total_size = -1  # Unknown size
            
            # Download in chunks with progress
            downloaded = 0
            block_size = 8192  # 8KB chunks
            
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                    
                out_file.write(buffer)
                downloaded += len(buffer)
                
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    sys.stdout.write(f"\r[{'#' * int(percent // 2)}{'.' * (50 - int(percent // 2))}] {percent:.1f}% ")
                else:
                    sys.stdout.write(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB")
                sys.stdout.flush()
        
        elapsed = time.time() - start_time
        print(f"\nDownload completed in {elapsed:.1f} seconds!")
        
        # Validate downloaded file
        if validate_file(file_path, file_size):
            print(f"Validation successful! File saved to {file_path}")
            # Check if it's a valid JSON
            is_json = validate_embeddings_json(file_path)
            if not is_json:
                print(f"Downloaded file is not a valid JSON. It might be an HTML error page.")
                return False
            return True
        else:
            print(f"Warning: Downloaded file has unexpected size. It may be incomplete.")
            # Check if it's HTML (most likely an error page)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content_start = f.read(1000).strip()
                    if content_start.startswith('<!DOCTYPE html>') or content_start.startswith('<html'):
                        print("Error: Downloaded file appears to be HTML, not JSON data.")
                        print("This typically happens with Google Drive's virus warning page.")
                        print(f"Please manually download from: {gdrive_url}")
                        print(f"And save to: {file_path}")
                        return False
            except UnicodeDecodeError:
                # Binary data, possibly incomplete file
                pass
            return False
    
    except urllib.error.URLError as e:
        print(f"\nError downloading file: {e}")
        print("\nPlease manually download the file:")
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