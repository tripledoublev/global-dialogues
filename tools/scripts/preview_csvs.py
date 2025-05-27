#!/usr/bin/env python3
import os
import sys
import csv
import argparse

def main():
    parser = argparse.ArgumentParser(description='Preview CSV files in a GD directory.')
    parser.add_argument('--gd_number', type=int, help='Global Dialogue cadence number (e.g., 1, 2, 3). Constructs default path.')
    parser.add_argument('--directory', help='Explicit directory path to preview (overrides --gd_number).')
    args = parser.parse_args()
    
    # Determine directory
    if args.directory:
        directory = args.directory
    elif args.gd_number:
        directory = f"Data/GD{args.gd_number}"
    else:
        parser.error("Either --gd_number or --directory must be specified.")
    
    # Define output file name
    output_filename = "csv_previews.txt"
    
    # Get all CSV files in the directory (not recursively)
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    # Sort files for consistent output
    csv_files.sort()
    
    # Process each CSV file
    output_text = ""
    for file in csv_files:
        file_path = os.path.join(directory, file)
        
        # Add file header to output
        output_text += f"<{file}>:\n"
        
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                
                # Get first 3 rows
                rows = []
                for i, row in enumerate(csv_reader):
                    if i < 3:
                        rows.append(row)
                    else:
                        break
                
                # Add rows to output
                for row in rows:
                    output_text += f"<{','.join(row)}>\n"
            
            # Add blank line after each file
            output_text += "\n"
        except Exception as e:
            output_text += f"<Error reading file: {str(e)}>\n\n"
    
    # Save output to file
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        outfile.write(output_text)
    
    print(f"CSV previews saved to {output_filename}")

# Example Usage:
# ./preview_csvs.py /path/to/csvs

if __name__ == "__main__":
    main()