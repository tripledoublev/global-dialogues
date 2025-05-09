import pandas as pd
import argparse
import os
import re

def extract_sanity_data(gd_number):
    """
    Extracts specified columns from the GD<N>_aggregate_standardized.csv file,
    filters for 'Ask Opinion' questions, renames columns, and saves to
    GD<N>_sanity_upload.csv.
    """
    input_file = os.path.join("Data", f"GD{gd_number}", f"GD{gd_number}_aggregate_standardized.csv")
    output_file = os.path.join("Data", f"GD{gd_number}", f"GD{gd_number}_sanity_upload.csv")

    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return

    # Filter for 'Ask Opinion' questions
    df_opinion = df[df['Question Type'] == 'Ask Opinion'].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Define columns to select and their new names
    columns_to_select = {
        'Question ID': 'Question ID',
        'Question Type': 'Question Type',
        'Question': 'Question',
        'Response': 'English Responses', # Original name is 'Response' in standardized file
        'Submitted By': 'Submitted By',
        'Language': 'Language',
        'Participant ID': 'Participant ID',
        'Sentiment': 'Sentiment',
        'All': 'All'
    }

    # Select and rename columns
    # Ensure all expected columns exist before trying to select them
    existing_columns_to_select = {og_col: new_col for og_col, new_col in columns_to_select.items() if og_col in df_opinion.columns}
    
    if len(existing_columns_to_select) != len(columns_to_select):
        missing_cols = set(columns_to_select.keys()) - set(existing_columns_to_select.keys())
        print(f"Warning: The following expected columns were not found in the input file and will be skipped: {missing_cols}")
        if not existing_columns_to_select:
            print("Error: None of the expected columns were found. Cannot proceed.")
            return
            
    df_selected = df_opinion[list(existing_columns_to_select.keys())].copy() # Use .copy()
    df_selected.rename(columns=existing_columns_to_select, inplace=True)

    # Extract the numeric part of 'All' column if it exists and was selected
    if 'All' in df_selected.columns:
        # Extract number before (N), e.g., "0.85 (N=123)" -> "0.85"
        # Handle cases where it might already be numeric or have different formatting
        df_selected['All'] = df_selected['All'].astype(str).str.extract(r'(\d*\.?\d+)').iloc[:, 0]
        df_selected['All'] = df_selected['All'].astype(str) + '%' # Append '%'


    # Save to new CSV
    try:
        df_selected.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Successfully created {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract specific columns for sanity checking from GD<N>_aggregate_standardized.csv.")
    parser.add_argument("gd_number", type=int, help="The Global Dialogue number (e.g., 3).")
    args = parser.parse_args()

    extract_sanity_data(args.gd_number) 