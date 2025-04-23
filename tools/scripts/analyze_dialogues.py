import pandas as pd
import numpy as np
import csv
import os
import pickle # Using pickle for simplicity for saving list of DFs
import argparse
import math
import re # For parsing segment columns
import warnings

# --- Configuration ---
# Default values, can be overridden by command-line args
DEFAULT_MIN_SEGMENT_SIZE = 30
CACHE_FILENAME = "processed_data.pkl"
# Set pandas display options for potentially wide DataFrames
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

# --- Helper Functions ---

def parse_percentage(value):
    """Converts percentage strings ('X%') or ' - ' to floats (0.X or NaN)."""
    if isinstance(value, (int, float)):
        # Allow for values that might already be numeric (e.g., from cache)
        if not np.isnan(value):
             return float(value)
        else:
             return np.nan # Keep existing NaNs
    if isinstance(value, str):
        value = value.strip()
        if value == '-' or value == '':
            return np.nan
        if value.endswith('%'):
            try:
                return float(value[:-1]) / 100.0
            except ValueError:
                # Catch cases like ' %' or non-numeric before %
                return np.nan # Error during conversion
    # Default to NaN if input is not recognized or fails conversion
    return np.nan

def get_segment_columns(df_columns):
    """
    Identifies segment columns (typically containing '(N)') and extracts N.

    Args:
        df_columns (list): List of column names from a DataFrame.

    Returns:
        tuple: (list_of_segment_column_names, dict_of_segment_name_to_size)
    """
    segment_cols = []
    segment_sizes = {}
    # Regex to find columns like "Segment Name (N)" or "All(N)" or "O1: xxx (N)"
    # Allows for variations in spacing, capitalization, and optional ':' before segment name
    pattern = re.compile(r'.*?\s*\(\s*([Nn])\s*\)\s*$')
    size_pattern = re.compile(r'\((\d+)\)') # Extracts the number N

    for col in df_columns:
        # Check if column name broadly matches segment pattern
        if pattern.match(col):
            segment_cols.append(col)
            # Try to extract the specific size N
            match = size_pattern.search(col)
            if match:
                try:
                    segment_sizes[col] = int(match.group(1))
                except ValueError:
                    segment_sizes[col] = np.nan # Failed to parse N
                    print(f"Warning: Could not parse size 'N' from segment column: {col}")
            else:
                 segment_sizes[col] = np.nan # Pattern matched but size extraction failed
                 # This might happen for 'All(N)' if N isn't numeric, handle later if needed
                 print(f"Warning: Could not extract numeric size 'N' from segment column: {col}")


    # Ensure 'All(N)' or 'All (N)' is included if missed by regex but present
    # And attempt to calculate its size if needed (e.g., sum of others? Needs context)
    all_n_col = None
    if 'All(N)' in df_columns and 'All(N)' not in segment_cols:
        all_n_col = 'All(N)'
    elif 'All (N)' in df_columns and 'All (N)' not in segment_cols:
         all_n_col = 'All (N)'

    if all_n_col:
        segment_cols.insert(0, all_n_col)
        match = size_pattern.search(all_n_col)
        if match:
            try:
                segment_sizes[all_n_col] = int(match.group(1))
            except ValueError:
                segment_sizes[all_n_col] = np.nan
        else:
             # If size wasn't in 'All(N)' column name, we might need to calculate it
             # For now, mark as NaN
             segment_sizes[all_n_col] = np.nan
             print(f"Warning: Size 'N' for '{all_n_col}' not found or calculable yet.")

    if not segment_cols and len(df_columns) > 7:
         print(f"Warning: Regex pattern did not find any segment columns like '(N)' in headers: {df_columns[:10]}... Check CSV format.")

    return segment_cols, segment_sizes


# --- Data Loading and Preprocessing ---

def load_and_preprocess_data(csv_path, cache_path, force_reparse=False, padding_rows=0):
    """
    Loads data from aggregate.csv, preprocesses it into a list of DataFrames
    (one per question), handling different question types and converting percentages.
    Uses caching (pickle) for speed.

    Args:
        csv_path (str): Path to the aggregate.csv file.
        cache_path (str): Path to save/load the cached data (.pkl).
        force_reparse (bool): If True, ignore cache and re-parse from CSV.
        padding_rows (int): Number of initial rows to skip in the CSV.

    Returns:
        list: A list of tuples, where each tuple contains (question_metadata, dataframe).
              question_metadata is a dict with 'id', 'type', 'text', 'segment_cols', 'segment_sizes'.
              Returns an empty list if loading fails.
    """
    if not force_reparse and os.path.exists(cache_path):
        print(f"Loading processed data from cache: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                processed_data = pickle.load(f)
            # Basic validation of cache structure
            if isinstance(processed_data, list):
                 # Check first element if list is not empty
                if not processed_data or \
                   (isinstance(processed_data[0], tuple) and len(processed_data[0]) == 2 and \
                    isinstance(processed_data[0][0], dict) and isinstance(processed_data[0][1], pd.DataFrame)):
                    print("Loaded from cache successfully.")
                    return processed_data
                else:
                    print("Cache data format invalid, reparsing...")
            else:
                 print("Cache data format invalid (not a list), reparsing...")
        except Exception as e:
            print(f"Error loading from cache: {e}. Reparsing...")

    print(f"Parsing data from CSV: {csv_path}")
    processed_data = []
    current_question_data = []

    try:
        # Use utf-8-sig to handle potential BOM (Byte Order Mark) at file start
        with open(csv_path, 'r', encoding='utf-8-sig') as file:
            csvreader = csv.reader(file)
            row_num = 0
            last_meaningful_row_num = 0 # Track last row with content

            for row in csvreader:
                row_num += 1
                if row_num <= padding_rows:
                    continue # Skip padding rows

                # Check if row is effectively empty (contains only whitespace or empty strings)
                is_empty_row = len(row) == 0 or all(not cell or cell.isspace() for cell in row)

                if not is_empty_row:
                    last_meaningful_row_num = row_num # Update on seeing content

                # Process block when an empty row is encountered *after* some data,
                # or when we hit the end of the file after the last meaningful row.
                # (This simplified end-of-file check might need refinement if files can end mid-data)
                should_process_block = is_empty_row and current_question_data

                if should_process_block:
                    try:
                        header = current_question_data[0]
                        # Ensure there's at least one data row beyond the header for metadata
                        if len(current_question_data) < 2:
                             print(f"Warning: Skipping block ending near row {row_num} - header only found. Header: {header}")
                             current_question_data = [] # Reset
                             continue

                        meta_row = current_question_data[1]

                        # Check for sufficient columns in metadata row
                        if len(meta_row) < 3:
                            print(f"Warning: Skipping block ending near row {row_num} - metadata row too short. MetaRow: {meta_row}")
                            current_question_data = [] # Reset
                            continue

                        q_id = meta_row[0].strip()
                        q_type = meta_row[1].strip()
                        q_text = meta_row[2].strip()

                        # Create DataFrame, handling potential variations in row lengths if necessary
                        # Pad shorter rows with NaN to match header length
                        num_cols = len(header)
                        data_rows = [row[:num_cols] + [np.nan]*(num_cols - len(row)) if len(row) < num_cols else row[:num_cols]
                                     for row in current_question_data[1:]]

                        df = pd.DataFrame(data_rows, columns=header)

                        # Identify segment columns and sizes
                        segment_cols, segment_sizes = get_segment_columns(df.columns)

                        # Apply percentage conversion only to identified segment columns
                        for col in segment_cols:
                            if col in df.columns:
                                # Suppress SettingWithCopyWarning temporarily if it occurs here
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)
                                    df[col] = df[col].apply(parse_percentage)
                            else:
                                print(f"Warning: Identified segment column '{col}' not found in DataFrame for QID {q_id}. Columns: {df.columns.tolist()}")


                        metadata = {'id': q_id, 'type': q_type, 'text': q_text, 'segment_cols': segment_cols, 'segment_sizes': segment_sizes}
                        processed_data.append((metadata, df))

                    except IndexError as e:
                        print(f"Error processing block ending near row {row_num}. Likely malformed data structure. Header: {current_question_data[0] if current_question_data else 'N/A'}. Error: {e}")
                    except Exception as e:
                        print(f"Unexpected error processing block ending near row {row_num}. Error: {e}")

                    current_question_data = [] # Reset for next question
                elif not is_empty_row:
                    current_question_data.append(row)

            # Process the last block if file doesn't end with blank row
            if current_question_data:
                try:
                    header = current_question_data[0]
                    if len(current_question_data) < 2:
                         print(f"Warning: Skipping final block - header only found. Header: {header}")
                    else:
                        meta_row = current_question_data[1]
                        if len(meta_row) < 3:
                            print(f"Warning: Skipping final block - metadata row too short. MetaRow: {meta_row}")
                        else:
                            q_id = meta_row[0].strip()
                            q_type = meta_row[1].strip()
                            q_text = meta_row[2].strip()

                            num_cols = len(header)
                            data_rows = [row[:num_cols] + [np.nan]*(num_cols - len(row)) if len(row) < num_cols else row[:num_cols]
                                         for row in current_question_data[1:]]
                            df = pd.DataFrame(data_rows, columns=header)

                            segment_cols, segment_sizes = get_segment_columns(df.columns)
                            for col in segment_cols:
                                if col in df.columns:
                                     with warnings.catch_warnings():
                                        warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)
                                        df[col] = df[col].apply(parse_percentage)
                                else:
                                    print(f"Warning: Identified segment column '{col}' not found in DataFrame for final QID {q_id}. Columns: {df.columns.tolist()}")

                            metadata = {'id': q_id, 'type': q_type, 'text': q_text, 'segment_cols': segment_cols, 'segment_sizes': segment_sizes}
                            processed_data.append((metadata, df))

                except IndexError as e:
                    print(f"Error processing final block. Likely malformed data structure. Header: {current_question_data[0] if current_question_data else 'N/A'}. Error: {e}")
                except Exception as e:
                    print(f"Unexpected error processing final block. Error: {e}")


    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return []
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        # Consider re-raising or logging more details depending on desired behavior
        return []

    # Save to cache if data was processed
    if processed_data:
        print(f"Saving processed data ({len(processed_data)} questions) to cache: {cache_path}")
        try:
            # Ensure cache directory exists
            cache_dir = os.path.dirname(cache_path)
            if cache_dir and not os.path.exists(cache_dir):
                 os.makedirs(cache_dir)
                 print(f"Created cache directory: {cache_dir}")
            with open(cache_path, 'wb') as f:
                pickle.dump(processed_data, f)
        except Exception as e:
            print(f"Error saving cache file {cache_path}: {e}")
    else:
         print("No data processed, cache not saved.")


    return processed_data

# --- Analysis Functions ---
# Placeholder for future functions (divergence, bridging, plotting, etc.)
# ...

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Global Dialogues aggregate data.")
    
    # --- Input file arguments (mutually exclusive) ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--gd_number", type=int, help="Global Dialogue cadence number (e.g., 1, 2, 3). Constructs path like Data/GDi/GDi_aggregate.csv")
    input_group.add_argument("--csv_filepath", help="Explicit path to the aggregate.csv file.")

    # --- Other arguments ---
    parser.add_argument("-o", "--output_dir", default="analysis_output", help="Directory to save results and cache.")
    parser.add_argument("-s", "--min_segment_size", type=int, default=DEFAULT_MIN_SEGMENT_SIZE, help="Minimum participant size for a segment to be included in analysis.")
    parser.add_argument("--force_reparse", action="store_true", help="Force reparsing from CSV, ignoring cache.")
    # Set default padding based on notebook observation
    parser.add_argument("--padding_rows", type=int, default=9, help="Number of header/junk rows to skip at the start of the CSV.")

    args = parser.parse_args()

    # --- Determine and Validate Input CSV Path ---
    input_csv_path = None
    if args.gd_number:
        gd_num = args.gd_number
        # Construct the relative path from the workspace root
        constructed_path = os.path.join("Data", f"GD{gd_num}", f"GD{gd_num}_aggregate.csv")
        print(f"Attempting to use constructed path for GD{gd_num}: {constructed_path}")
        if not os.path.exists(constructed_path):
             print(f"Error: Constructed path does not exist: {constructed_path}")
             print("Please ensure the Data/GDi/GDi_aggregate.csv structure exists relative to your workspace root, or provide an explicit --csv_filepath.")
             exit(1)
        input_csv_path = constructed_path
    elif args.csv_filepath:
        print(f"Attempting to use provided file path: {args.csv_filepath}")
        if not os.path.exists(args.csv_filepath):
            print(f"Error: Provided file path does not exist: {args.csv_filepath}")
            exit(1)
        input_csv_path = args.csv_filepath
    else:
        # This case should technically not be reached due to the mutually exclusive group being required
        print("Error: No input specified. Please use either --gd_number or --csv_filepath.")
        exit(1)

    print(f"Using input file: {input_csv_path}")

    # --- Setup Output Directory ---
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            print(f"Created output directory: {args.output_dir}")
        except OSError as e:
             print(f"Error creating output directory {args.output_dir}: {e}")
             exit(1) # Exit if we can't create the output dir

    cache_file = os.path.join(args.output_dir, CACHE_FILENAME)

    # --- Load and Preprocess Data ---
    all_questions_data = load_and_preprocess_data(
        input_csv_path, # Use the validated path
        cache_file,
        args.force_reparse,
        args.padding_rows
    )

    if not all_questions_data:
        print("Failed to load or process data. Exiting.")
        exit(1)

    print(f"\nSuccessfully loaded/processed {len(all_questions_data)} questions from {input_csv_path}.")

    # --- Example: Print info about the first few questions ---
    print("\n--- Sample Processed Data ---")
    for i, (metadata, df) in enumerate(all_questions_data[:3]):
        print(f"\n--- Question {i+1} ---")
        print(f"  ID  : {metadata.get('id', 'N/A')}")
        print(f"  Type: {metadata.get('type', 'N/A')}")
        q_text = metadata.get('text', '')
        print(f"  Text: {q_text[:100]}{'...' if len(q_text) > 100 else ''}") # Print first 100 chars
        print(f"  Shape: {df.shape}")
        segment_cols = metadata.get('segment_cols', [])
        segment_sizes = metadata.get('segment_sizes', {})
        print(f"  Segment Columns Identified ({len(segment_cols)}): {segment_cols[:5]}{'...' if len(segment_cols) > 5 else ''}") # Print first 5 segments
        print(f"  Segment Sizes (first 5): { {k: segment_sizes.get(k, 'N/A') for k in segment_cols[:5]} }")
        print(f"  DataFrame Columns: {df.columns.tolist()[:8]}{'...' if len(df.columns) > 8 else ''}") # First 8 cols
        # print(df.head(2)) # Uncomment to see first 2 rows of DataFrame


    # --- Filter segments based on size ---
    print(f"\nFiltering segments with size < {args.min_segment_size}...")
    original_total_segments = 0
    filtered_total_segments = 0
    processed_questions_data_filtered = [] # Store results after filtering

    for metadata, df in all_questions_data:
        original_segments = metadata.get('segment_cols', [])
        segment_sizes = metadata.get('segment_sizes', {})
        original_total_segments += len(original_segments)

        # Filter segment columns based on size
        # Use np.nan_to_num to treat NaN sizes as 0 for comparison
        valid_segments = [
            col for col in original_segments
            if col in segment_sizes and np.nan_to_num(segment_sizes[col]) >= args.min_segment_size
        ]
        filtered_total_segments += len(valid_segments)

        # Create a new metadata dict with filtered segments list
        # Keep original segment_sizes dict untouched for reference if needed
        filtered_metadata = metadata.copy()
        filtered_metadata['analysis_segment_cols'] = valid_segments # Use a distinct key

        processed_questions_data_filtered.append((filtered_metadata, df))

        # Log changes for this question if any segments were removed
        removed_count = len(original_segments) - len(valid_segments)
        if removed_count > 0:
            print(f"  QID {metadata.get('id', 'N/A')}: Removed {removed_count} segments. Kept {len(valid_segments)} / {len(original_segments)}.")

    # Replace the main list with the one containing filtered segment lists for analysis
    all_questions_data = processed_questions_data_filtered
    print(f"Segment filtering complete. Total segments kept for analysis: {filtered_total_segments} / {original_total_segments}.")


    # --- Placeholder for calling analysis functions ---
    print("\nAnalysis functions (divergence, consensus, heatmaps) need to be implemented next.")
    # Example (future):
    # divergence_results = calculate_divergence(all_questions_data, args.output_dir)
    # consensus_results = calculate_consensus_profiles(all_questions_data, args.output_dir)
    # generate_indicator_heatmaps(indicator_codesheet_path, all_questions_data, args.output_dir)

    print("\nScript finished.") 