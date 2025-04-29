# Placeholder for shared analysis utility functions
import pandas as pd
import logging
import re # For parsing segment columns
import numpy as np
import os # For commonprefix in segment parsing helper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TODO: Define utility functions (e.g., load_data, parse_percentage, etc.)

def load_standardized_data(csv_path):
    """Loads the standardized aggregate CSV into a pandas DataFrame."""
    logging.info(f"Loading standardized data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        logging.info(f"Successfully loaded dataframe with shape: {df.shape}")
        # Basic validation - check for expected columns
        expected_cols = ["Question ID", "Question Type", "Question", "Responses"]
        # Removed Participant ID as it might not be present after standardization if only aggregate matters
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            logging.warning(f"Standardized CSV missing expected columns: {missing_cols}")
        return df
    except FileNotFoundError:
        logging.error(f"Standardized data file not found: {csv_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading standardized data from {csv_path}: {e}")
        return None

def parse_percentage(value):
    """Converts percentage strings ('X%') or ' - ' or empty to floats (0.X or NA)."""
    if isinstance(value, str):
        value = value.strip()
        if value == '-' or value == '':
            return pd.NA
        if value.endswith('%'):
            try:
                return float(value[:-1]) / 100.0
            except ValueError:
                return pd.NA
    elif pd.isna(value):
        return pd.NA
    # Handle potential numeric values that might be 0-100 instead of 0-1
    try:
        numeric_val = float(value)
        if 1 < numeric_val <= 100:
             return numeric_val / 100.0
        elif 0 <= numeric_val <= 1:
             return numeric_val
        else: # Handle unexpected numeric ranges if necessary
            return pd.NA
    except (ValueError, TypeError):
         return pd.NA

def get_segment_columns(df_columns):
    """
    Identifies segment columns (typically ending in '(Number)'), extracts the
    segment name, optional 'O' code (e.g., O1), and the participant count (N).
    Adapted from analyze_dialogues.py.

    Args:
        df_columns (list): List of column names from a DataFrame.

    Returns:
        tuple: (list_of_segment_column_names, dict_of_segment_details)
               The dict maps column names to {'name': str, 'o_code': str|None, 'size': int|np.nan}.
    """
    segment_cols = []
    segment_details = {}
    # Regex to capture:
    # Group 1 (Optional): 'O' code like 'O1:', 'O2:', etc. including the colon and potential spaces.
    # Group 2: The segment name (non-greedy match).
    # Group 3: The count within parentheses.
    # Allows for variations in spacing.
    pattern = re.compile(r'^(O\d+:\s*)?(.*?)\s*\(\s*(\d+|N)\s*\)\s*$') # Allow (N) as well

    for col in df_columns:
        match = pattern.match(col)
        if match:
            segment_cols.append(col)
            o_code_match = match.group(1) # Might be None if no 'O' prefix
            name = match.group(2).strip() # The captured segment name
            size_str = match.group(3) # The captured digits for size or 'N'

            # Extract just the 'O' code number if present
            o_code = None
            if o_code_match:
                 o_code_num_match = re.search(r'O(\d+)', o_code_match)
                 if o_code_num_match:
                     o_code = f"O{o_code_num_match.group(1)}" # Store as O1, O2 etc.

            size = np.nan # Default to NaN
            if size_str.isdigit():
                try:
                    size = int(size_str)
                except ValueError:
                    logging.warning(f"Could not parse extracted size digits '{size_str}' from segment column: {col}")
            elif size_str == 'N':
                # Keep size as NaN if it's just (N), indicates a placeholder
                pass
            else:
                 logging.warning(f"Unexpected size format '{size_str}' in segment column: {col}")


            segment_details[col] = {'name': name, 'o_code': o_code, 'size': size}

    # Sort columns: 'All(...)' first, then others alphabetically
    all_cols = [col for col in segment_cols if col.lower().startswith('all(')]
    other_cols = [col for col in segment_cols if not col.lower().startswith('all(')]
    sorted_segment_cols = sorted(all_cols) + sorted(other_cols)

    if not sorted_segment_cols and len(df_columns) > 5:
         # Check standard non-segment columns first before warning
         standard_cols = {'Question ID', 'Question Type', 'Question', 'Responses', 'English Response', 'Original Response'}
         potential_segments = [c for c in df_columns if c not in standard_cols and '(' in c]
         if potential_segments:
              logging.warning(f"Regex pattern did not find segment columns matching format 'Name (Number)' or 'O#: Name (Number)' in headers: {df_columns[:10]}... Check CSV format or regex pattern.")

    # Return both the sorted list of column names and the detailed dictionary
    return sorted_segment_cols, segment_details

# Removed placeholder print 