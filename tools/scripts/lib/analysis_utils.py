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

def get_segment_columns(header_row):
    """
    Identifies segment columns (typically ending in '(Number)'), extracts the
    segment name, optional 'O' code (e.g., O1), and the participant count (N).
    Also returns the starting index of the segments in the header.

    Args:
        header_row (list): List of column names from the header.

    Returns:
        tuple: (list_of_core_segment_names, dict_of_segment_details, start_index)
               - list_of_core_segment_names: Contains names like 'All', 'O1: English', 'Africa'.
               - dict_of_segment_details: Maps *original full column name* (e.g., 'All (967)') 
                 to {'core_name': str, 'name': str, 'o_code': str|None, 'size': int|np.nan}.
               - start_index: The 0-based index where segments begin, or None if not found.
    """
    core_segment_names = []
    segment_details = {} # Maps original full name -> details
    start_index = None

    # Define potential markers *before* segments for different types
    # Order matters - check for more specific ones first
    poll_marker = "Responses"
    ask_opinion_markers = ["Sentiment"] # Segments usually start right after Sentiment
    ask_experience_marker = "Categories" # Segments usually start right after Categories
    
    # Check for Ask Opinion first (most distinct end columns before segments)
    if "Sentiment" in header_row and "Star" in header_row:
        try:
            start_index = header_row.index("Sentiment") + 1
        except ValueError:
            start_index = None 
    # Check for Ask Experience
    elif ask_experience_marker in header_row:
        try:
            start_index = header_row.index(ask_experience_marker) + 1
        except ValueError:
            start_index = None
    # Check for Poll
    elif poll_marker in header_row:
        try:
            start_index = header_row.index(poll_marker) + 1
        except ValueError:
            start_index = None
            
    if start_index is None:
         logging.warning(f"Could not determine segment start index based on typical markers in header: {header_row[:10]}...")
         return [], {}, None # Cannot proceed without a start index
         
    # Define known columns that might appear *after* segments in Ask types
    ask_end_markers = ["Submitted By", "Language", "Sample ID", "Participant ID"]

    # Regex to validate segment format and capture core name
    # Group 1 (Optional): O-code prefix (e.g., 'O1: ') 
    # Group 2: The actual segment name (e.g., 'English', 'Africa', 'All')
    # Group 3 (Optional): The size part including parens (e.g., ' (967)') - we'll extract number later
    pattern = re.compile(r'^(O\d+:\s*)?(.*?)\s*(\(\s*(?:\d+|N)\s*\))?\s*$')
    size_pattern = re.compile(r'\((\d+)\)') # To extract number from size part

    # Define columns that are definitely NOT segments, even if they appear after start_index
    KNOWN_NON_SEGMENT_COLS = {
        "Sentiment", "Star", "Categories", 
        "English Responses", "Original Responses", "Responses" # Add core names used by specific question types
    } 

    unique_core_names = set() # Keep track of core names found in this header

    for i, col_name in enumerate(header_row[start_index:], start=start_index):
        # Stop if we hit a known column that signals the end of segments for Ask questions
        if col_name in ask_end_markers:
            logging.debug(f"Stopping segment search at known end marker: '{col_name}'")
            break
        
        # Explicitly skip known non-segment columns that might appear mid-header
        if col_name in KNOWN_NON_SEGMENT_COLS:
            logging.debug(f"Skipping known non-segment column found after start_index: '{col_name}'")
            continue

        # Now, try to match the segment pattern
        match = pattern.match(col_name.strip()) # Strip whitespace from col name
        if match:
            o_code_prefix = match.group(1) if match.group(1) else "" # Includes 'O#: '
            name_part = match.group(2).strip()
            size_part = match.group(3) if match.group(3) else "" # Includes '(#)'

            # Construct the core name (O-code + Name Part)
            core_name = (o_code_prefix + name_part).strip()
            if not core_name: # Skip if somehow the core name is empty
                 logging.warning(f"Empty core segment name derived from column: '{col_name}'")
                 continue

            # Add to list only if it's the first time seeing this core name in *this header*
            if core_name not in unique_core_names:
                core_segment_names.append(core_name)
                unique_core_names.add(core_name)

            # --- Extract Details (Size, O-code number) --- 
            o_code = None
            if o_code_prefix:
                o_code_num_match = re.search(r'O(\d+)', o_code_prefix)
                if o_code_num_match:
                    o_code = f"O{o_code_num_match.group(1)}"
            
            size = np.nan
            if size_part:
                size_match = size_pattern.search(size_part)
                if size_match:
                    size_str = size_match.group(1)
                    if size_str.isdigit():
                        try:
                            size = int(size_str)
                        except ValueError:
                            logging.warning(f"Could not parse size digits '{size_str}' from segment column: {col_name}")
                elif '(N)' not in size_part: # Allow (N) explicitly
                    logging.warning(f"Unexpected size format '{size_part}' in segment column: {col_name}")

            # Store details mapped by the *original full column name*
            segment_details[col_name] = {
                'core_name': core_name,
                'name': name_part, # Just the name part (e.g., English, Africa)
                'o_code': o_code, 
                'size': size
            }
        else:
            # If it doesn't match segment pattern (and wasn't skipped/end marker), assume segments ended.
            logging.debug(f"Stopping segment search at column '{col_name}' (index {i}) as it doesn't match segment pattern.")
            break

    # Sort core names: 'All' first, then others alphabetically
    all_core = [name for name in core_segment_names if name.lower() == 'all']
    other_core = [name for name in core_segment_names if name.lower() != 'all']
    sorted_core_segment_names = sorted(all_core) + sorted(other_core)

    if not sorted_core_segment_names:
        logging.warning(f"No columns matched segment pattern after start index {start_index} in header: {header_row[start_index:start_index+5]}...")

    # Return list of unique core names, and the details dict mapped by original full name
    return sorted_core_segment_names, segment_details, start_index

# Removed placeholder print 