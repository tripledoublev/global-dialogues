import csv
import argparse
import os
import logging
import re
from collections import OrderedDict # To preserve segment order somewhat
from lib.analysis_utils import get_segment_columns # Import the updated function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Define Standardized Output Columns (Core and Ask-Specific) ---
# We will add segment columns dynamically later
CORE_COLS = ["Question ID", "Question Type", "Question"]
POLL_SPECIFIC_COLS = ["ResponseOption"] # For the chosen option in Polls
ASK_SPECIFIC_COLS = [
    "ResponseText",        # Standardized name for English/primary text response
    "OriginalResponseText", # Standardized name for original language response
    "Star",                # Ask Opinion specific
    "Categories",          # Ask Experience specific
    "Sentiment",
    "Submitted By",
    "Language",
    "Sample ID",
    "Participant ID"
]
# Order for the final header: Core -> Poll -> Ask -> Segments
FINAL_HEADER_ORDER_BASE = CORE_COLS + POLL_SPECIFIC_COLS + ASK_SPECIFIC_COLS

# --- Helper Functions ---

def is_metadata_row(row, min_cols=10):
    """Heuristic check if a row is likely metadata (e.g., Title, Date)."""
    if not row or len(row) < 2 or len(row) > min_cols: # Metadata rows are usually short
        return False
    # Check for common metadata keywords or lack of UUID
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    if not uuid_pattern.match(row[0].strip()) and row[0].strip() != "Question ID":
         # Simplified: Assume rows not starting with UUID or 'Question ID' and being short are metadata
         # This might need refinement if other data row types exist without UUIDs
         return True
    return False

def is_header_row(row):
    """Checks if a row looks like a header row (starts with 'Question ID')."""
    return bool(row) and row[0].strip() == "Question ID"

def determine_header_type(header_row):
    """Determine if header is for Poll, Ask Opinion, or Ask Experience."""
    if "Star" in header_row and "English Responses" in header_row:
        return "Ask Opinion"
    elif "Categories" in header_row and "English Responses" in header_row:
         # Note: Could potentially conflict if 'Categories' appears elsewhere. Robustness check needed?
         # Assuming 'Categories' column is specific to Ask Experience for now.
         return "Ask Experience"
    elif "Responses" in header_row and "English Responses" not in header_row:
         # Polls have "Responses" but not "English Responses" or "Star" or "Categories" typically
         return "Poll"
    else:
        logging.warning(f"Could not determine header type for: {header_row[:10]}...")
        return "Unknown" # Fallback

def build_column_map(header_row, header_type, segment_cols_in_header):
    """Builds a map from input header columns to standardized column names."""
    mapping = {}
    for col in header_row:
        # Core columns map directly
        if col in CORE_COLS:
            mapping[col] = col
        # Type-specific mappings
        elif header_type == "Poll" and col == "Responses":
            mapping[col] = "ResponseOption"
        elif header_type in ["Ask Opinion", "Ask Experience"] and col == "English Responses":
            mapping[col] = "ResponseText"
        elif header_type in ["Ask Opinion", "Ask Experience"] and col == "Original Responses":
             mapping[col] = "OriginalResponseText"
        # Ask specific columns map directly
        elif col in ["Star", "Categories", "Sentiment", "Submitted By", "Language", "Sample ID", "Participant ID"]:
             mapping[col] = col
        # Segment columns map directly (their names are the standard)
        elif col in segment_cols_in_header:
             mapping[col] = col
        # Handle potential edge cases or ignore others?
        # else:
        #     logging.debug(f"Column '{col}' in header type '{header_type}' not explicitly mapped.")
    return mapping

def collect_all_segment_columns(input_csv_path):
    """Pass 1: Read the CSV to find all unique *core* segment column names across all headers."""
    all_core_segments = OrderedDict() # Use OrderedDict to keep insertion order
    logging.info("Starting Pass 1: Collecting all unique *core* segment column names...")
    try:
        with open(input_csv_path, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            processed_headers = 0
            temp_details_store = {} # Optional: store details from first encounter if needed later

            for i, row in enumerate(reader):
                if is_header_row(row):
                    processed_headers += 1
                    # get_segment_columns now returns core names in the list
                    core_names, details_dict, seg_start_idx = get_segment_columns(row)
                    if core_names:
                         for core_name in core_names:
                            if core_name not in all_core_segments:
                                all_core_segments[core_name] = None # Add core segment name
                                # Optional: Store details if useful (e.g., size from first time seen)
                                # Find the original col name that produced this core name from details_dict
                                # original_col = next((k for k, v in details_dict.items() if v.get('core_name') == core_name), None)
                                # if original_col: temp_details_store[core_name] = details_dict[original_col]
                    else:
                         # This condition remains the same
                         logging.warning(f"Pass 1: No segments identified in header row {i+1}. Header: {row[:10]}...")

        logging.info(f"Pass 1 complete. Found {len(all_core_segments)} unique *core* segment names across {processed_headers} headers.")
        # Sort core names: 'All' first, then others alphabetically
        all_core = [name for name in all_core_segments.keys() if name.lower() == 'all']
        other_core = [name for name in all_core_segments.keys() if name.lower() != 'all']
        sorted_core_segment_names = sorted(all_core) + sorted(other_core)
        
        return sorted_core_segment_names # Return the sorted list of unique core names

    except FileNotFoundError:
        logging.error(f"Input file not found during Pass 1: {input_csv_path}")
        return None
    except Exception as e:
        logging.error(f"Error during Pass 1 (collecting segments): {e}", exc_info=True)
        return None


def standardize_aggregate_csv(input_csv_path, output_csv_path):
    """
    Reads an aggregate CSV with varying headers per question block,
    and writes a standardized version with a single, comprehensive header,
    correctly mapping data from each block based on core segment names.

    Args:
        input_csv_path (str): Path to the input aggregate CSV file.
        output_csv_path (str): Path to write the standardized output CSV file.
    """
    if not os.path.exists(input_csv_path):
        logging.error(f"Input file does not exist: {input_csv_path}")
        return

    # --- Pass 1: Get all unique *core* segment columns ---
    all_core_segment_names = collect_all_segment_columns(input_csv_path)
    if all_core_segment_names is None:
        logging.error("Failed to collect segment columns. Aborting standardization.")
        return

    # --- Define the Final Standardized Header (using core segment names) ---
    standardized_header = FINAL_HEADER_ORDER_BASE + all_core_segment_names
    logging.info(f"Final standardized header defined with *core* segment names ({len(standardized_header)} columns).")
    logging.debug(f"Standardized Header (Core Segments): {standardized_header}")

    # --- Pass 2: Process and Write Data ---
    logging.info("Starting Pass 2: Processing data and writing standardized file...")
    rows_written = 0
    rows_skipped_meta = 0
    rows_processed_data = 0
    headers_encountered = 0
    current_header_row = []
    current_column_map = {} # Maps *input column name* -> *standardized column name*
    current_segment_details = {} # Stores details for the current header block

    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(input_csv_path, 'r', encoding='utf-8') as infile, \
             open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile:

            reader = csv.reader(infile)
            # Use the standardized header with CORE segment names
            writer = csv.DictWriter(outfile, fieldnames=standardized_header, extrasaction='ignore')
            writer.writeheader()
            rows_written += 1

            for i, row in enumerate(reader):
                if not row or all(not cell or cell.isspace() for cell in row):
                    continue # Skip empty or blank rows

                if is_header_row(row):
                    headers_encountered += 1
                    logging.debug(f"Processing header row {i+1}")
                    current_header_row = row
                    header_type = determine_header_type(current_header_row)
                    # Get segment details (maps original full name -> details incl. core_name)
                    _, current_segment_details, _ = get_segment_columns(current_header_row)
                    
                    # --- Rebuild current_column_map correctly --- 
                    current_column_map = {} 
                    for input_col_name in current_header_row:
                        # Standard non-segment columns
                        if input_col_name in CORE_COLS:
                            current_column_map[input_col_name] = input_col_name
                        elif header_type == "Poll" and input_col_name == "Responses":
                            current_column_map[input_col_name] = "ResponseOption"
                        elif header_type in ["Ask Opinion", "Ask Experience"] and input_col_name == "English Responses":
                            current_column_map[input_col_name] = "ResponseText"
                        elif header_type in ["Ask Opinion", "Ask Experience"] and input_col_name == "Original Responses":
                            current_column_map[input_col_name] = "OriginalResponseText"
                        elif input_col_name in ["Star", "Categories", "Sentiment", "Submitted By", "Language", "Sample ID", "Participant ID"]:
                            current_column_map[input_col_name] = input_col_name
                        # Segment columns: Map original name to its core name
                        elif input_col_name in current_segment_details:
                            core_name = current_segment_details[input_col_name].get('core_name')
                            if core_name:
                                current_column_map[input_col_name] = core_name
                            else:
                                logging.warning(f"Could not find core_name for segment '{input_col_name}' in header row {i+1}")
                    # ----------------------------------------

                    logging.debug(f"  Header type: {header_type}. Map created for {len(current_column_map)} columns.")
                    continue # Don't write header rows to output

                elif is_metadata_row(row):
                     rows_skipped_meta += 1
                     logging.debug(f"Skipping metadata row {i+1}: {row[:2]}...")
                     continue

                elif not current_header_row:
                     # Data row encountered before the first header
                     logging.warning(f"Skipping data row {i+1} found before any header: {row[:5]}...")
                     continue

                else:
                    # --- Process Data Row ---
                    output_row_dict = {h: '' for h in standardized_header} # Initialize with blanks
                    num_cols_in_data_row = len(row)
                    num_cols_in_current_header = len(current_header_row)

                    # Use the map derived from the most recent header
                    for idx, input_col_name in enumerate(current_header_row):
                        if idx >= num_cols_in_data_row:
                            # Data row is shorter than its header, stop processing this row's columns
                            if num_cols_in_data_row < num_cols_in_current_header:
                                 logging.debug(f"Row {i+1} is shorter ({num_cols_in_data_row}) than its header ({num_cols_in_current_header}). Truncating data.")
                            break

                        standardized_col_name = current_column_map.get(input_col_name)
                        if standardized_col_name:
                             # Only write if the column exists in our standardized header
                             if standardized_col_name in output_row_dict:
                                output_row_dict[standardized_col_name] = row[idx]
                        #else:
                             # Log columns from input header that weren't mapped?
                             # logging.debug(f"Column '{input_col_name}' from row {i+1}'s header was not mapped to standardized header.")

                    writer.writerow(output_row_dict)
                    rows_written += 1
                    rows_processed_data += 1

        if headers_encountered == 0:
             logging.error("Processing finished, but no header rows were found. Output file might be invalid.")
             raise ValueError("No header rows found in input file.")
        if rows_processed_data == 0:
             logging.warning("Processing finished, but no data rows were processed. Output file might only contain the header.")


        logging.info(f"Successfully standardized CSV.")
        logging.info(f"Headers encountered: {headers_encountered}")
        logging.info(f"Data rows processed: {rows_processed_data}")
        logging.info(f"Total rows written (incl. header): {rows_written}")
        logging.info(f"Metadata rows skipped: {rows_skipped_meta}")
        logging.info(f"Standardized file saved to: {output_csv_path}")

    except Exception as e:
        logging.error(f"An error occurred during Pass 2 (processing data): {e}", exc_info=True)
        # Clean up partially written file if error occurs
        if os.path.exists(output_csv_path):
             try:
                 os.remove(output_csv_path)
                 logging.warning(f"Removed partially written output file: {output_csv_path}")
             except OSError as remove_err:
                 logging.error(f"Failed to remove partially written file {output_csv_path}: {remove_err}")

def main():
    parser = argparse.ArgumentParser(description='Standardize an aggregate CSV file by removing repeated headers and metadata, handling variable headers per question block, and ensuring consistent columns.')
    parser.add_argument('input_file', help='Path to the input aggregate CSV file.')
    parser.add_argument('output_file', help='Path to save the standardized output CSV file.')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Basic validation
    if not args.input_file.lower().endswith('.csv'):
        logging.warning("Input file might not be a CSV.")
    if not args.output_file.lower().endswith('.csv'):
        logging.warning("Output file might not be a CSV.")

    standardize_aggregate_csv(args.input_file, args.output_file)

if __name__ == "__main__":
    main() 