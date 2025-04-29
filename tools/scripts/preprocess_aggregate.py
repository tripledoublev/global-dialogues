import csv
import argparse
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration: Define Standard Output Columns ---
# Core columns expected in all rows
CORE_COLS = ["Question ID", "Question Type", "Question"]
# Columns specifically for Ask questions (will be blank for Polls)
# Star is sometimes included in Poll headers but should map here
ASK_SPECIFIC_COLS = ["Star", "Responses", "Original Responses", "Sentiment", "Submitted By", "Language", "Sample ID", "Participant ID"]
# Poll questions have a "Responses" column which maps to Ask's "Responses"
# We will handle the mapping logic later.

# --- Helper Functions ---

def is_metadata_row(row, min_cols=10):
    """Heuristic check if a row is likely metadata (e.g., Title, Date)."""
    if not row or len(row) < 2 or len(row) > min_cols: # Metadata rows are usually short
        return False
    # Check for common metadata keywords or lack of UUID
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    if not uuid_pattern.match(row[0].strip()) and row[0].strip() != "Question ID":
        # Could be 'Title', 'Date', etc. or the very first Name/Value line
        # Let's assume rows not starting with a UUID or 'Question ID' and being short are metadata
        if row[0].strip() not in ["Title", "Date", "Duration", "Participants", "Participant Limit", "Total Screener Polls", "Total Conversation Questions"]:
             # Check for the Name/Value pair format like ["""Name"""", "Global Dialogues..."]
             if len(row) == 2 and row[0].startswith('"""') and row[0].endswith('"""'):
                 return True
             else:
                 return False # Not a known metadata keyword or format
        return True
    return False

def is_header_row(row):
    """Checks if a row looks like a header row (starts with 'Question ID')."""
    return bool(row) and row[0].strip() == "Question ID"

def get_segment_columns(header_row):
    """Extracts segment columns (e.g., 'All(986)', 'O1: French (14)')."""
    segment_cols = []
    in_segments = False
    # Identify where standard columns end and segments begin
    # Poll header ends with 'Responses', Ask headers have more standard columns after
    poll_end_marker = "Responses"
    ask_end_markers = ["Star", "English Responses", "Original Responses", "Sentiment", "Submitted By", "Language", "Sample ID", "Participant ID"]

    for i, col in enumerate(header_row):
        if in_segments:
            # Stop if we hit one of the known ASK_SPECIFIC_COLS that might appear *after* segments
            if col in ask_end_markers:
                break
            segment_cols.append(col)
        # Determine start of segments
        elif col == poll_end_marker and i+1 < len(header_row) and '(' in header_row[i+1]: # Check next col looks like a segment
            in_segments = True
        elif col in ask_end_markers and i+1 < len(header_row) and '(' in header_row[i+1]: # Check next col looks like a segment
             # Handle case where segments start right after Star (if Star is present)
             if col == "Star":
                in_segments = True
             else:
                 # Segments might start after 'Responses' even in an 'Ask' header type
                 # Let's find the 'Responses' or 'English Responses' column first
                 try:
                     response_col_index = header_row.index("Responses") if "Responses" in header_row else header_row.index("English Responses")
                     if i > response_col_index and '(' in col:
                         in_segments = True
                         segment_cols.append(col) # Add the first segment col
                 except ValueError:
                     pass # No response column found, unlikely valid header

    # Basic sanity check
    if not segment_cols:
        logging.warning(f"Could not reliably identify segment columns in header: {header_row[:10]}...")
        # Fallback: assume everything after the initial known CORE_COLS and up to ASK_SPECIFIC_COLS start is a segment
        ask_start_indices = [i for i, col in enumerate(header_row) if col in ASK_SPECIFIC_COLS]
        first_ask_col_index = min(ask_start_indices) if ask_start_indices else len(header_row)
        potential_segments = header_row[len(CORE_COLS):first_ask_col_index]
        segment_cols = [col for col in potential_segments if '(' in col and ')' in col] # Filter for typical segment format
        if segment_cols:
             logging.warning(f"Using fallback logic, identified segments: {segment_cols[:5]}...")
        else:
            logging.error(f"Fallback failed. Cannot proceed without segment columns.")
            return []

    return segment_cols

def standardize_aggregate_csv(input_csv_path, output_csv_path):
    """
    Reads an aggregate CSV with potentially repeated headers and metadata,
    and writes a standardized version with only one header row and consistent columns.

    Args:
        input_csv_path (str): Path to the input aggregate CSV file.
        output_csv_path (str): Path to write the standardized output CSV file.
    """
    if not os.path.exists(input_csv_path):
        logging.error(f"Input file does not exist: {input_csv_path}")
        return

    standardized_header = None
    segment_cols = []
    rows_written = 0
    rows_skipped_meta = 0
    rows_skipped_header = 0
    current_header_map = {}

    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(input_csv_path, 'r', encoding='utf-8') as infile, \
             open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile:

            reader = csv.reader(infile)
            writer = None # Initialize DictWriter after header is finalized

            for i, row in enumerate(reader):
                if not row:
                    continue # Skip empty rows

                # --- Initial Header and Metadata Handling ---
                if standardized_header is None:
                    if is_metadata_row(row):
                        rows_skipped_meta += 1
                        logging.debug(f"Skipping metadata row {i+1}: {row[:2]}...")
                        continue
                    elif is_header_row(row):
                        logging.info(f"Found first header row at line {i+1}. Determining standard format.")
                        first_header = row
                        segment_cols = get_segment_columns(first_header)
                        if not segment_cols:
                            raise ValueError("Failed to identify segment columns from the first header.")

                        # Construct the final standardized header
                        standardized_header = CORE_COLS + segment_cols + ASK_SPECIFIC_COLS
                        logging.info(f"Standardized header set ({len(standardized_header)} columns): {standardized_header[:5]}...{standardized_header[-5:]}")

                        # Initialize DictWriter
                        writer = csv.DictWriter(outfile, fieldnames=standardized_header, extrasaction='ignore')
                        writer.writeheader()
                        rows_written += 1

                        # Set the map for the first block of data
                        current_header_map = {h: h for h in first_header}
                        # Special handling for response columns
                        if "English Responses" in current_header_map:
                            current_header_map["English Responses"] = "Responses" # Map English Responses to standard 'Responses'
                        if "Responses" in current_header_map and current_header_map["Responses"] != "Responses":
                            # Ensure the Poll 'Responses' also maps correctly if present in the first header
                            current_header_map["Responses"] = "Responses"

                        rows_skipped_header += 1 # Count this first header as skipped (only write standardized one)
                        continue
                    else:
                        # Should not happen if file is well-formed, but skip unexpected rows before first header
                        logging.warning(f"Skipping unexpected row {i+1} before first header: {row[:5]}...")
                        continue

                # --- Subsequent Row Handling ---
                if is_metadata_row(row):
                     rows_skipped_meta += 1
                     logging.debug(f"Skipping metadata row {i+1}: {row[:2]}...")
                     continue
                elif is_header_row(row):
                    rows_skipped_header += 1
                    logging.debug(f"Skipping repeated header row {i+1}")
                    # Update current header map for the next block
                    current_header_map = {h: h for h in row}
                    if "English Responses" in current_header_map:
                         current_header_map["English Responses"] = "Responses"
                    if "Responses" in current_header_map and current_header_map["Responses"] != "Responses":
                         current_header_map["Responses"] = "Responses"
                    continue
                else:
                    # --- Process Data Row ---
                    if writer is None:
                        logging.error(f"Error: Encountered data row {i+1} before a valid header was found. Skipping.")
                        continue

                    output_row_dict = {h: '' for h in standardized_header} # Initialize with blanks
                    num_cols_in_row = len(row)

                    # Map values based on the *current* header structure
                    current_cols = list(current_header_map.keys())
                    for idx, input_col_name in enumerate(current_cols):
                        if idx < num_cols_in_row:
                            target_col_name = current_header_map.get(input_col_name)
                            if target_col_name in output_row_dict:
                                output_row_dict[target_col_name] = row[idx]
                            # Handle Poll 'Responses' specifically if it wasn't mapped above
                            elif input_col_name == "Responses" and "Responses" in output_row_dict:
                                output_row_dict["Responses"] = row[idx]
                        else:
                            # Handle rows that might be shorter than the current header (e.g., missing optional end columns)
                            logging.debug(f"Row {i+1} has fewer columns ({num_cols_in_row}) than current header expects ({len(current_cols)}). Stopping mapping at index {idx}.")
                            break

                    # Handle potential merging/overwriting for the 'Responses' column
                    # The logic above prioritizes mapping from 'English Responses' first if it exists
                    # in the current_header_map due to the mapping `current_header_map["English Responses"] = "Responses"`

                    writer.writerow(output_row_dict)
                    rows_written += 1

        if writer is None:
             logging.error("Processing finished, but no valid header or data rows were found/written. Output file might be empty or incomplete.")
             raise ValueError("No valid data processed.")

        logging.info(f"Successfully standardized CSV.")
        logging.info(f"Rows written (including header): {rows_written}")
        logging.info(f"Metadata rows skipped: {rows_skipped_meta}")
        logging.info(f"Repeated header rows skipped: {rows_skipped_header}")
        logging.info(f"Standardized file saved to: {output_csv_path}")

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}", exc_info=True)
        # Clean up partially written file if error occurs
        if os.path.exists(output_csv_path):
             try:
                 os.remove(output_csv_path)
                 logging.warning(f"Removed partially written output file: {output_csv_path}")
             except OSError as remove_err:
                 logging.error(f"Failed to remove partially written file {output_csv_path}: {remove_err}")

def main():
    parser = argparse.ArgumentParser(description='Standardize an aggregate CSV file by removing repeated headers and metadata, and ensuring consistent columns.')
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