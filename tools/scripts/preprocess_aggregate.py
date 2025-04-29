import csv
import argparse
import os
import logging
import re
from collections import OrderedDict # To preserve segment order somewhat
from lib.analysis_utils import get_segment_columns # Import the updated function
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Define Standardized Output Columns ---
# Define the base columns in the desired final order.
# We will add *core* segment columns dynamically later.
FINAL_HEADER_ORDER_BASE = [
    "Question ID", 
    "Question Type", 
    "Question",
    "Response",             # Merged column for Poll options or Ask text responses
    "OriginalResponse",     # Renamed original text response column
    "Star",                 # Ask Opinion specific
    "Categories",           # Ask Experience specific
    "Sentiment",            # Single Sentiment column
    "Submitted By",
    "Language",
    "Sample ID",
    "Participant ID"
]

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
    """Builds a map from input header columns to *new* standardized column names."""
    mapping = {}
    for col in header_row:
        # Core columns map directly (Question ID, Type, Text)
        if col in ["Question ID", "Question Type", "Question"]:
            mapping[col] = col
        # Map Poll 'Responses' to new 'Response'
        elif header_type == "Poll" and col == "Responses":
            mapping[col] = "Response"
        # Map Ask 'English Responses' to new 'Response'
        elif header_type in ["Ask Opinion", "Ask Experience"] and col == "English Responses":
            mapping[col] = "Response" 
        # Map Ask 'Original Responses' to new 'OriginalResponse'
        elif header_type in ["Ask Opinion", "Ask Experience"] and col == "Original Responses":
             mapping[col] = "OriginalResponse"
        # Other specific columns map directly to their single standardized name
        elif col in ["Star", "Categories", "Sentiment", "Submitted By", "Language", "Sample ID", "Participant ID"]:
             mapping[col] = col
        # Segment columns: Map original full name to its *core* name
        # This part relies on segment_details being available where this is called
        # We will handle this adjustment directly in the main processing loop now.
        # elif col in segment_cols_in_header: # This check needs context
        #      mapping[col] = get_core_name_for_segment(col) # Need a way to get core name
        
    return mapping # Note: Segment mapping is now handled in the main loop

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

def get_question_info_from_row(row, header_row, column_map):
    """Extracts QID and QText from the first data row of a block."""
    qid, qtext = None, None
    try:
        # Find the standardized column name mapped from the input header's columns
        qid_input_col = next((k for k, v in column_map.items() if v == "Question ID"), None)
        qtext_input_col = next((k for k, v in column_map.items() if v == "Question"), None)
        
        if qid_input_col and qtext_input_col:
            qid_idx = header_row.index(qid_input_col)
            qtext_idx = header_row.index(qtext_input_col)
            if qid_idx < len(row) and qtext_idx < len(row):
                 qid = row[qid_idx]
                 qtext = row[qtext_idx]
    except (ValueError, IndexError) as e:
        logging.warning(f"Could not extract QID/QText from row: {row[:5]}... Error: {e}")
    return qid, qtext

def standardize_aggregate_csv(input_csv_path, output_csv_path, segment_counts_output_path=None):
    """
    Reads an aggregate CSV with varying headers per question block,
    writes a standardized version with a single, comprehensive header,
    and optionally writes a CSV containing segment participant counts per question.

    Args:
        input_csv_path (str): Path to the input aggregate CSV file.
        output_csv_path (str): Path to write the standardized output CSV file.
        segment_counts_output_path (str, optional): Path to write the segment counts per question CSV.
    """
    if not os.path.exists(input_csv_path):
        logging.error(f"Input file does not exist: {input_csv_path}")
        return

    # --- Pass 1: Get all unique *core* segment columns ---
    all_core_segment_names = collect_all_segment_columns(input_csv_path)
    if all_core_segment_names is None:
        logging.error("Failed to collect segment columns. Aborting standardization.")
        return

    # --- Define Headers --- 
    standardized_header = FINAL_HEADER_ORDER_BASE + all_core_segment_names
    segment_counts_header = ["Question ID", "Question Text"] + all_core_segment_names
    
    logging.info(f"Standardized header defined ({len(standardized_header)} columns).")
    if segment_counts_output_path:
        logging.info(f"Segment counts header defined ({len(segment_counts_header)} columns).")

    # --- Pass 2: Process and Write Data --- 
    logging.info("Starting Pass 2: Processing data and writing standardized file...")
    if segment_counts_output_path:
        logging.info("Segment counts per question will also be generated.")
        
    rows_written_std = 0
    rows_written_counts = 0
    rows_skipped_meta = 0
    rows_processed_data = 0
    headers_encountered = 0
    current_header_row = []
    current_column_map = {} # Maps input col -> standardized col
    current_segment_details = {} # Maps input full segment name -> details (core_name, size)
    question_segment_counts = OrderedDict() # Stores {qid: {segment_core_name: count, ...}} 
    current_qid = None
    current_qtext = None

    output_dir_std = os.path.dirname(output_csv_path)
    if output_dir_std:
        os.makedirs(output_dir_std, exist_ok=True)
    if segment_counts_output_path:
         output_dir_counts = os.path.dirname(segment_counts_output_path)
         if output_dir_counts:
             os.makedirs(output_dir_counts, exist_ok=True)

    try:
        with open(input_csv_path, 'r', encoding='utf-8') as infile, \
             open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile_std:

            reader = csv.reader(infile)
            writer_std = csv.DictWriter(outfile_std, fieldnames=standardized_header, extrasaction='ignore')
            writer_std.writeheader()
            rows_written_std += 1

            for i, row in enumerate(reader):
                if not row or all(not cell or cell.isspace() for cell in row):
                    current_qid = None # Reset QID tracker on blank rows
                    continue # Skip empty or blank rows

                if is_header_row(row):
                    headers_encountered += 1
                    current_qid = None # Reset QID, needs to be found in first data row
                    current_qtext = None
                    logging.debug(f"Processing header row {i+1}")
                    current_header_row = row
                    header_type = determine_header_type(current_header_row)
                    # Get segment details (maps original full name -> details incl. core_name, size)
                    _, current_segment_details, _ = get_segment_columns(current_header_row)
                    
                    # Rebuild current_column_map for this header block
                    current_column_map = {} 
                    for input_col_name in current_header_row:
                        # Map core Question/ID/Type
                        if input_col_name in ["Question ID", "Question Type", "Question"]:
                            current_column_map[input_col_name] = input_col_name
                        # Map Poll 'Responses' to 'Response'
                        elif header_type == "Poll" and input_col_name == "Responses":
                            current_column_map[input_col_name] = "Response"
                        # Map Ask 'English Responses' to 'Response'
                        elif header_type in ["Ask Opinion", "Ask Experience"] and input_col_name == "English Responses":
                            current_column_map[input_col_name] = "Response" 
                        # Map Ask 'Original Responses' to 'OriginalResponse'
                        elif header_type in ["Ask Opinion", "Ask Experience"] and input_col_name == "Original Responses":
                            current_column_map[input_col_name] = "OriginalResponse"
                        # Map other standard metadata cols
                        elif input_col_name in ["Star", "Categories", "Sentiment", "Submitted By", "Language", "Sample ID", "Participant ID"]:
                            current_column_map[input_col_name] = input_col_name
                        # Segment columns: Map original full name to its core name
                        elif input_col_name in current_segment_details:
                            core_name = current_segment_details[input_col_name].get('core_name')
                            if core_name:
                                current_column_map[input_col_name] = core_name
                            else:
                                logging.warning(f"Could not find core_name for segment '{input_col_name}' in header row {i+1}")
                        # else: Input column not needed in standardized output, ignore
                    
                    logging.debug(f"  Header type: {header_type}. Map created for {len(current_column_map)} columns.")
                    continue # Don't write header rows to output

                elif is_metadata_row(row):
                     rows_skipped_meta += 1
                     logging.debug(f"Skipping metadata row {i+1}: {row[:2]}...")
                     current_qid = None # Reset QID tracker
                     continue

                elif not current_header_row:
                     # Data row encountered before the first header
                     logging.warning(f"Skipping data row {i+1} found before any header: {row[:5]}...")
                     continue

                else:
                    # --- Process Data Row ---
                    # If generating segment counts, try to get QID/QText from first data row
                    if segment_counts_output_path and current_qid is None:
                         # Try to extract QID and QText from this row
                         potential_qid, potential_qtext = get_question_info_from_row(row, current_header_row, current_column_map)
                         if potential_qid:
                              current_qid = potential_qid
                              current_qtext = potential_qtext
                              logging.debug(f"  Identified QID: {current_qid} for current block.")
                              # Initialize count dict for this QID if not present
                              if current_qid not in question_segment_counts:
                                   question_segment_counts[current_qid] = {
                                       'Question ID': current_qid,
                                       'Question Text': current_qtext
                                   }
                                   # Populate counts from the current_segment_details (from the header)
                                   for details in current_segment_details.values():
                                       core_name = details.get('core_name')
                                       size = details.get('size')
                                       if core_name and core_name in segment_counts_header: # Check if core name is expected
                                           # Use pd.NA for numpy NaN, or keep integer
                                           question_segment_counts[current_qid][core_name] = int(size) if pd.notna(size) else pd.NA
                                       elif core_name:
                                            logging.warning(f"Core segment '{core_name}' found in header details but not in expected counts header.")
                         else:
                             logging.warning(f"Could not extract QID from first data row {i+1} after header. Segment counts for this block might be missed.")
                    
                    # --- Write standardized data row ---
                    output_row_dict = {h: '' for h in standardized_header} # Initialize with blanks
                    num_cols_in_data_row = len(row)
                    num_cols_in_current_header = len(current_header_row)

                    # Use the map derived from the most recent header
                    for idx, input_col_name in enumerate(current_header_row):
                        if idx >= num_cols_in_data_row:
                            if num_cols_in_data_row < num_cols_in_current_header:
                                 logging.debug(f"Row {i+1} is shorter ({num_cols_in_data_row}) than its header ({num_cols_in_current_header}). Truncating data.")
                            break

                        standardized_col_name = current_column_map.get(input_col_name)
                        if standardized_col_name:
                             if standardized_col_name in output_row_dict:
                                output_row_dict[standardized_col_name] = row[idx]

                    writer_std.writerow(output_row_dict)
                    rows_written_std += 1
                    rows_processed_data += 1

        # --- Final Checks and Summary for Standardized Output ---
        if headers_encountered == 0:
             logging.error("Processing finished, but no header rows were found. Standardized output file might be invalid.")
             # Don't raise error here yet, maybe counts can still be written
        if rows_processed_data == 0:
             logging.warning("Processing finished, but no data rows were processed for standardized output.")
             
        logging.info(f"Standardized CSV processing complete.")
        logging.info(f"  Headers encountered: {headers_encountered}")
        logging.info(f"  Data rows processed: {rows_processed_data}")
        logging.info(f"  Total rows written (incl. header): {rows_written_std}")
        logging.info(f"  Metadata rows skipped: {rows_skipped_meta}")
        logging.info(f"  Standardized file saved to: {output_csv_path}")

    except Exception as e:
        logging.error(f"An error occurred during Pass 2 (standardized data processing): {e}", exc_info=True)
        # Clean up partially written standardized file
        if os.path.exists(output_csv_path):
             try: os.remove(output_csv_path); logging.warning(f"Removed partially written standardized output file: {output_csv_path}")
             except OSError as remove_err: logging.error(f"Failed to remove partially written file {output_csv_path}: {remove_err}")
        # Re-raise? or just return?
        return # Stop if standardized processing fails
        
    # --- Write Segment Counts File (if requested and data available) ---
    if segment_counts_output_path:
        if question_segment_counts:
            logging.info(f"Writing segment counts per question ({len(question_segment_counts)} questions) to: {segment_counts_output_path}")
            try:
                with open(segment_counts_output_path, 'w', encoding='utf-8', newline='') as outfile_counts:
                    # Use Int64 dtype for pandas NA compatibility if needed later
                    writer_counts = csv.DictWriter(outfile_counts, fieldnames=segment_counts_header)
                    writer_counts.writeheader()
                    rows_written_counts += 1
                    for qid, count_data in question_segment_counts.items():
                         # Ensure all expected segment columns are present, fill missing with empty string for CSV
                         row_to_write = {col: count_data.get(col, '') for col in segment_counts_header}
                         writer_counts.writerow(row_to_write)
                         rows_written_counts += 1
                logging.info(f"  Segment counts file saved successfully ({rows_written_counts} rows incl. header).")
            except Exception as e:
                 logging.error(f"Error writing segment counts file {segment_counts_output_path}: {e}", exc_info=True)
                 # Clean up partially written counts file
                 if os.path.exists(segment_counts_output_path):
                     try: os.remove(segment_counts_output_path); logging.warning(f"Removed partially written segment counts file: {segment_counts_output_path}")
                     except OSError as remove_err: logging.error(f"Failed to remove partially written file {segment_counts_output_path}: {remove_err}")
        else:
            logging.warning(f"Segment counts output requested, but no segment count data was collected (check QID extraction or segment details). File not written: {segment_counts_output_path}")

def main():
    parser = argparse.ArgumentParser(description='Standardize an aggregate CSV file and optionally extract segment counts per question.')
    
    # Group for mutually exclusive input specification
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--gd_number", type=int, help="Global Dialogue cadence number (e.g., 1, 2, 3). Constructs default paths like Data/GD<N>/GD<N>_*.csv")
    input_group.add_argument("--input_file", help="Explicit path to the input aggregate CSV file (required if --gd_number is not used).")

    # Output paths - conditionally required
    parser.add_argument('--output_file', help='Path to save the standardized output CSV file (required if --gd_number is not used).')
    parser.add_argument('--segment_counts_output', help='(Optional) Path to save the segment counts per question CSV file. Default constructed if --gd_number is used.')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Determine File Paths --- 
    input_path = None
    output_path = None
    counts_output_path = None

    if args.gd_number:
        gd_num = args.gd_number
        gd_identifier = f"GD{gd_num}"
        base_dir = os.path.join("Data", gd_identifier)
        input_path = os.path.join(base_dir, f"{gd_identifier}_aggregate.csv")
        output_path = os.path.join(base_dir, f"{gd_identifier}_aggregate_standardized.csv")
        # Use provided counts path if given, otherwise construct default
        counts_output_path = args.segment_counts_output if args.segment_counts_output else os.path.join(base_dir, f"{gd_identifier}_segment_counts_by_question.csv")
        
        logging.info(f"Using GD number {gd_num} to determine paths:")
        logging.info(f"  Input: {input_path}")
        logging.info(f"  Output (Standardized): {output_path}")
        logging.info(f"  Output (Segment Counts): {counts_output_path}")

        # Validate constructed input path
        if not os.path.exists(input_path):
            logging.error(f"Constructed input file path does not exist: {input_path}")
            parser.error(f"Input file not found for GD{gd_num}. Expected at: {input_path}") # Use parser.error to exit
            
    else: # Explicit paths provided
        if not args.input_file or not args.output_file:
            parser.error("--input_file and --output_file are required when --gd_number is not used.")
        input_path = args.input_file
        output_path = args.output_file
        counts_output_path = args.segment_counts_output # Can be None if not provided
        
        logging.info(f"Using explicitly provided paths:")
        logging.info(f"  Input: {input_path}")
        logging.info(f"  Output (Standardized): {output_path}")
        if counts_output_path:
             logging.info(f"  Output (Segment Counts): {counts_output_path}")
        else:
             logging.info(f"  Output (Segment Counts): Not requested.")


    # --- Basic Filename Validation (optional but helpful) ---
    if not input_path.lower().endswith('.csv'):
        logging.warning("Input file might not be a CSV.")
    if not output_path.lower().endswith('.csv'):
        logging.warning("Standardized output file might not be a CSV.")
    if counts_output_path and not counts_output_path.lower().endswith('.csv'):
        logging.warning("Segment counts output file might not be a CSV.")

    # --- Run Standardization --- 
    standardize_aggregate_csv(input_path, output_path, counts_output_path)

if __name__ == "__main__":
    main() 