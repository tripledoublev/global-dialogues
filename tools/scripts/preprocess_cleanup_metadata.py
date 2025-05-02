import argparse
import logging
from pathlib import Path
import pandas as pd
import csv # For more granular reading if needed
import shutil # For replacing the file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_header_row(file_path, expected_markers):
    """
    Attempts to find the 0-based index of the header row in a CSV file.

    Args:
        file_path (Path): Path to the CSV file.
        expected_markers (list): A list of column names expected to be in the header.

    Returns:
        int: The 0-based index of the header row, or -1 if not found or error occurs.
    """
    logging.debug(f"Attempting to find header in {file_path} using markers: {expected_markers}")
    # Try common encodings
    encodings_to_try = ['utf-8-sig', 'utf-8', 'latin1', 'iso-8859-1']
    detected_encoding = None
    header_row_index = -1

    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                # Use csv reader for more control over quoting/delimiters if needed
                # For now, simple line reading is okay
                for i, line in enumerate(f):
                    line_content = line.strip()
                    # Skip empty lines
                    if not line_content:
                        continue
                    # Basic checks: contains commas, contains markers
                    if ',' in line_content:
                        # Check if all markers are present (case-insensitive)
                        line_lower = line_content.lower()
                        if all(marker.lower() in line_lower for marker in expected_markers):
                            # Perform a more robust check with pandas sniffing
                            try:
                                # Try reading just this line as header to validate columns
                                df_header_test = pd.read_csv(file_path, header=i, nrows=0, encoding=enc)
                                df_header_test.columns = df_header_test.columns.str.replace('^\\ufeff', '', regex=True).str.strip() # Clean potential BOM
                                if all(marker in df_header_test.columns for marker in expected_markers):
                                    logging.info(f"Confirmed header on line {i+1} (index {i}) in {file_path} using encoding '{enc}'. Markers: {expected_markers}")
                                    header_row_index = i
                                    detected_encoding = enc
                                    break # Found header
                                else:
                                     logging.debug(f"Line {i+1} contained markers but pandas validation failed. Found columns: {df_header_test.columns.tolist()}")

                            except Exception as pd_err:
                                logging.debug(f"Pandas validation failed for line {i+1} in {file_path} with encoding {enc}: {pd_err}")
                                continue # Try next line
                        # else: # Optional: log lines that have commas but not all markers
                            # logging.debug(f"Line {i+1} did not contain all markers '{expected_markers}'.")
            if header_row_index != -1:
                break # Stop trying encodings if header found
        except UnicodeDecodeError:
            logging.warning(f"Encoding {enc} failed for {file_path}, trying next...")
            continue
        except Exception as file_read_error:
            logging.error(f"Unexpected error reading {file_path} with encoding {enc}: {file_read_error}")
            return -1 # Indicate error

    if header_row_index == -1:
        logging.warning(f"Could not reliably find header row in {file_path} using markers {expected_markers}.")

    return header_row_index

def clean_csv_metadata(file_path, header_index):
    """
    Reads a CSV starting from the header row and overwrites the original file
    using basic file I/O to avoid pandas parsing errors during cleanup.

    Args:
        file_path (Path): Path to the CSV file.
        header_index (int): 0-based index of the header row.

    Returns:
        bool: True if successful, False otherwise.
    """
    logging.info(f"Cleaning metadata from {file_path} (header found at index {header_index}) using basic I/O...")
    temp_file_path = file_path.with_suffix(file_path.suffix + '.tmp')

    try:
        # Determine the original encoding if possible (using find_header_row logic again briefly)
        # This is slightly redundant but ensures we read correctly
        original_encoding = 'utf-8-sig' # Default
        encodings_to_try = ['utf-8-sig', 'utf-8', 'latin1', 'iso-8859-1']
        for enc in encodings_to_try:
            try:
                 with open(file_path, 'r', encoding=enc) as f_check:
                     f_check.readline() # Try reading one line
                 original_encoding = enc
                 logging.debug(f"Determined encoding '{original_encoding}' for reading {file_path}")
                 break
            except UnicodeDecodeError:
                 continue
            except Exception:
                 pass # Ignore other errors during encoding check

        with open(file_path, 'r', encoding=original_encoding) as infile, \
             open(temp_file_path, 'w', encoding='utf-8') as outfile: # Write standard UTF-8

            # Skip lines before the header
            for _ in range(header_index):
                next(infile)

            # Read header and remaining lines, write to temp file
            shutil.copyfileobj(infile, outfile)

        # Replace original file with the temp file
        shutil.move(str(temp_file_path), str(file_path))
        logging.info(f"Successfully cleaned and overwrote {file_path}.")
        return True

    except Exception as e:
        logging.error(f"Failed to clean or overwrite {file_path} using basic I/O: {e}", exc_info=True)
        # Clean up temp file if it exists
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except OSError:
                logging.error(f"Could not remove temporary file: {temp_file_path}")
        return False
    finally:
         # Ensure temp file is removed in case of unexpected exit before move
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
                logging.debug(f"Removed leftover temp file: {temp_file_path}")
            except OSError:
                 # Log if removal fails but don't stop the process
                 logging.error(f"Could not remove temporary file during final cleanup: {temp_file_path}")

def main(args):
    gd_number = args.gd_number
    data_dir = Path("./Data") / f"GD{gd_number}"

    if not data_dir.is_dir():
        logging.error(f"Data directory not found: {data_dir}")
        return

    logging.info(f"Starting metadata cleanup for GD{gd_number} in {data_dir}")

    # Define files to check and their key header markers
    # Exclude aggregate.csv (handled by preprocess_aggregate.py)
    # Exclude *_standardized.csv, *_report.csv etc.
    files_to_clean = {
        f"GD{gd_number}_participants.csv": ['Participant Id', 'Sample Provider Id'],
        f"GD{gd_number}_discussion_guide.csv": ['Item type (dropdown)', 'Content'],
        f"GD{gd_number}_binary.csv": ['Question ID', 'Participant ID', 'Thought ID', 'Vote'],
        f"GD{gd_number}_preference.csv": ['Question ID', 'Participant ID', 'Thought A ID', 'Thought B ID', 'Vote'],
        f"GD{gd_number}_verbatim_map.csv": ['Question ID', 'Question Text', 'Participant ID', 'Thought ID', 'Thought Text'],
        f"GD{gd_number}_summary.csv": ['Conversation ID', 'Conversation Title', 'Questions Selected'],
        # Add other raw Remesh files if necessary
    }

    # Hardcoded header indices for known problematic files (0-based)
    hardcoded_headers = {
        f"GD{gd_number}_participants.csv": 11,
        f"GD{gd_number}_discussion_guide.csv": 12,
        # f"GD{gd_number}_summary.csv": 7 # Removed - rely on marker detection
    }

    for filename, markers in files_to_clean.items():
        file_path = data_dir / filename
        if file_path.exists():
            logging.info(f"Processing file: {file_path}")
            header_idx = -1
            needs_cleaning = False

            # 1. Try marker-based detection first
            detected_idx = find_header_row(file_path, markers)

            if detected_idx == 0:
                logging.info(f"Header already at line 1 (index 0) for {filename}. No cleanup needed.")
                header_idx = 0 # Mark as found at 0
            elif detected_idx > 0:
                logging.info(f"Header found at index {detected_idx} using markers. Scheduling cleanup.")
                header_idx = detected_idx
                needs_cleaning = True
            else: # detected_idx == -1 (markers didn't find it)
                logging.warning(f"Marker-based detection failed for {filename}.")
                # 2. Check if it's a file with a hardcoded index
                if filename in hardcoded_headers:
                    logging.info(f"Checking hardcoded index for {filename}.")
                    # 2a. Check if header is *already* at line 1 (index 0) even though markers failed
                    # We need a quick check here without full find_header_row logic
                    try:
                        df_check = pd.read_csv(file_path, nrows=1, encoding='utf-8-sig') # Read first line + header
                        df_check.columns = df_check.columns.str.replace('^\\ufeff', '', regex=True).str.strip()
                        if all(marker in df_check.columns for marker in markers):
                            logging.info(f"Header check confirms header is already at line 1 for {filename} (hardcoded case). No cleanup needed.")
                            header_idx = 0 # Mark as found at 0
                        else:
                            logging.warning(f"Header not at line 1 for {filename}. Using hardcoded index.")
                            header_idx = hardcoded_headers[filename]
                            needs_cleaning = True
                    except Exception as e:
                        logging.error(f"Error checking line 1 for {filename} before using hardcoded index: {e}")
                        # Fallback to assuming it needs cleaning with hardcoded index
                        logging.warning(f"Falling back to hardcoded index {hardcoded_headers[filename]} for {filename}.")
                        header_idx = hardcoded_headers[filename]
                        needs_cleaning = True
                else:
                    # Markers failed, and no hardcoded index available
                    logging.error(f"Could not find header for {filename} using markers and no hardcoded index provided. Skipping cleanup.")
                    # Keep header_idx = -1

            # 3. Perform cleanup if needed and possible
            if needs_cleaning and header_idx > 0:
                clean_csv_metadata(file_path, header_idx)
            elif not needs_cleaning and header_idx == 0:
                pass # Already clean, do nothing
            elif header_idx == -1:
                logging.error(f"Final decision: Skipping cleanup for {filename} as header could not be reliably located.")

        else:
            logging.warning(f"File not found, skipping: {file_path}")

    logging.info(f"Metadata cleanup for GD{gd_number} completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove initial metadata rows from raw Remesh CSV files, making the header the first line.")
    parser.add_argument("gd_number", type=int, help="Global Dialogue cadence number (e.g., 3).")
    args = parser.parse_args()
    main(args) 