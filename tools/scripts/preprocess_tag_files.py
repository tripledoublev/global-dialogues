import os
import pandas as pd
import argparse
import re
import glob
import csv

def extract_metadata_and_header_row(file_path):
    """
    Reads the start of a Remesh CSV export to find metadata (like Question ID)
    and the row number where the actual data header starts.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: (metadata_dict, header_row_num) or (None, None) if fails.
               metadata_dict contains extracted key-value pairs, including 'Question IDs'.
               header_row_num is the 0-based index of the data header row.
    """
    metadata = {}
    header_row_num = None
    potential_header_start = None
    expected_headers_cat = ["Category", "Tag"]
    expected_headers_labels = ["Thought ID", "Participant ID", "Thought Text", "Sentiment", "Tag"]

    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                # Check for metadata rows (Key, Value)
                if len(row) == 2 and row[0] and row[1]:
                    key = row[0].strip()
                    value = row[1].strip()
                    metadata[key] = value
                # Check for potential data header (look for known column names)
                elif len(row) > 1:
                    row_lower = [str(h).lower().strip() for h in row]
                    # Check if it matches expected headers for either type
                    is_cat_header = any(h in row_lower for h in [eh.lower() for eh in expected_headers_cat])
                    is_labels_header = any(h in row_lower for h in [eh.lower() for eh in expected_headers_labels])

                    if is_cat_header or is_labels_header:
                        header_row_num = i
                        # print(f"Debug: Found header at row {i}: {row}")
                        break # Found the header, stop searching
                
                # Stop searching after a reasonable number of lines if no header found
                if i > 30: 
                    print(f"Warning: Could not definitively locate data header in {file_path} within first 30 lines.")
                    break

        if "Question IDs" not in metadata:
            print(f"Warning: 'Question IDs' not found in metadata for {file_path}. Cannot process.")
            return None, None
        if header_row_num is None:
             print(f"Warning: Data header row not found for {file_path}. Cannot process.")
             return None, None

        return metadata, header_row_num

    except Exception as e:
        print(f"Error reading or parsing metadata/header for {file_path}: {e}")
        return None, None

def process_raw_file(file_path, output_dir):
    """
    Processes a single raw Remesh tag CSV file, saves a cleaned version.
    Returns the Question ID and file type if successful.
    """
    print(f"Processing raw file: {os.path.basename(file_path)}")
    metadata, header_row = extract_metadata_and_header_row(file_path)

    if metadata is None or header_row is None:
        return None, None

    # Assuming only one QID for simplicity, might need refinement if multiple possible
    qid = metadata.get("Question IDs", "").strip()
    if not qid:
        print(f"Warning: No Question ID extracted from metadata in {file_path}")
        return None, None
    # Basic check for valid QID format (UUID-like) - adjust if needed
    if not re.match(r'^[0-9a-fA-F]{8}-([0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}$', qid):
         print(f"Warning: Extracted Question ID '{qid}' from {file_path} does not look like a standard UUID. Skipping.")
         # You might want to handle non-UUID IDs differently if they are expected
         return None, None


    file_type = None
    cleaned_df = None
    output_filename = None

    try:
        df = pd.read_csv(file_path, header=header_row, encoding='utf-8-sig')
        df.columns = [str(c).strip() for c in df.columns] # Clean header whitespace

        # Identify file type and clean
        if "Category" in df.columns and "Tag" in df.columns:
            file_type = "categories"
            tag_cols = [col for col in df.columns if col.startswith("Tag")]
            keep_cols = ["Category"] + tag_cols
            cleaned_df = df[keep_cols].copy()
            # Drop rows where Category is completely empty/NA
            cleaned_df.dropna(subset=["Category"], how='all', inplace=True)
            output_filename = f"{qid}_tag_categories.csv"

        elif "Thought ID" in df.columns and "Sentiment" in df.columns:
            file_type = "labels"
            tag_cols = [col for col in df.columns if col.startswith("Tag")]
            keep_cols = ["Thought ID", "Participant ID", "Thought Text", "Sentiment"] + tag_cols
            # Ensure all required columns exist before selecting
            missing_cols = [col for col in keep_cols if col not in df.columns]
            if missing_cols:
                 print(f"Warning: Missing expected columns {missing_cols} in label file {file_path}. Skipping.")
                 return None, None
            cleaned_df = df[keep_cols].copy()
            # Drop rows where Thought ID is completely empty/NA
            cleaned_df.dropna(subset=["Thought ID"], how='all', inplace=True)
            output_filename = f"{qid}_thought_labels.csv"

        else:
            print(f"Warning: Could not determine file type (Categories or Labels) for {file_path} based on columns. Skipping.")
            return None, None

        # Save cleaned individual file
        output_path = os.path.join(output_dir, output_filename)
        cleaned_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  Saved cleaned data to: {output_path}")
        return qid, file_type

    except Exception as e:
        print(f"Error processing DataFrame for {file_path}: {e}")
        return None, None


def rebuild_combined_files(output_dir):
    """
    Scans the output directory for processed individual files and rebuilds
    the combined all_tag_categories.csv and all_thought_labels.csv files.
    """
    print("\nRebuilding combined files...")

    # --- Rebuild Categories ---
    all_categories_dfs = []
    category_files = glob.glob(os.path.join(output_dir, '*_tag_categories.csv'))
    print(f"Found {len(category_files)} processed category files.")
    for file_path in category_files:
        filename = os.path.basename(file_path)
        # Extract QID from filename
        match = re.match(r'([0-9a-fA-F\-]+)_tag_categories\.csv', filename)
        if match:
            qid = match.group(1)
            try:
                df = pd.read_csv(file_path)
                df["Question ID"] = qid # Add Question ID column
                # Reorder columns to put Question ID first
                cols = ["Question ID"] + [col for col in df.columns if col != "Question ID"]
                all_categories_dfs.append(df[cols])
            except Exception as e:
                print(f"  Error reading processed category file {file_path}: {e}")
        else:
             print(f"  Warning: Could not extract QID from filename {filename}")

    if all_categories_dfs:
        combined_categories_df = pd.concat(all_categories_dfs, ignore_index=True)
        output_path = os.path.join(output_dir, 'all_tag_categories.csv')
        try:
            combined_categories_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Saved combined categories file: {output_path}")
        except Exception as e:
            print(f"  Error saving combined categories file: {e}")
    else:
        print("No processed category files found to combine.")

    # --- Rebuild Labels ---
    all_labels_dfs = []
    label_files = glob.glob(os.path.join(output_dir, '*_thought_labels.csv'))
    print(f"Found {len(label_files)} processed label files.")
    for file_path in label_files:
        filename = os.path.basename(file_path)
        match = re.match(r'([0-9a-fA-F\-]+)_thought_labels\.csv', filename)
        if match:
            qid = match.group(1)
            try:
                df = pd.read_csv(file_path)
                df["Question ID"] = qid
                # Reorder columns
                id_cols = ["Question ID", "Thought ID", "Participant ID"]
                text_sentiment_cols = ["Thought Text", "Sentiment"]
                tag_cols = [col for col in df.columns if col.startswith("Tag")]
                # Filter out any cols not in the expected set
                final_cols = id_cols + text_sentiment_cols + tag_cols
                final_cols = [col for col in final_cols if col in df.columns] 
                all_labels_dfs.append(df[final_cols])
            except Exception as e:
                print(f"  Error reading processed label file {file_path}: {e}")
        else:
             print(f"  Warning: Could not extract QID from filename {filename}")


    if all_labels_dfs:
        # Find max number of tag columns across all files for clean concat
        max_tags = 0
        for df in all_labels_dfs:
             max_tags = max(max_tags, len([col for col in df.columns if col.startswith('Tag')]))
        
        # Ensure all dataframes have the same tag columns (filling missing with NaN)
        standard_tag_cols = [f'Tag {i+1}' for i in range(max_tags)]
        processed_labels_dfs = []
        for df in all_labels_dfs:
             current_tags = [col for col in df.columns if col.startswith('Tag')]
             new_df = df.reindex(columns=df.columns.tolist() + [t for t in standard_tag_cols if t not in current_tags], fill_value=pd.NA)
             processed_labels_dfs.append(new_df)

        if processed_labels_dfs:
            combined_labels_df = pd.concat(processed_labels_dfs, ignore_index=True)
            # Final reorder just in case
            id_cols = ["Question ID", "Thought ID", "Participant ID"]
            text_sentiment_cols = ["Thought Text", "Sentiment"]
            tag_cols = [col for col in combined_labels_df.columns if col.startswith("Tag")]
            final_cols = id_cols + text_sentiment_cols + sorted(tag_cols) # Sort tag columns for consistency
            final_cols = [col for col in final_cols if col in combined_labels_df.columns]
            combined_labels_df = combined_labels_df[final_cols]

            output_path = os.path.join(output_dir, 'all_thought_labels.csv')
            try:
                combined_labels_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"Saved combined labels file: {output_path}")
            except Exception as e:
                print(f"  Error saving combined labels file: {e}")
        else:
             print("Processed labels list became empty after standardization.")
             
    else:
        print("No processed label files found to combine.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Remesh tag export files.")
    parser.add_argument("--raw_dir", required=True, help="Directory containing raw Remesh *_Tag_Categories.csv and *_Thought_Labels.csv files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save cleaned individual and combined tag files.")

    args = parser.parse_args()

    # Validate directories
    if not os.path.isdir(args.raw_dir):
        print(f"Error: Raw directory not found: {args.raw_dir}")
        exit(1)
    if not os.path.isdir(args.output_dir):
        print(f"Output directory not found, creating: {args.output_dir}")
        try:
            os.makedirs(args.output_dir)
        except OSError as e:
            print(f"Error creating output directory {args.output_dir}: {e}")
            exit(1)

    # --- Process Raw Files ---
    raw_files = glob.glob(os.path.join(args.raw_dir, '*.csv'))
    print(f"Found {len(raw_files)} CSV files in raw directory.")
    processed_count = 0
    for file_path in raw_files:
        # Basic check for expected filenames before processing
        basename = os.path.basename(file_path)
        if "_Tag_Categories" in basename or "_Thought_Labels" in basename:
            qid, file_type = process_raw_file(file_path, args.output_dir)
            if qid and file_type:
                processed_count += 1
        else:
            print(f"Skipping file with unexpected name format: {basename}")
            
    print(f"\nProcessed {processed_count} raw files.")

    # --- Rebuild Combined Files ---
    rebuild_combined_files(args.output_dir)

    print("\nPreprocessing complete.")
    print(f"Cleaned files saved in: {args.output_dir}")
    print("Consider adding the raw directory to .gitignore:")
    print(f"Example: echo '{os.path.basename(args.raw_dir)}/' >> .gitignore") 