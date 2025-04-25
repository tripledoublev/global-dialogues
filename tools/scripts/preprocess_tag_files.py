import os
import pandas as pd
import argparse
import re
import glob
import csv
import traceback # Added for better error reporting

def extract_metadata_and_find_header(file_path):
    """
    Reads the start of a Remesh CSV export to find metadata (like Question ID)
    and the actual header row content and its index.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        tuple: (metadata_dict, header_content, header_row_num) or (None, None, None) if fails.
               metadata_dict contains extracted key-value pairs.
               header_content is the list of strings in the header row.
               header_row_num is the 0-based index of the data header row.
    """
    metadata = {}
    header_content = None
    header_row_num = None
    expected_headers_cat_keys = ["category", "tag"]
    expected_headers_labels_keys = ["participant id", "sentiment"]
    optional_response_keys = ["response", "thought text"]

    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                # Check for metadata rows (Key, Value) - usually before the header
                if len(row) == 2 and row[0] and row[1] and header_content is None:
                    key = row[0].strip()
                    value = row[1].strip()
                    metadata[key] = value
                # Check for potential data header
                elif len(row) > 1:
                    row_lower_stripped = [str(h).lower().strip() for h in row]
                    
                    # Check Categories header
                    is_cat_header = all(eh in row_lower_stripped for eh in expected_headers_cat_keys)
                    
                    # Check Labels header (updated logic)
                    has_required_label_keys = all(eh in row_lower_stripped for eh in expected_headers_labels_keys)
                    has_optional_response_key = any(resp_key in row_lower_stripped for resp_key in optional_response_keys)
                    is_labels_header = has_required_label_keys and has_optional_response_key

                    if is_cat_header or is_labels_header:
                        header_row_num = i
                        header_content = [str(h).strip() for h in row]
                        # print(f"Debug: Found header at row {i}: {header_content}")
                        # Stop searching once header is found
                        break 
                
                # Optimization: Stop searching for header after too many rows
                if i > 30 and header_content is None: 
                    print(f"Warning: Could not definitively locate data header in {file_path} within first 30 lines.")
                    break

        # Validation after reading whole file (or stopping early)
        if "Question IDs" not in metadata:
            print(f"Warning: 'Question IDs' not found in metadata for {file_path}. Cannot process.")
            return None, None, None
        if header_content is None:
             print(f"Warning: Data header row not found for {file_path}. Cannot process.")
             return None, None, None

        return metadata, header_content, header_row_num

    except Exception as e:
        print(f"Error reading or parsing metadata/header for {file_path}: {e}")
        return None, None, None

def process_raw_file(file_path, output_dir):
    """
    Processes a single raw Remesh tag CSV file using manual CSV reading,
    saves a cleaned version.
    Returns the Question ID and file type if successful.
    """
    print(f"Processing raw file: {os.path.basename(file_path)}")
    metadata, header, header_idx = extract_metadata_and_find_header(file_path)

    if metadata is None or header is None or header_idx is None:
        return None, None

    # --- QID Validation --- (same as before)
    qid = metadata.get("Question IDs", "").strip()
    if not qid: print(f"Warning: No Question ID extracted..."); return None, None
    if not re.match(r'^[0-9a-fA-F]{8}-([0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}$', qid):
         print(f"Warning: Extracted Question ID '{qid}' invalid format..."); return None, None

    # --- Manual Data Reading --- 
    data_rows = []
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i > header_idx: # Only read rows after the header
                    # Ensure row has at least as many columns as header 
                    # (Pad shorter rows, truncate longer ones to match header length)
                    # This helps handle rows with extra/fewer commas potentially
                    row_len = len(row)
                    header_len = len(header)
                    if row_len < header_len:
                        row.extend([None] * (header_len - row_len)) # Pad with None
                    elif row_len > header_len:
                        print(f"  Warning: Row {i+1} in {os.path.basename(file_path)} has {row_len} fields (expected {header_len}). Truncating.")
                        row = row[:header_len] # Truncate
                    data_rows.append(row)
    except Exception as e:
        print(f"Error reading data rows for {file_path}: {e}")
        return None, None
        
    if not data_rows:
        print(f"Warning: No data rows found after header in {file_path}. Skipping.")
        return None, None
        
    # --- Create DataFrame and Clean --- 
    try:
        df = pd.DataFrame(data_rows, columns=header)
        # print(f"Debug: Initial columns: {df.columns.tolist()}")
        df.columns = [str(c).strip() for c in df.columns] # Clean header whitespace first

        # --- Rename Duplicate/Generic Tag Columns ---
        # Find indices of columns starting with 'Tag' (case-insensitive)
        tag_col_indices = [i for i, col in enumerate(df.columns) if str(col).lower().strip().startswith('tag')]
        
        if tag_col_indices:
            new_columns = list(df.columns)
            tag_counter = 1
            for i in tag_col_indices:
                new_columns[i] = f"Tag {tag_counter}" # Rename to "Tag 1", "Tag 2", ...
                tag_counter += 1
            df.columns = new_columns
            # print(f"Debug: Renamed tag columns: {df.columns.tolist()}")

        file_type = None
        cleaned_df = None
        output_filename = None

        # Identify file type and clean (USING NEW, UNIQUE COLUMN NAMES)
        # Check now uses 'Tag ' prefix to find the renamed columns
        if "Category" in df.columns and any(col.startswith("Tag ") for col in df.columns):
            file_type = "categories"
            tag_cols = [col for col in df.columns if col.startswith("Tag ")]
            # Sort tags numerically based on the number after "Tag "
            tag_cols.sort(key=lambda x: int(x.split(' ')[1])) 
            
            keep_cols = ["Category"] + tag_cols
            keep_cols = [col for col in keep_cols if col in df.columns] # Ensure they exist
            
            if "Category" not in keep_cols: 
                 print(f"Warning: 'Category' column missing after processing {file_path}. Skipping.")
                 return None, None
                 
            cleaned_df = df[keep_cols].copy()
            cleaned_df.dropna(subset=["Category"], how='all', inplace=True)
            output_filename = f"{qid}_tag_categories.csv"

        elif "Participant ID" in df.columns and \
             "Sentiment" in df.columns and \
             (any(c in df.columns for c in ["Response", "Thought Text"])): # Check for original response cols

            # Standardize response column name to "ResponseText" *before* selecting columns
            response_col_name_in_df = None
            if "Response" in df.columns:
                response_col_name_in_df = "Response"
            elif "Thought Text" in df.columns:
                response_col_name_in_df = "Thought Text"
                 
            if response_col_name_in_df is None:
                 # This case should ideally not happen if the elif condition passed, but safety check
                 print(f"Warning: Missing response column ('Response' or 'Thought Text') in {file_path}. Skipping.")
                 return None, None
                 
            # Rename the identified response column to 'ResponseText'
            df.rename(columns={response_col_name_in_df: "ResponseText"}, inplace=True)
            # print(f"Debug: Columns after rename: {df.columns.tolist()}")

            file_type = "labels"
            # Find and sort tag columns numerically
            tag_cols = [col for col in df.columns if col.startswith("Tag ")] 
            tag_cols.sort(key=lambda x: int(x.split(' ')[1]))
            
            # Define columns to keep, including the standardized 'ResponseText'
            keep_cols = ["Participant ID", "ResponseText", "Sentiment"] + tag_cols
            if "Thought ID" in df.columns: # Include Thought ID if present
                keep_cols.insert(1, "Thought ID") 
            
            # Ensure all required columns actually exist in the dataframe after potential renaming/filtering
            keep_cols = [col for col in keep_cols if col in df.columns] 
            
            essential_cols = ["Participant ID", "ResponseText", "Sentiment"]
            if not all(ec in keep_cols for ec in essential_cols):
                 print(f"Warning: Missing essential label columns after processing {file_path} (Needed: {essential_cols}, Found: {keep_cols}). Skipping.")
                 return None, None
                 
            cleaned_df = df[keep_cols].copy()

            primary_id_col = "Thought ID" if "Thought ID" in cleaned_df.columns else "Participant ID"
            cleaned_df.dropna(subset=[primary_id_col], how='all', inplace=True)
            output_filename = f"{qid}_thought_labels.csv"

        else:
            # Improve warning message
            cols_present = list(df.columns)
            print(f"Warning: Could not determine file type for {file_path}. Required columns not found.")
            print(f"  - For Categories: Need 'Category' and 'Tag ...'. Found: {cols_present}")
            print(f"  - For Labels: Need 'Participant ID', 'Sentiment', ('Response' or 'Thought Text'), and 'Tag ...'. Found: {cols_present}")
            return None, None

        # Save cleaned individual file
        output_path = os.path.join(output_dir, output_filename)
        cleaned_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  Saved cleaned data to: {output_path}")
        return qid, file_type

    except Exception as e:
        print(f"Error creating/processing DataFrame for {file_path}: {e}")
        traceback.print_exc() # Print detailed error traceback
        return None, None


def rebuild_combined_files(output_dir):
    """
    Scans the output directory for processed individual files and rebuilds
    the combined all_tag_categories.csv and all_thought_labels.csv files.
    """
    print("\nRebuilding combined files...")

    # --- Rebuild Labels ---
    all_labels_dfs = []
    label_files = glob.glob(os.path.join(output_dir, '*_thought_labels.csv'))
    print(f"\nFound {len(label_files)} processed label files.")
    for file_path in label_files:
        filename = os.path.basename(file_path)
        # Skip the combined file itself
        if filename == 'all_thought_labels.csv': continue 
        match = re.match(r'([0-9a-fA-F\-]+)_thought_labels\.csv', filename)
        if match:
            qid = match.group(1)
            try:
                df = pd.read_csv(file_path)
                df["Question ID"] = qid
                # Basic check for essential columns before appending
                # Assuming individual files now have correct 'ResponseText' and 'Tag N' columns
                essential_cols = ["Question ID", "Participant ID", "ResponseText", "Sentiment"]
                if not all(ec in df.columns for ec in essential_cols[:2]): # Check QID, PartID
                    print(f"Warning: Skipping {filename} during rebuild - missing Question ID or Participant ID.")
                    continue
                if "ResponseText" not in df.columns: # Check ResponseText explicitly
                     print(f"Warning: Skipping {filename} during rebuild - missing ResponseText column.")
                     continue
                if "Sentiment" not in df.columns: # Check Sentiment explicitly
                     print(f"Warning: Skipping {filename} during rebuild - missing Sentiment column.")
                     continue

                all_labels_dfs.append(df) # Append the whole df, selection happens after concat
            except Exception as e: print(f"  Error reading processed label file {file_path}: {e}")
        else: print(f"  Warning: Could not extract QID from label filename {filename}")

    if all_labels_dfs:
        # --- Standardize Columns Across All Label DFs ---
        # Find max tag number N across all dataframes based on "Tag N" format
        max_tag_num = 0
        for df_inner in all_labels_dfs:
            tag_cols = [col for col in df_inner.columns if col.startswith('Tag ')]
            if tag_cols:
                try:
                    # Extract numbers and find the maximum N
                    max_n_in_df = max(int(col.split(' ')[1]) for col in tag_cols)
                    max_tag_num = max(max_tag_num, max_n_in_df)
                except (ValueError, IndexError):
                    print(f"  Warning: Could not parse tag number from columns {tag_cols} in one of the label files. Skipping for max tag calculation.")


        standard_tag_cols = [f'Tag {i+1}' for i in range(max_tag_num)] # e.g., ['Tag 1', 'Tag 2', ..., 'Tag N']

        processed_labels_dfs = []
        for df_inner in all_labels_dfs:
             existing_cols = df_inner.columns.tolist()
             # Identify standard tag columns missing from this df
             cols_to_add = [t for t in standard_tag_cols if t not in existing_cols]
             # Reindex *only if necessary*, adding missing tags as NA
             if cols_to_add:
                 new_df = df_inner.reindex(columns=existing_cols + cols_to_add, fill_value=pd.NA)
             else:
                 new_df = df_inner # No tags to add
             processed_labels_dfs.append(new_df)

        if processed_labels_dfs:
            # Concatenate all potentially reindexed dataframes
            combined_labels_df = pd.concat(processed_labels_dfs, ignore_index=True)

            # --- Define Final Column Order ---
            id_cols = ["Question ID"]
            # Check for Participant ID and Thought ID *after* concat
            if "Participant ID" in combined_labels_df.columns: id_cols.append("Participant ID")
            if "Thought ID" in combined_labels_df.columns: id_cols.append("Thought ID")

            text_sentiment_cols = []
            if "ResponseText" in combined_labels_df.columns: text_sentiment_cols.append("ResponseText")
            if "Sentiment" in combined_labels_df.columns: text_sentiment_cols.append("Sentiment")

            # Get all tag columns present after concat and standardization
            tag_cols = [col for col in combined_labels_df.columns if col.startswith("Tag ")]
            # Sort tags NUMERICALLY based on the number N
            tag_cols.sort(key=lambda x: int(x.split(' ')[1]))

            # Construct final list, ensuring columns actually exist
            final_cols = id_cols + text_sentiment_cols + tag_cols
            final_cols = [col for col in final_cols if col in combined_labels_df.columns]

            combined_labels_df = combined_labels_df[final_cols] # Reorder columns
            output_path = os.path.join(output_dir, 'all_thought_labels.csv')
            try:
                combined_labels_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"Saved combined labels file: {output_path}")
            except Exception as e: print(f"  Error saving combined labels file: {e}")
        else: print("Processed labels list was empty after standardization or initial read.")
    else: print("No processed label files found to combine.")


    # --- Rebuild Categories ---
    all_categories_dfs = []
    category_files = glob.glob(os.path.join(output_dir, '*_tag_categories.csv'))
    print(f"\nFound {len(category_files)} processed category files.")
    for file_path in category_files:
        filename = os.path.basename(file_path)
        # Skip the combined file itself
        if filename == 'all_tag_categories.csv': continue 
        match = re.match(r'([0-9a-fA-F\-]+)_tag_categories\.csv', filename)
        if match:
            qid = match.group(1)
            try:
                df = pd.read_csv(file_path)
                df["Question ID"] = qid
                # Basic check
                if "Category" not in df.columns:
                    print(f"Warning: Skipping {filename} during rebuild - missing Category column.")
                    continue
                all_categories_dfs.append(df)
            except Exception as e: print(f"  Error reading processed category file {file_path}: {e}")
        else: print(f"  Warning: Could not extract QID from category filename {filename}")

    if all_categories_dfs:
        # --- Standardize Columns Across All Category DFs ---
        max_cat_tag_num = 0
        for df_inner in all_categories_dfs:
            tag_cols = [col for col in df_inner.columns if col.startswith('Tag ')]
            if tag_cols:
                try:
                    max_n_in_df = max(int(col.split(' ')[1]) for col in tag_cols)
                    max_cat_tag_num = max(max_cat_tag_num, max_n_in_df)
                except (ValueError, IndexError):
                     print(f"  Warning: Could not parse tag number from columns {tag_cols} in one of the category files. Skipping for max tag calculation.")

        standard_cat_tag_cols = [f'Tag {i+1}' for i in range(max_cat_tag_num)]

        processed_categories_dfs = []
        for df_inner in all_categories_dfs:
             existing_cols = df_inner.columns.tolist()
             cols_to_add = [t for t in standard_cat_tag_cols if t not in existing_cols]
             if cols_to_add:
                 new_df = df_inner.reindex(columns=existing_cols + cols_to_add, fill_value=pd.NA)
             else:
                 new_df = df_inner
             processed_categories_dfs.append(new_df)

        if processed_categories_dfs:
            combined_categories_df = pd.concat(processed_categories_dfs, ignore_index=True)

            # --- Define Final Column Order ---
            id_cols = ["Question ID"]
            cat_col = ["Category"] if "Category" in combined_categories_df.columns else []
            
            # Get tag columns and sort numerically
            tag_cols = [col for col in combined_categories_df.columns if col.startswith("Tag ")]
            tag_cols.sort(key=lambda x: int(x.split(' ')[1])) # Sort numerically

            final_cols = id_cols + cat_col + tag_cols
            final_cols = [col for col in final_cols if col in combined_categories_df.columns] # Ensure they exist

            combined_categories_df = combined_categories_df[final_cols] # Reorder
            output_path = os.path.join(output_dir, 'all_tag_categories.csv')
            try:
                combined_categories_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"Saved combined categories file: {output_path}")
            except Exception as e: print(f"  Error saving combined categories file: {e}")
        else: print("Processed categories list was empty after standardization or initial read.")
    else: print("No processed category files found to combine.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Remesh tag export files.")
    parser.add_argument("--raw_dir", required=True, help="Directory containing raw Remesh *_Tag_Categories.csv and *_Thought_Labels.csv files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save cleaned individual and combined tag files.")
    args = parser.parse_args()
    if not os.path.isdir(args.raw_dir): print(f"Error: Raw directory not found: {args.raw_dir}"); exit(1)
    if not os.path.isdir(args.output_dir): print(f"Output directory not found, creating: {args.output_dir}"); os.makedirs(args.output_dir)

    # --- Process Raw Files --- 
    raw_files = glob.glob(os.path.join(args.raw_dir, '*.csv'))
    print(f"Found {len(raw_files)} CSV files in raw directory.")
    processed_count = 0
    for file_path in raw_files:
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
    raw_dir_basename = os.path.basename(os.path.normpath(args.raw_dir))
    print("Consider adding the raw directory to .gitignore:")
    print(f"Example: echo '{raw_dir_basename}/' >> .gitignore") 