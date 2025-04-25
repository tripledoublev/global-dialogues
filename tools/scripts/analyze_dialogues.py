import pandas as pd
import numpy as np
import csv
import os
import pickle # Using pickle for simplicity for saving list of DFs
import argparse
import math
import re # For parsing segment columns
import warnings
import matplotlib.pyplot as plt
import seaborn as sns # Optional: for nicer plots
import textwrap # Import textwrap

# --- Configuration ---
# Default values, can be overridden by command-line args
DEFAULT_MIN_SEGMENT_SIZE = 30
CACHE_FILENAME = "processed_data.pkl"
# Set pandas display options for potentially wide DataFrames
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 150) # Adjust display width for reports

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

def longest_common_suffix(strings):
    """Calculates the longest common suffix of a list of strings."""
    if not strings:
        return ""
    reversed_strings = [s[::-1] for s in strings]
    # Find common prefix of reversed strings
    lcp_reversed = os.path.commonprefix(reversed_strings)
    # Reverse back to get common suffix
    return lcp_reversed[::-1]

def get_segment_columns(df_columns):
    """
    Identifies segment columns (typically ending in '(Number)'), extracts the
    segment name, optional 'O' code (e.g., O1), and the participant count (N).

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
    pattern = re.compile(r'^(O\d+:\s*)?(.*?)\s*\(\s*(\d+)\s*\)\s*$')

    for col in df_columns:
        match = pattern.match(col)
        if match:
            segment_cols.append(col)
            o_code_match = match.group(1) # Might be None if no 'O' prefix
            name = match.group(2).strip() # The captured segment name
            size_str = match.group(3) # The captured digits for size

            # Extract just the 'O' code number if present
            o_code = None
            if o_code_match:
                 o_code_num_match = re.search(r'O(\d+)', o_code_match)
                 if o_code_num_match:
                     o_code = f"O{o_code_num_match.group(1)}" # Store as O1, O2 etc.

            try:
                size = int(size_str)
            except ValueError:
                size = np.nan
                print(f"Warning: Could not parse extracted size digits '{size_str}' from segment column: {col}")

            segment_details[col] = {'name': name, 'o_code': o_code, 'size': size}

    # Sort to bring 'All(...)' column(s) to the front if they exist
    # Note: segment_details dict remains unsorted, sorting only affects segment_cols list
    all_cols = [col for col in segment_cols if col.startswith('All(') or col.startswith('All (')]
    other_cols = [col for col in segment_cols if not (col.startswith('All(') or col.startswith('All ('))]
    sorted_segment_cols = sorted(all_cols) + sorted(other_cols)

    if not sorted_segment_cols and len(df_columns) > 5: # Adjusted threshold slightly
         # Check standard non-segment columns first before warning
         standard_cols = {'Question ID', 'Question Type', 'Question', 'Responses', 'English Response', 'Original Response'}
         potential_segments = [c for c in df_columns if c not in standard_cols and '(' in c]
         if potential_segments:
              print(f"Warning: Regex pattern did not find segment columns matching format 'Name (Number)' or 'O#: Name (Number)' in headers: {df_columns[:10]}... Check CSV format or regex pattern.")
         # Else: Likely a question type without segments (e.g. Ask Experience), so no warning needed.

    # Return both the sorted list of column names and the detailed dictionary
    return sorted_segment_cols, segment_details


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
    first_question_segment_details = None # <<<<<<< NEW: Initialize variable
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
                        segment_cols, segment_details = get_segment_columns(df.columns)

                        # Apply percentage conversion only to identified segment columns
                        for col in segment_cols:
                            if col in df.columns:
                                # Suppress SettingWithCopyWarning temporarily if it occurs here
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)
                                    df[col] = df[col].apply(parse_percentage)
                            else:
                                print(f"Warning: Identified segment column '{col}' not found in DataFrame for QID {q_id}. Columns: {df.columns.tolist()}")


                        metadata = {'id': q_id, 'type': q_type, 'text': q_text, 'segment_cols': segment_cols, 'segment_details': segment_details}
                        processed_data.append((metadata, df))

                        # <<<<<<< NEW: Capture segment details from the first question
                        if first_question_segment_details is None:
                            first_question_segment_details = segment_details
                            print(f"  Captured segment details from first question (QID: {q_id})")

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

                            segment_cols, segment_details = get_segment_columns(df.columns)
                            for col in segment_cols:
                                if col in df.columns:
                                     with warnings.catch_warnings():
                                        warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)
                                        df[col] = df[col].apply(parse_percentage)
                                else:
                                    print(f"Warning: Identified segment column '{col}' not found in DataFrame for final QID {q_id}. Columns: {df.columns.tolist()}")

                            metadata = {'id': q_id, 'type': q_type, 'text': q_text, 'segment_cols': segment_cols, 'segment_details': segment_details}
                            processed_data.append((metadata, df))

                            # <<<<<<< NEW: Capture segment details if this is the first (and only) question
                            if first_question_segment_details is None:
                                first_question_segment_details = segment_details
                                print(f"  Captured segment details from first/only question (QID: {q_id})")

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

    # <<<<< MODIFIED: Return both processed data and first question segment details
    return processed_data, first_question_segment_details

# --- Analysis Functions ---

def calculate_divergence_report(questions_data, divergence_output_dir, top_n_per_question=20, top_n_overall=50):
    """
    Calculates divergence for Ask Opinion questions and generates reports.

    Args:
        questions_data (list): The list of (metadata, df) tuples.
        divergence_output_dir (str): Directory to save the divergence report CSV files.
        top_n_per_question (int): Number of top divergent responses to show per question.
        top_n_overall (int): Number of top divergent responses to show overall.

    Returns:
        pd.DataFrame: DataFrame containing all divergence results (score > 0).
                    Returns empty DataFrame if no Ask Opinion questions found or processed.
    """
    print("\n--- Calculating Divergence Report --- ")
    # Ensure output dir exists
    os.makedirs(divergence_output_dir, exist_ok=True)
    all_divergence_results = []

    for metadata, df in questions_data:
        q_id = metadata.get('id')
        q_text = metadata.get('text')
        q_type = metadata.get('type')

        if q_type != 'Ask Opinion':
            continue

        analysis_segments = metadata.get('analysis_segment_cols', [])

        # Need at least two segments to calculate divergence
        if len(analysis_segments) < 2:
            print(f"  Skipping QID {q_id} (Ask Opinion) - Fewer than 2 valid segments ({len(analysis_segments)}) for divergence calculation.")
            continue

        # Check if expected columns exist
        response_col = 'English Response' # As per DATA_GUIDE.md
        if response_col not in df.columns:
             # Fallback check (based on notebook, might differ in actual data)
             response_col_fallback = 'English Responses'
             if response_col_fallback in df.columns:
                 response_col = response_col_fallback
             else:
                 print(f"  Skipping QID {q_id} - Could not find response column ('{response_col}' or '{response_col_fallback}'). Columns: {df.columns}")
                 continue

        print(f"  Processing QID {q_id} ('{q_text[:50]}...') with {len(analysis_segments)} segments.")
        question_results = []

        # Select only the valid segment columns for calculation
        segment_data = df[analysis_segments].copy()

        # Ensure data is numeric, converting errors to NaN
        for col in analysis_segments:
            segment_data[col] = pd.to_numeric(segment_data[col], errors='coerce')

        for index, row in segment_data.iterrows():
            # Get the agreement rates for valid segments for this specific response
            # Drop NaNs for this row's min/max calculation
            valid_rates = row.dropna()

            if len(valid_rates) < 2:
                # Cannot calculate divergence without at least two valid data points
                continue

            min_rate = valid_rates.min()
            max_rate = valid_rates.max()

            # Calculate divergence score
            max_div = max(max_rate - 0.5, 0)
            min_div = max(0.5 - min_rate, 0)
            divergence_score = math.sqrt(max_div * min_div) if (max_div > 0 and min_div > 0) else 0

            if divergence_score > 0:
                # Find segment names corresponding to min/max rates for this row
                # Handle potential ties by taking the first occurrence
                min_segment = valid_rates.idxmin()
                max_segment = valid_rates.idxmax()

                # Get response text using the original DataFrame index
                response_text = df.loc[index, response_col]

                question_results.append({
                    'Question ID': q_id,
                    'Question Text': q_text,
                    'Response Text': response_text,
                    'Divergence Score': divergence_score,
                    'Min Segment': min_segment,
                    'Min Agreement': min_rate,
                    'Max Segment': max_segment,
                    'Max Agreement': max_rate
                })

        all_divergence_results.extend(question_results)

    if not all_divergence_results:
        print("No responses with positive divergence found across any Ask Opinion questions.")
        return pd.DataFrame() # Return empty DataFrame

    # Create DataFrame from all results
    results_df = pd.DataFrame(all_divergence_results)
    results_df = results_df.sort_values(by='Divergence Score', ascending=False)

    # --- Generate Reports ---

    # 1. Top N per Question Report
    top_per_question = results_df.groupby('Question ID').head(top_n_per_question)
    report_path_per_q = os.path.join(divergence_output_dir, 'divergence_by_question.csv')
    try:
        top_per_question.to_csv(report_path_per_q, index=False, float_format='%.4f')
        print(f"  Saved divergence per question report to: {report_path_per_q}")
    except Exception as e:
        print(f"  Error saving per-question divergence report: {e}")

    # 2. Top N Overall Report
    top_overall = results_df.head(top_n_overall)
    report_path_overall = os.path.join(divergence_output_dir, 'divergence_overall.csv')
    try:
        top_overall.to_csv(report_path_overall, index=False, float_format='%.4f')
        print(f"  Saved overall divergence report to: {report_path_overall}")
    except Exception as e:
        print(f"  Error saving overall divergence report: {e}")

    print("--- Divergence Calculation Complete ---")
    return results_df # Return the full results DataFrame

def calculate_consensus_profiles(questions_data, consensus_output_dir,
                                 percentiles_to_calc = [100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10],
                                 top_n_percentiles = [100, 95, 90],
                                 top_n_count = 5):
    """
    Calculates consensus profiles (percentile minimums) for Ask Opinion questions.

    Args:
        questions_data (list): The list of (metadata, df) tuples.
        consensus_output_dir (str): Directory to save the consensus report CSV files.
        percentiles_to_calc (list): List of percentiles (e.g., 100, 95, 90) for which to calculate the min agreement.
        top_n_percentiles (list): List of percentiles to use for generating Top N reports.
        top_n_count (int): Number of top responses to include in the Top N reports.

    Returns:
        pd.DataFrame: DataFrame containing consensus profiles for all responses.
                     Returns empty DataFrame if no Ask Opinion questions found or processed.
    """
    print("\n--- Calculating Consensus Profiles --- ")
    # Ensure output dir exists
    os.makedirs(consensus_output_dir, exist_ok=True)
    all_consensus_results = []
    percentiles_to_calc.sort(reverse=True) # Ensure descending order for clarity
    percentile_cols = [f'MinAgree_{p}pct' for p in percentiles_to_calc]

    for metadata, df in questions_data:
        q_id = metadata.get('id')
        q_text = metadata.get('text')
        q_type = metadata.get('type')

        if q_type != 'Ask Opinion':
            continue

        analysis_segments = metadata.get('analysis_segment_cols', [])

        # Need segments to calculate consensus
        if not analysis_segments:
            print(f"  Skipping QID {q_id} (Ask Opinion) - No valid segments for consensus calculation.")
            continue

        response_col = 'English Response'
        if response_col not in df.columns:
             response_col_fallback = 'English Responses'
             if response_col_fallback in df.columns:
                 response_col = response_col_fallback
             else:
                 print(f"  Skipping QID {q_id} - Could not find response column ('{response_col}' or '{response_col_fallback}').")
                 continue

        print(f"  Processing QID {q_id} ('{q_text[:50]}...') with {len(analysis_segments)} segments.")

        # Select only the valid segment columns for calculation
        segment_data = df[analysis_segments].copy()
        # Ensure data is numeric, converting errors to NaN
        for col in analysis_segments:
            segment_data[col] = pd.to_numeric(segment_data[col], errors='coerce')

        for index, row in segment_data.iterrows():
            # Get agreement rates for this response, drop NaNs
            valid_rates = row.dropna()

            if valid_rates.empty:
                continue # Skip if no valid rates for this response

            # Sort rates descending to easily find percentile minimums
            sorted_rates = valid_rates.sort_values(ascending=False)
            num_valid_segments = len(sorted_rates)

            response_profile = {
                'Question ID': q_id,
                'Question Text': q_text,
                'Response Text': df.loc[index, response_col],
                'Num Valid Segments': num_valid_segments
            }

            # Calculate minimum agreement for each requested percentile
            for p in percentiles_to_calc:
                col_name = f'MinAgree_{p}pct'
                if p == 0: # Avoid division by zero if 0 is requested
                    response_profile[col_name] = np.nan
                    continue
                
                # Calculate the index corresponding to the percentile minimum
                # We want the agreement rate of the segment at the cutoff defined by (100-p)%
                # Example: For 90%, we want the minimum of the top 90%. If sorted DESC, 
                # this is the value at index floor(N * 0.9) - 1 if N*0.9 is whole, or floor(N*0.9)
                # More simply: calculate how many segments to *keep* (N * p/100)
                # The minimum value among these kept segments is at index: ceil(N * p/100) - 1
                
                # Ensure k is at least 1 and does not exceed N
                k = max(1, min(num_valid_segments, math.ceil(num_valid_segments * p / 100.0)))
                idx_for_min = k - 1 # 0-based index
                
                # The value at this index in the DESC sorted list is the minimum agreement 
                # achieved by at least p% of the segments.
                response_profile[col_name] = sorted_rates.iloc[idx_for_min]

            all_consensus_results.append(response_profile)

    if not all_consensus_results:
        print("No consensus profiles generated (no valid Ask Opinion responses found?).")
        return pd.DataFrame()

    # Create DataFrame
    results_df = pd.DataFrame(all_consensus_results)

    # --- Generate Reports ---
    # 1. Full Profiles Report
    report_path_profiles = os.path.join(consensus_output_dir, 'consensus_profiles.csv')
    try:
        results_df.sort_values(by=f'MinAgree_{top_n_percentiles[0]}pct', ascending=False).to_csv(report_path_profiles, index=False, float_format='%.4f')
        print(f"  Saved full consensus profiles to: {report_path_profiles}")
    except Exception as e:
        print(f"  Error saving full consensus profiles report: {e}")

    # 2. Top N Reports for specified percentiles
    for p in top_n_percentiles:
        col_name = f'MinAgree_{p}pct'
        if col_name in results_df.columns:
            top_n_df = results_df.sort_values(by=col_name, ascending=False).head(top_n_count)
            report_path_top_n = os.path.join(consensus_output_dir, f'consensus_top{top_n_count}_{p}pct.csv')
            try:
                top_n_df.to_csv(report_path_top_n, index=False, float_format='%.4f')
                print(f"  Saved Top {top_n_count} consensus responses ({p} percentile) to: {report_path_top_n}")
            except Exception as e:
                print(f"  Error saving Top {top_n_count} ({p} percentile) report: {e}")
        else:
            print(f"  Warning: Cannot generate Top {top_n_count} report for {p} percentile - column '{col_name}' not found.")

    print("--- Consensus Profile Calculation Complete ---")
    return results_df

def generate_indicator_heatmaps(indicator_codesheet_path, questions_data, indicators_output_dir):
    """
    Generates heatmaps for Indicator poll questions, grouped by category.
    (With improved formatting for labels and layout, and structured titles)
    """
    print("\n--- Generating Indicator Heatmaps --- ")
    # Ensure output dir exists
    os.makedirs(indicators_output_dir, exist_ok=True)

    # --- Load Indicator Codesheet ---
    try:
        indicator_df = pd.read_csv(indicator_codesheet_path)
        indicator_polls = indicator_df[indicator_df['question_type'] == 'Poll Single Select'].copy()
        indicator_polls.dropna(subset=['question_text'], inplace=True)
        qtext_to_category = indicator_polls.set_index('question_text')['question_category'].to_dict()
        qtext_to_code = indicator_polls.set_index('question_text')['question_code'].to_dict()
        print(f"  Loaded {len(indicator_polls)} indicator poll questions from: {indicator_codesheet_path}")
    except FileNotFoundError: print(f"  Error: Indicator codesheet not found at {indicator_codesheet_path}"); return
    except Exception as e: print(f"  Error loading indicator codesheet: {e}"); return

    # --- Map Aggregate Data to Indicators ---
    indicator_question_data = {}
    for metadata, df in questions_data:
        q_text = metadata.get('text'); q_type = metadata.get('type')
        if q_type == 'Poll Single Select' and q_text in qtext_to_category:
            category = qtext_to_category[q_text]
            if category not in indicator_question_data: indicator_question_data[category] = []
            indicator_question_data[category].append((metadata, df, q_text))
    if not indicator_question_data: print("  Warning: No matching indicator poll questions found."); return

    # --- Generate Heatmap per Category ---
    for category, questions_in_category in indicator_question_data.items():
        print(f"  Generating heatmap for category: {category} ({len(questions_in_category)} questions)")
        category_data_for_pivot = []; ordered_labels_info = []
        full_texts_in_category = [q_text for meta, df, q_text in questions_in_category]

        # --- Calculate LCP/LCSuf and derive/wrap labels & title ---
        text_to_label = {}; 
        title_line1 = category # Always start with category name
        title_line2 = "" # Second line for common text or full question
        y_label_type = 'code' # Default to using question codes for y-labels
        y_label_wrap_width = 35 # Characters

        if len(full_texts_in_category) > 1:
            lcp = os.path.commonprefix(full_texts_in_category); lcsuf = longest_common_suffix(full_texts_in_category)
            min_len = min(len(s) for s in full_texts_in_category)
            # Check if LCP/LCSuf are meaningful
            if len(lcp) + len(lcsuf) < min_len and (len(lcp) > 0 or len(lcsuf) > 0): 
                y_label_type = 'varying_part' # Use derived labels
                for text in full_texts_in_category:
                    label = text[len(lcp):len(text)-len(lcsuf)].strip()
                    text_to_label[text] = label if len(label) > 1 else qtext_to_code.get(text, text[:30])
                
                # Construct second title line with common parts
                if len(lcp) > 5 and len(lcsuf) > 5: title_line2 = f"{lcp} ___ {lcsuf}"
                elif len(lcp) > 5: title_line2 = f"{lcp}..."
                elif len(lcsuf) > 5: title_line2 = f"... {lcsuf}"
                else: title_line2 = "(Multiple Questions)" # Fallback if LCP/LCSuf are short
            else:
                # LCP/LCSuf overlap or too short, use codes as labels
                for text in full_texts_in_category: text_to_label[text] = qtext_to_code.get(text, text[:30])
                title_line2 = "(Multiple Questions)" # Indicate multiple questions without clear pattern
        elif len(full_texts_in_category) == 1:
             # Single question: use code for label, full text for title line 2
             text = full_texts_in_category[0]
             text_to_label[text] = qtext_to_code.get(text, text[:30])
             title_line2 = text # Use full text as second title line

        # --- Prepare data for pivoting ---
        max_lines_in_ylabel = 1
        for metadata, df, q_text in questions_in_category:
            q_id = metadata.get('id'); all_n_col = next((col for col in df.columns if col.startswith("All(") and col.endswith(")")), None)
            if 'Responses' not in df.columns or not all_n_col: print(f"    Warning: Skipping QID {q_id} - Missing columns."); continue
            df[all_n_col] = pd.to_numeric(df[all_n_col], errors='coerce')
            temp_df = df[['Responses', all_n_col]].rename(columns={all_n_col: 'Percentage'}); temp_df['QuestionText'] = q_text
            category_data_for_pivot.append(temp_df)
            current_label = text_to_label.get(q_text)
            if q_text not in [t for t, l in ordered_labels_info]:
                wrapped_label = textwrap.fill(current_label, width=y_label_wrap_width)
                ordered_labels_info.append((q_text, wrapped_label))
                max_lines_in_ylabel = max(max_lines_in_ylabel, wrapped_label.count('\n') + 1)
        if not category_data_for_pivot: print(f"    Warning: No valid data for category '{category}'."); continue
        combined_df = pd.concat(category_data_for_pivot, ignore_index=True)
        
        # --- Pivot and Reindex ---
        try:
            heatmap_pivot = combined_df.pivot_table(index='QuestionText', columns='Responses', values='Percentage', aggfunc='first')
            ordered_texts = [text for text, label in ordered_labels_info]
            heatmap_pivot = heatmap_pivot.reindex(ordered_texts)
            heatmap_pivot = heatmap_pivot.sort_index(axis=1)
        except Exception as e: print(f"    Error pivoting data for category '{category}': {e}"); continue
        if heatmap_pivot.empty: print(f"    Warning: Pivoted data empty for '{category}'."); continue
             
        # --- Plotting (with structured title) ---
        n_rows, n_cols = heatmap_pivot.shape
        fig_width = max(8, n_cols * 0.9 + max(0, max_lines_in_ylabel -1) * 1.5 )
        fig_height = max(5, n_rows * 0.7 + 2.5) # Increase base height slightly more for title
        plt.figure(figsize=(fig_width, fig_height))

        # Wrap the second title line if it's long (e.g., full question text)
        wrapped_title_line2 = textwrap.fill(title_line2, width=80) # Adjust width as needed
        final_title = f"{title_line1}\n{wrapped_title_line2}".strip() # Combine lines

        ax = sns.heatmap(heatmap_pivot, annot=True, fmt=".0%", cmap="Blues", linewidths=.5, cbar=False, annot_kws={"size": 9})
        
        ordered_wrapped_labels = [label for text, label in ordered_labels_info]
        ax.set_yticklabels(ordered_wrapped_labels, rotation=0, fontsize=9, va='center') 
        
        plt.xticks(rotation=30, ha='right', fontsize=9)
        # Use final combined title, adjust padding
        plt.title(final_title, fontsize=11, pad=25) # Slightly smaller title font, more padding
        plt.xlabel("Response Options", fontsize=10)
        plt.ylabel("Question Detail" if y_label_type == 'varying_part' else "Question Code", fontsize=10) # Adjust y-axis title based on label type
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust top boundary slightly for title
        
        # --- Save heatmap ---
        safe_category_name = re.sub(r'[^\w\-\. ]', '', category).strip().replace(' ', '_')
        heatmap_filename = f"indicator_heatmap_{safe_category_name}.png"
        heatmap_path = os.path.join(indicators_output_dir, heatmap_filename)
        try:
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            print(f"    Saved heatmap to: {heatmap_path}")
        except Exception as e:
            print(f"    Error saving heatmap for category '{category}': {e}")
        plt.close()

    print("--- Indicator Heatmap Generation Complete ---")

# --- NEW Segment Report Function ---

def generate_segment_report(segment_details_first_q, segments_output_dir):
    """
    Generates a CSV report summarizing segments based on the first question's data.

    Args:
        segment_details_first_q (dict | None): Dictionary mapping segment column names
            to their details ({'name', 'o_code', 'size'}) from the first question.
        segments_output_dir (str): Directory to save the segment summary CSV.
    """
    print("\n--- Generating Segment Summary Report --- ")
    if not segment_details_first_q:
        print("  Skipping segment report: No segment details found from the first question.")
        return

    report_data = []
    for col_name, details in segment_details_first_q.items():
        # Skip the 'All' segment if present
        if col_name.lower().startswith('all('):
            continue

        report_data.append({
            'Segment Name': details.get('name'),
            'Participant Count': details.get('size'),
            'Onboarding Question': details.get('o_code', 'Derived') # Default to 'Derived' if no o_code
        })

    if not report_data:
        print("  Skipping segment report: No non-'All' segments found to report.")
        return

    # Create DataFrame
    report_df = pd.DataFrame(report_data)
    # Optional: Sort by Onboarding Question then Name for consistency
    report_df = report_df.sort_values(by=['Onboarding Question', 'Segment Name'])

    # Save CSV
    report_path = os.path.join(segments_output_dir, 'segment_summary.csv')
    try:
        report_df.to_csv(report_path, index=False)
        print(f"  Saved segment summary report to: {report_path}")
    except Exception as e:
        print(f"  Error saving segment summary report: {e}")

    print("--- Segment Summary Report Generation Complete ---")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Global Dialogues aggregate data.")
    
    # --- Input file arguments (mutually exclusive) ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--gd_number", type=int, help="Global Dialogue cadence number (e.g., 1, 2, 3). Constructs path like Data/GDi/GDi_aggregate.csv")
    input_group.add_argument("--csv_filepath", help="Explicit path to the aggregate.csv file.")

    # --- Other arguments ---
    parser.add_argument("-o", "--output_dir", default="analysis_output", help="Base directory to save results and cache.")
    parser.add_argument("-s", "--min_segment_size", type=int, default=DEFAULT_MIN_SEGMENT_SIZE, help="Minimum participant size for a segment to be included in analysis.")
    parser.add_argument("--force_reparse", action="store_true", help="Force reparsing from CSV, ignoring cache.")
    # Set default padding based on notebook observation
    parser.add_argument("--padding_rows", type=int, default=9, help="Number of header/junk rows to skip at the start of the CSV.")
    parser.add_argument("--top_n_divergence", type=int, default=20, help="Number of top divergent responses to report per question.")
    parser.add_argument("--top_n_divergence_overall", type=int, default=50, help="Number of top divergent responses to report overall.")
    # Consensus Args
    parser.add_argument("--consensus_percentiles", type=int, nargs='+', default=[100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10], help="Percentiles for consensus profile calculation (e.g., 100 95 90)")
    parser.add_argument("--consensus_top_n_percentiles", type=int, nargs='+', default=[100, 95, 90], help="Percentiles to generate Top N reports for (e.g., 100 95)")
    parser.add_argument("--consensus_top_n_count", type=int, default=5, help="Number of responses for Top N consensus reports.")
    # Add arg for indicator codesheet 
    parser.add_argument("--indicator_codesheet", default="Data/Documentation/INDICATOR_CODESHEET.csv", help="Path to the Indicator Codesheet CSV.")

    args = parser.parse_args()

    # --- Determine Input CSV Path ---
    input_csv_path = None; gd_identifier = "custom_input" # Default identifier
    if args.gd_number:
        gd_num = args.gd_number
        gd_identifier = f"GD{gd_num}" # Set identifier based on number
        constructed_path = os.path.join("Data", gd_identifier, f"{gd_identifier}_aggregate.csv")
        if not os.path.exists(constructed_path): print(f"Error: Constructed path does not exist: {constructed_path}"); exit(1)
        input_csv_path = constructed_path
    elif args.csv_filepath:
        # Try to infer GD number from path for directory name, fallback to default
        match = re.search(r'GD(\d+)', args.csv_filepath, re.IGNORECASE)
        if match: gd_identifier = f"GD{match.group(1)}"
        if not os.path.exists(args.csv_filepath): print(f"Error: Provided file path does not exist: {args.csv_filepath}"); exit(1)
        input_csv_path = args.csv_filepath
    else: print("Error: No input specified."); exit(1)
    print(f"Using input file: {input_csv_path}")
    print(f"Using output identifier: {gd_identifier}")

    # --- Setup Output Directories ---
    base_output_dir = os.path.join(args.output_dir, gd_identifier)
    consensus_output_dir = os.path.join(base_output_dir, "consensus")
    divergence_output_dir = os.path.join(base_output_dir, "divergence")
    indicators_output_dir = os.path.join(base_output_dir, "indicators")
    segments_output_dir = os.path.join(base_output_dir, "segments")
    
    # Create directories if they don't exist
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(consensus_output_dir, exist_ok=True)
    os.makedirs(divergence_output_dir, exist_ok=True)
    os.makedirs(indicators_output_dir, exist_ok=True)
    os.makedirs(segments_output_dir, exist_ok=True)
    print(f"Output will be saved under: {base_output_dir}/")

    # Update cache file path to be inside the base GDi/custom directory
    cache_file = os.path.join(base_output_dir, CACHE_FILENAME)
    print(f"Using cache file: {cache_file}")
    print("(Consider adding this cache file path to your .gitignore)")

    # --- Load and Preprocess Data ---
    all_questions_data, first_question_segment_details = load_and_preprocess_data(input_csv_path, cache_file, args.force_reparse, args.padding_rows)
    if not all_questions_data: print("Failed to load data."); exit(1)
    print(f"\nSuccessfully loaded/processed {len(all_questions_data)} questions from {input_csv_path}.")

    # --- Generate Segment Report (using first question data) ---
    generate_segment_report(first_question_segment_details, segments_output_dir)

    # --- Filter segments based on size for subsequent analyses ---
    print(f"\nFiltering segments with size < {args.min_segment_size} for divergence/consensus/indicators...")
    original_total_segments = 0
    filtered_total_segments = 0
    processed_questions_data_filtered = [] # Store results after filtering

    for metadata, df in all_questions_data:
        original_segments = metadata.get('segment_cols', [])
        segment_details = metadata.get('segment_details', {})
        original_total_segments += len(original_segments)

        # Filter segment columns based on size
        # Use np.nan_to_num to treat NaN sizes as 0 for comparison
        valid_segments = [
            col for col in original_segments
            if col in segment_details and np.nan_to_num(segment_details[col]['size']) >= args.min_segment_size
        ]
        filtered_total_segments += len(valid_segments)

        # Create a new metadata dict with filtered segments list
        # Keep original segment_details dict untouched for reference if needed
        filtered_metadata = metadata.copy()
        filtered_metadata['analysis_segment_cols'] = valid_segments # Use a distinct key

        processed_questions_data_filtered.append((filtered_metadata, df))

        # Log changes for this question if any segments were removed
        removed_count = len(original_segments) - len(valid_segments)
        if removed_count > 0:
            print(f"  QID {metadata.get('id', 'N/A')}: Removed {removed_count} segments. Kept {len(valid_segments)} / {len(original_segments)}.")

    # Replace the main list with the one containing filtered segment lists for analysis
    analysis_questions_data = processed_questions_data_filtered # Use a new variable name for clarity
    print(f"Segment filtering complete. Total segments kept for divergence/consensus/indicators analysis: {filtered_total_segments} / {original_total_segments}.")


    # --- Run Analysis Functions (using filtered data) ---
    divergence_results = calculate_divergence_report(
        analysis_questions_data,  # Use filtered data
        divergence_output_dir, # Pass specific directory
        top_n_per_question=args.top_n_divergence,
        top_n_overall=args.top_n_divergence_overall
    )

    consensus_results = calculate_consensus_profiles(
        analysis_questions_data, # Use filtered data
        consensus_output_dir, # Pass specific directory
        percentiles_to_calc=args.consensus_percentiles,
        top_n_percentiles=args.consensus_top_n_percentiles,
        top_n_count=args.consensus_top_n_count
    )
    
    generate_indicator_heatmaps(
        args.indicator_codesheet, 
        analysis_questions_data, # Use filtered data
        indicators_output_dir # Pass specific directory
    )

    # --- Optional: Further summary based on results ---
    if divergence_results is not None and not divergence_results.empty:
        # Example: Find most divergent question overall
        most_divergent_q_group = divergence_results.loc[divergence_results['Divergence Score'].idxmax()]
        print("\n--- Overall Summary Snippets ---")
        print(f"Most Divergent Response Overall (Score: {most_divergent_q_group['Divergence Score']:.4f}):")
        print(f"  Question : {most_divergent_q_group['Question Text'][:100]}...")
        print(f"  Response : {most_divergent_q_group['Response Text'][:100]}...")
        print(f"  Segments : {most_divergent_q_group['Min Segment']} ({most_divergent_q_group['Min Agreement']:.1%}) vs {most_divergent_q_group['Max Segment']} ({most_divergent_q_group['Max Agreement']:.1%})")
    else:
        print("\nNo divergence results to summarize.")

    # Consensus summary (add example)
    if consensus_results is not None and not consensus_results.empty and 'MinAgree_90pct' in consensus_results.columns:
         highest_consensus_90pct = consensus_results.loc[consensus_results['MinAgree_90pct'].idxmax()]
         print("\nHighest Consensus Response (90th Percentile Minimum):")
         print(f"  Score    : {highest_consensus_90pct['MinAgree_90pct']:.4f}")
         print(f"  Question : {highest_consensus_90pct['Question Text'][:100]}...")
         print(f"  Response : {highest_consensus_90pct['Response Text'][:100]}...")
         print(f"  (Based on {highest_consensus_90pct['Num Valid Segments']} segments)")
    else:
         print("\nNo consensus results to summarize (or 90pct column missing).")

    print("\nScript finished.") 