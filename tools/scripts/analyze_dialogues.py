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
    Identifies segment columns (typically ending in '(Number)') and extracts the number N.

    Args:
        df_columns (list): List of column names from a DataFrame.

    Returns:
        tuple: (list_of_segment_column_names, dict_of_segment_name_to_size)
    """
    segment_cols = []
    segment_sizes = {}
    # Regex updated to find columns ending in (Number), allowing for whitespace variations.
    # It captures the digits within the parentheses.
    pattern = re.compile(r'.*?\s*\(\s*(\d+)\s*\)\s*$') # Updated pattern

    for col in df_columns:
        match = pattern.match(col)
        if match:
            segment_cols.append(col)
            try:
                # The number is captured by group 1 of the updated pattern
                segment_sizes[col] = int(match.group(1))
            except ValueError:
                # This should be less likely now pattern requires digits, but handle just in case
                segment_sizes[col] = np.nan
                print(f"Warning: Could not parse extracted size digits from segment column: {col}")
            except IndexError:
                 # Should not happen if pattern matched, means regex logic error
                 segment_sizes[col] = np.nan
                 print(f"Warning: Regex matched but failed to extract size group from segment column: {col}")

    # Sort to bring 'All(...)' column(s) to the front if they exist
    all_cols = [col for col in segment_cols if col.startswith('All(') or col.startswith('All (')]
    other_cols = [col for col in segment_cols if not (col.startswith('All(') or col.startswith('All ('))]
    # Ensure sorting is stable if needed, though order of non-'All' segments may not matter
    sorted_segment_cols = sorted(all_cols) + sorted(other_cols)

    if not sorted_segment_cols and len(df_columns) > 5: # Adjusted threshold slightly
         # Check standard non-segment columns first before warning
         standard_cols = {'Question ID', 'Question Type', 'Question', 'Responses', 'English Response', 'Original Response'}
         potential_segments = [c for c in df_columns if c not in standard_cols and '(' in c]
         if potential_segments:
              print(f"Warning: Regex pattern did not find segment columns ending like '(Number)' in headers: {df_columns[:10]}... Check CSV format or regex pattern.")
         # Else: Likely a question type without segments (e.g. Ask Experience), so no warning needed.

    return sorted_segment_cols, segment_sizes


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

def calculate_divergence_report(questions_data, output_dir, top_n_per_question=20, top_n_overall=50):
    """
    Calculates divergence for Ask Opinion questions and generates reports.

    Args:
        questions_data (list): The list of (metadata, df) tuples.
        output_dir (str): Directory to save the report CSV files.
        top_n_per_question (int): Number of top divergent responses to show per question.
        top_n_overall (int): Number of top divergent responses to show overall.

    Returns:
        pd.DataFrame: DataFrame containing all divergence results (score > 0).
                    Returns empty DataFrame if no Ask Opinion questions found or processed.
    """
    print("\n--- Calculating Divergence Report --- ")
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
    report_path_per_q = os.path.join(output_dir, 'divergence_by_question.csv')
    try:
        top_per_question.to_csv(report_path_per_q, index=False, float_format='%.4f')
        print(f"  Saved top {top_n_per_question} divergent responses per question to: {report_path_per_q}")
    except Exception as e:
        print(f"  Error saving per-question divergence report: {e}")

    # 2. Top N Overall Report
    top_overall = results_df.head(top_n_overall)
    report_path_overall = os.path.join(output_dir, 'divergence_overall.csv')
    try:
        top_overall.to_csv(report_path_overall, index=False, float_format='%.4f')
        print(f"  Saved top {top_n_overall} divergent responses overall to: {report_path_overall}")
    except Exception as e:
        print(f"  Error saving overall divergence report: {e}")

    print("--- Divergence Calculation Complete ---")
    return results_df # Return the full results DataFrame

def calculate_consensus_profiles(questions_data, output_dir, 
                                 percentiles_to_calc = [100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10],
                                 top_n_percentiles = [100, 95, 90],
                                 top_n_count = 5):
    """
    Calculates consensus profiles (percentile minimums) for Ask Opinion questions.

    Args:
        questions_data (list): The list of (metadata, df) tuples.
        output_dir (str): Directory to save the report CSV files.
        percentiles_to_calc (list): List of percentiles (e.g., 100, 95, 90) for which to calculate the min agreement.
        top_n_percentiles (list): List of percentiles to use for generating Top N reports.
        top_n_count (int): Number of top responses to include in the Top N reports.

    Returns:
        pd.DataFrame: DataFrame containing consensus profiles for all responses.
                     Returns empty DataFrame if no Ask Opinion questions found or processed.
    """
    print("\n--- Calculating Consensus Profiles --- ")
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
    report_path_profiles = os.path.join(output_dir, 'consensus_profiles.csv')
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
            report_path_top_n = os.path.join(output_dir, f'consensus_top{top_n_count}_{p}pct.csv')
            try:
                top_n_df.to_csv(report_path_top_n, index=False, float_format='%.4f')
                print(f"  Saved Top {top_n_count} consensus responses ({p} percentile) to: {report_path_top_n}")
            except Exception as e:
                print(f"  Error saving Top {top_n_count} ({p} percentile) report: {e}")
        else:
            print(f"  Warning: Cannot generate Top {top_n_count} report for {p} percentile - column '{col_name}' not found.")

    print("--- Consensus Profile Calculation Complete ---")
    return results_df

def generate_indicator_heatmaps(indicator_codesheet_path, questions_data, output_dir):
    """
    Generates heatmaps for Indicator poll questions, grouped by category.

    Args:
        indicator_codesheet_path (str): Path to the INDICATOR_CODESHEET.csv file.
        questions_data (list): The list of (metadata, df) tuples from aggregate data.
        output_dir (str): Directory to save the heatmap PNG files.
    """
    print("\n--- Generating Indicator Heatmaps --- ")

    # --- Load Indicator Codesheet ---
    try:
        indicator_df = pd.read_csv(indicator_codesheet_path)
        indicator_polls = indicator_df[indicator_df['question_type'] == 'Poll Single Select'].copy()
        indicator_polls.dropna(subset=['question_text'], inplace=True)
        qtext_to_category = indicator_polls.set_index('question_text')['question_category'].to_dict()
        qtext_to_code = indicator_polls.set_index('question_text')['question_code'].to_dict()
        print(f"  Loaded {len(indicator_polls)} indicator poll questions from: {indicator_codesheet_path}")
    except FileNotFoundError:
        print(f"  Error: Indicator codesheet not found at {indicator_codesheet_path}"); return
    except Exception as e:
        print(f"  Error loading indicator codesheet: {e}"); return

    # --- Map Aggregate Data to Indicators ---
    indicator_question_data = {}
    for metadata, df in questions_data:
        q_text = metadata.get('text')
        q_type = metadata.get('type')
        if q_type == 'Poll Single Select' and q_text in qtext_to_category:
            category = qtext_to_category[q_text]
            if category not in indicator_question_data:
                indicator_question_data[category] = []
            # Store metadata, df, and full question text for label generation
            indicator_question_data[category].append((metadata, df, q_text))

    if not indicator_question_data:
        print("  Warning: No matching indicator poll questions found in the aggregate data.")
        return

    # --- Generate Heatmap per Category ---
    for category, questions_in_category in indicator_question_data.items():
        print(f"  Generating heatmap for category: {category} ({len(questions_in_category)} questions)")
        category_data_for_pivot = []
        # Store tuples of (full_text, derived_label) to maintain order and map labels
        ordered_labels_info = [] 
        full_texts_in_category = [q_text for meta, df, q_text in questions_in_category]

        # --- Calculate LCP/LCSuf and derive labels ---
        text_to_label = {}
        plot_title = f"Indicator: {category}" # Default title
        if len(full_texts_in_category) > 1:
            lcp = os.path.commonprefix(full_texts_in_category)
            lcsuf = longest_common_suffix(full_texts_in_category)
            
            # Basic check to see if LCP/LCSuf are meaningful
            min_len = min(len(s) for s in full_texts_in_category)
            # Only use LCP/LCSuf if they don't overlap and leave some middle part
            if len(lcp) + len(lcsuf) < min_len:
                 for text in full_texts_in_category:
                     label = text[len(lcp):len(text)-len(lcsuf)].strip()
                     # Use short code as fallback if label is empty/too short
                     text_to_label[text] = label if len(label) > 1 else qtext_to_code.get(text, text[:30])
                 
                 # Refine title if LCP/LCSuf seem helpful
                 if len(lcp) > 5 and len(lcsuf) > 5: plot_title = f"{category}\n{lcp} ___ {lcsuf}"
                 elif len(lcp) > 5: plot_title = f"{category}\n{lcp}..."
                 elif len(lcsuf) > 5: plot_title = f"{category}\n... {lcsuf}"
            else:
                 # LCP/LCSuf overlap or cover everything, use codes as labels
                 for text in full_texts_in_category:
                     text_to_label[text] = qtext_to_code.get(text, text[:30])
        elif len(full_texts_in_category) == 1:
             # Single question, use code as label
             text = full_texts_in_category[0]
             text_to_label[text] = qtext_to_code.get(text, text[:30])

        # --- Prepare data for pivoting ---
        for metadata, df, q_text in questions_in_category:
            q_id = metadata.get('id')
            all_n_col = next((col for col in df.columns if col.startswith("All(") and col.endswith(")")), None)
            
            if 'Responses' not in df.columns or not all_n_col:
                print(f"    Warning: Skipping QID {q_id} - Missing 'Responses' or 'All(N)' column.")
                continue

            # Ensure percentages are numeric
            df[all_n_col] = pd.to_numeric(df[all_n_col], errors='coerce')
            temp_df = df[['Responses', all_n_col]].copy()
            temp_df.rename(columns={all_n_col: 'Percentage'}, inplace=True)
            temp_df['QuestionText'] = q_text # Use the full text for pivot index
            category_data_for_pivot.append(temp_df)
            
            # Store label info in order, ensuring uniqueness based on text
            current_label = text_to_label.get(q_text) 
            if q_text not in [t for t, l in ordered_labels_info]:
                 ordered_labels_info.append((q_text, current_label))
                 
        if not category_data_for_pivot:
            print(f"    Warning: No valid data extracted for category '{category}'. Skipping heatmap.")
            continue

        combined_df = pd.concat(category_data_for_pivot, ignore_index=True)
        
        # --- Pivot and Reindex ---
        try:
            heatmap_pivot = combined_df.pivot_table(
                index='QuestionText', # Pivot using full text
                columns='Responses', 
                values='Percentage',
                aggfunc='first' 
            )
            # Reindex rows using the original texts to maintain order
            ordered_texts = [text for text, label in ordered_labels_info]
            heatmap_pivot = heatmap_pivot.reindex(ordered_texts)
            # Sort columns (response options) alphabetically
            heatmap_pivot = heatmap_pivot.sort_index(axis=1)
        except Exception as e:
            print(f"    Error pivoting data for category '{category}': {e}")
            continue

        if heatmap_pivot.empty:
             print(f"    Warning: Pivoted data is empty for category '{category}'. Skipping heatmap.")
             continue
             
        # --- Plotting ---
        plt.figure(figsize=(max(10, heatmap_pivot.shape[1] * 1.2), max(6, heatmap_pivot.shape[0] * 0.8)))
        ax = sns.heatmap(heatmap_pivot, annot=True, fmt=".0%", cmap="Blues", linewidths=.5, cbar=False)
        
        # Set Y-tick labels to the derived varying parts
        ordered_varying_labels = [label for text, label in ordered_labels_info]
        ax.set_yticklabels(ordered_varying_labels, rotation=0)
        
        plt.xticks(rotation=15, ha='right')
        plt.title(plot_title, fontsize=14) # Use generated title
        plt.xlabel("Response Options", fontsize=10)
        plt.ylabel("Question Detail", fontsize=10) # Use generic Y-axis label
        plt.tight_layout()

        # --- Save heatmap ---
        safe_category_name = re.sub(r'[^\w\-\. ]', '', category).strip().replace(' ', '_')
        heatmap_filename = f"indicator_heatmap_{safe_category_name}.png"
        heatmap_path = os.path.join(output_dir, heatmap_filename)
        try:
            plt.savefig(heatmap_path, dpi=150)
            print(f"    Saved heatmap to: {heatmap_path}")
        except Exception as e:
            print(f"    Error saving heatmap for category '{category}': {e}")
        plt.close()

    print("--- Indicator Heatmap Generation Complete ---")

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
    parser.add_argument("--top_n_divergence", type=int, default=20, help="Number of top divergent responses to report per question.")
    parser.add_argument("--top_n_divergence_overall", type=int, default=50, help="Number of top divergent responses to report overall.")
    # Consensus Args
    parser.add_argument("--consensus_percentiles", type=int, nargs='+', default=[100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10], help="Percentiles for consensus profile calculation (e.g., 100 95 90)")
    parser.add_argument("--consensus_top_n_percentiles", type=int, nargs='+', default=[100, 95, 90], help="Percentiles to generate Top N reports for (e.g., 100 95)")
    parser.add_argument("--consensus_top_n_count", type=int, default=5, help="Number of responses for Top N consensus reports.")
    # Add arg for indicator codesheet 
    parser.add_argument("--indicator_codesheet", default="Data/Documentation/INDICATOR_CODESHEET.csv", help="Path to the Indicator Codesheet CSV.")

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


    # --- Run Analysis Functions --- 
    divergence_results = calculate_divergence_report(
        all_questions_data,
        args.output_dir,
        top_n_per_question=args.top_n_divergence,
        top_n_overall=args.top_n_divergence_overall
        )

    consensus_results = calculate_consensus_profiles(
        all_questions_data,
        args.output_dir,
        percentiles_to_calc=args.consensus_percentiles,
        top_n_percentiles=args.consensus_top_n_percentiles,
        top_n_count=args.consensus_top_n_count
    )
    
    generate_indicator_heatmaps(
        args.indicator_codesheet, 
        all_questions_data, 
        args.output_dir
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