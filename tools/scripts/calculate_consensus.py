import argparse
import logging
import os
import pandas as pd
import numpy as np
import math
import re
# Assuming analysis_utils has the refined get_segment_columns
from lib.analysis_utils import parse_percentage, get_segment_columns

# --- Suppress PerformanceWarning ---
import warnings
from pandas.errors import PerformanceWarning
# Suppress the specific PerformanceWarning related to fragmentation
warnings.filterwarnings('ignore', category=PerformanceWarning)
# --------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Core Calculation Functions (modified for standardized data) ---

def calculate_consensus_profiles(standardized_df, segment_counts_df, output_dir,
                                 min_segment_size, # Now required for per-question filtering
                                 percentiles_to_calc=[100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10],
                                 top_n_percentiles=[100, 95, 90],
                                 top_n_count=5):
    """
    Calculates consensus profiles for Ask Opinion questions using standardized data.
    Filters segments per question based on counts from segment_counts_df.

    Args:
        standardized_df (pd.DataFrame): DataFrame from _aggregate_standardized.csv.
        segment_counts_df (pd.DataFrame): DataFrame from _segment_counts_by_question.csv.
        output_dir (str): Directory to save the consensus report CSV files.
        min_segment_size (int): Minimum participant count for a segment to be included in analysis for a specific question.
        percentiles_to_calc (list): List of percentiles to calculate consensus for.
        top_n_percentiles (list): List of percentiles to show top N responses for.
        top_n_count (int): Number of top responses to show for each percentile.

    Returns:
        pd.DataFrame: DataFrame containing all consensus results.
                    Returns empty DataFrame if no Ask Opinion questions found or processed.
    """
    print("\n--- Calculating Consensus Profiles (using standardized data) --- ")
    os.makedirs(output_dir, exist_ok=True)
    all_consensus_results = []
    percentiles_to_calc.sort(reverse=True) # Ensure descending order for clarity
    percentile_cols = [f'MinAgree_{p}pct' for p in percentiles_to_calc]

    # Identify all potential segment columns from the standardized data
    # Exclude known non-segment columns that are part of the base header
    base_cols = ["Question ID", "Question Type", "Question", "Response", "OriginalResponse",
                 "Star", "Categories", "Sentiment", "Submitted By", "Language", "Sample ID", "Participant ID"]
    all_segment_columns = [col for col in standardized_df.columns if col not in base_cols]

    if not all_segment_columns:
         print("  Error: No segment columns found in the standardized data header.")
         return pd.DataFrame()

    print(f"  Identified {len(all_segment_columns)} potential segment columns in standardized data.")

    # Pre-process segment counts for easier lookup
    segment_counts_df.set_index('Question ID', inplace=True)
    # Convert counts columns to numeric, coercing errors (like empty strings) to NaN
    for col in all_segment_columns:
        if col in segment_counts_df.columns:
             segment_counts_df[col] = pd.to_numeric(segment_counts_df[col], errors='coerce')

    # Group by question and process each one
    for q_id, group in standardized_df.groupby('Question ID'):
        q_text = group['Question'].iloc[0]
        q_type = group['Question Type'].iloc[0]

        if q_type != 'Ask Opinion':
            continue

        # --- Filter segments for *this specific question* ---
        try:
            q_counts = segment_counts_df.loc[q_id]
            valid_segments_for_q = [
                col for col in all_segment_columns
                if col in q_counts.index and pd.notna(q_counts[col]) and q_counts[col] >= min_segment_size
            ]
        except KeyError:
             print(f"  Skipping QID {q_id} - Could not find segment counts for this question ID.")
             continue
        except Exception as e:
             print(f"  Error accessing segment counts for QID {q_id}: {e}")
             continue

        if not valid_segments_for_q:
            print(f"  Skipping QID {q_id} (Ask Opinion) - No segments met min size ({min_segment_size}) for this question.")
            continue

        print(f"  Processing QID {q_id} ('{q_text[:50]}...') with {len(valid_segments_for_q)} valid segments (>= {min_segment_size} participants).")

        # Select only the valid segment columns for calculation from this group
        segment_data = group[valid_segments_for_q].copy()

        # Parse percentages and ensure numeric type
        for col in valid_segments_for_q:
            segment_data[col] = segment_data[col].apply(parse_percentage)
            segment_data[col] = pd.to_numeric(segment_data[col], errors='coerce')

        for index, row in segment_data.iterrows():
            # Get agreement rates for this response, drop NaNs
            valid_rates = row.dropna()

            if valid_rates.empty:
                continue # Skip if no valid rates for this response

            # Sort rates descending to easily find percentile minimums
            sorted_rates = valid_rates.sort_values(ascending=False)
            num_valid_segments_for_row = len(sorted_rates)

            # Use the group's index to get the original response text
            response_text = group.loc[index, 'Response'] # Use new standard column name

            response_profile = {
                'Question ID': q_id,
                'Question Text': q_text,
                'Response Text': response_text,
                'Num Valid Segments': num_valid_segments_for_row
            }

            # Calculate minimum agreement for each requested percentile
            for p in percentiles_to_calc:
                col_name = f'MinAgree_{p}pct'
                if p == 0:
                    response_profile[col_name] = np.nan; continue

                # Calculate index for percentile minimum (same logic as before)
                k = max(1, min(num_valid_segments_for_row, math.ceil(num_valid_segments_for_row * p / 100.0)))
                idx_for_min = k - 1
                response_profile[col_name] = sorted_rates.iloc[idx_for_min]

            all_consensus_results.append(response_profile)

    if not all_consensus_results:
        print("No consensus profiles generated (no valid Ask Opinion responses found or processed?).")
        return pd.DataFrame()

    # Create DataFrame
    results_df = pd.DataFrame(all_consensus_results)

    # --- Generate Reports ---
    # 1. Full Profiles Report
    report_path_profiles = os.path.join(output_dir, 'consensus_profiles.csv')
    try:
        # Sort by highest percentile first
        results_df.sort_values(by=f'MinAgree_{top_n_percentiles[0]}pct', ascending=False).to_csv(report_path_profiles, index=False, float_format='%.4f')
        print(f"  Saved full consensus profiles to: {report_path_profiles}")
    except Exception as e:
        print(f"  Error saving full consensus profiles report: {e}")

    print("--- Consensus Profile Calculation Complete ---")
    return results_df


def calculate_major_segment_consensus(standardized_df, segment_counts_df, segment_details_map, output_dir,
                                      min_segment_size, # Used for dynamic threshold calc
                                      top_n=10):
    """
    Calculates the minimum agreement rate across *dynamically determined major segments*
    for each Ask Opinion response (excluding NaN and 0% rates) and reports the top N
    responses per question based on this minimum agreement rate. Major segments are
    determined *per question* based on counts and criteria (>= threshold, not O1/O7).

    Args:
        standardized_df (pd.DataFrame): DataFrame from _aggregate_standardized.csv.
        segment_counts_df (pd.DataFrame): DataFrame from _segment_counts_by_question.csv.
        segment_details_map (dict): Maps core segment names to their details (o_code).
        output_dir (str): Directory to save the major segment consensus report CSV files.
        min_segment_size (int): The *base* minimum segment size threshold (used if dynamic calc fails or as lower bound).
        top_n (int): Number of top responses per question to report.

    Returns:
        pd.DataFrame: DataFrame containing the top N responses per question,
                    sorted by the calculated minimum agreement rate.
                    Returns empty DataFrame if no valid minimums are found.
    """
    print(f"\n--- Calculating Highest Minimum Agreement (Top {top_n}) Across Major Segments --- ")
    os.makedirs(output_dir, exist_ok=True)
    all_min_rates_per_response = []
    all_major_segments_used = set() # Track all major segments encountered across all questions

    # Identify all potential segment columns from the standardized data
    base_cols = ["Question ID", "Question Type", "Question", "Response", "OriginalResponse",
                 "Star", "Categories", "Sentiment", "Submitted By", "Language", "Sample ID", "Participant ID"]
    all_segment_columns = [col for col in standardized_df.columns if col not in base_cols]

    if not all_segment_columns:
         print("  Error: No segment columns found in the standardized data header.")
         return pd.DataFrame()

    # Ensure segment_counts_df is indexed by Question ID if not already done
    if segment_counts_df.index.name != 'Question ID':
        segment_counts_df = segment_counts_df.set_index('Question ID')
        # Ensure counts are numeric
        for col in all_segment_columns:
            if col in segment_counts_df.columns:
                segment_counts_df[col] = pd.to_numeric(segment_counts_df[col], errors='coerce')

    # Group by question and process each one
    for q_id, group in standardized_df.groupby('Question ID'):
        q_text = group['Question'].iloc[0]
        q_type = group['Question Type'].iloc[0]

        if q_type != 'Ask Opinion':
            continue

        # --- Dynamically Determine Major Segments for *this specific question* ---
        major_segments_for_q = []
        try:
            q_counts = segment_counts_df.loc[q_id]
            # Determine threshold for this question
            min_threshold_q = min_segment_size # Default
            if 'All' in q_counts.index and pd.notna(q_counts['All']) and q_counts['All'] > 0:
                total_N_q = q_counts['All']
                dynamic_threshold = max(1, round(total_N_q / 100.0))
                min_threshold_q = max(min_segment_size, dynamic_threshold)
            else:
                 logging.debug(f"QID {q_id}: Cannot calculate dynamic threshold, using base min_segment_size ({min_segment_size}).")

            for core_seg_name in all_segment_columns:
                if core_seg_name.lower() == 'all': continue # Exclude 'All'

                # Check size for this question
                size = q_counts.get(core_seg_name)
                if pd.isna(size) or size < min_threshold_q: continue # Skip if too small or NaN

                # Check O-code from the details map
                details = segment_details_map.get(core_seg_name)
                o_code = details.get('o_code') if details else None

                if o_code not in ['O1', 'O7']:
                    major_segments_for_q.append(core_seg_name)

            if not major_segments_for_q:
                # print(f"  Skipping QID {q_id} - No major segments identified meeting criteria for this question (threshold: {min_threshold_q}).")
                continue
            all_major_segments_used.update(major_segments_for_q) # Add to overall set
            print(f"  Processing QID {q_id} ('{q_text[:50]}...') with {len(major_segments_for_q)} major segments (threshold: {min_threshold_q}).")

        except KeyError:
             print(f"  Skipping QID {q_id} - Could not find segment counts.")
             continue
        except Exception as e:
             print(f"  Error determining major segments for QID {q_id}: {e}")
             continue

        # Loop through responses (rows in the group)
        for index, row in group.iterrows():
            response_text = row['Response'] # Use new standard name
            parsed_rates = {}

            # 1. Get & Parse Rates for the major segments identified *for this question*
            for seg_col in major_segments_for_q:
                 # Value should already be in the row from standardized_df
                 parsed_value = parse_percentage(row.get(seg_col)) # Use .get for safety
                 if pd.notna(parsed_value):
                      parsed_rates[seg_col] = parsed_value

            # 2. Filter Rates for minimum calculation (Exclude zeros)
            rates_for_min_calc = pd.Series(list(parsed_rates.values()))
            rates_for_min_calc = rates_for_min_calc[rates_for_min_calc > 0.0]

            # 3. Calculate Minimum if valid rates (>0) remain
            if not rates_for_min_calc.empty:
                min_rate = rates_for_min_calc.min()

                # 4. Store result, including individual rates
                result_data = {
                    'Question ID': q_id,
                    'Question Text': q_text,
                    'Response Text': response_text,
                    'Min Agreement Rate': min_rate
                }
                # Add individual parsed rates (including 0.0 if present)
                result_data.update(parsed_rates)
                all_min_rates_per_response.append(result_data)

    if not all_min_rates_per_response:
        print("\nNo responses found with a minimum agreement rate > 0% across identified major segments for any question.")
        return pd.DataFrame()

    # Create DataFrame from all calculated minimums
    results_df = pd.DataFrame(all_min_rates_per_response)

    # Sort by QID, then by Min Agreement Rate (descending)
    results_df = results_df.sort_values(by=['Question ID', 'Min Agreement Rate'], ascending=[True, False])

    # Get Top N per Question
    top_n_df = results_df.groupby('Question ID').head(top_n).reset_index(drop=True)

    # --- Ensure all major segment columns *encountered* are present ---
    core_cols = ['Question ID', 'Question Text', 'Response Text', 'Min Agreement Rate']
    # Use the set of all major segments found across all questions
    all_expected_cols = core_cols + sorted(list(all_major_segments_used))
    # Add any missing columns and fill with NA
    top_n_df = top_n_df.reindex(columns=all_expected_cols, fill_value=pd.NA)

    # Reorder columns for better readability
    segment_rate_cols_in_df = [col for col in top_n_df.columns if col in all_major_segments_used]
    segment_rate_cols_in_df.sort()
    final_column_order = core_cols + segment_rate_cols_in_df
    top_n_df = top_n_df[final_column_order]

    # Save report
    report_path = os.path.join(output_dir, f'major_segment_min_agreement_top{top_n}.csv')
    try:
        top_n_df.to_csv(report_path, index=False, float_format='%.4f')
        print(f"  Saved Top {top_n} responses per question by minimum major segment agreement to: {report_path}")
    except Exception as e:
        print(f"  Error saving minimum agreement report: {e}")

    print("--- Highest Minimum Agreement Calculation Complete ---")
    return top_n_df


def main():
    parser = argparse.ArgumentParser(description='Calculate consensus analysis from standardized data.')

    # Input specification (Mutually Exclusive Group)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--gd_number", type=int, help="Global Dialogue cadence number (e.g., 1, 2, 3). Constructs default paths.")
    input_group.add_argument("--standardized_csv", help="Explicit path to the standardized aggregate CSV file.")

    # Conditionally required input path for segment counts
    parser.add_argument('--segment_counts_csv', help='Path to the segment counts per question CSV file (required if --standardized_csv is used).')

    # Output Directory
    parser.add_argument('-o', '--output_dir', help='Directory to save consensus output files (required if --standardized_csv is used).')

    # Analysis Parameters
    parser.add_argument('--percentiles', type=int, nargs='+', default=[100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10],
                       help='List of percentiles to calculate consensus profiles for.')
    parser.add_argument('--top_n_percentiles', type=int, nargs='+', default=[100, 95, 90],
                       help='List of percentiles to generate Top N reports for.')
    parser.add_argument('--top_n_count', type=int, default=5,
                       help='Number of responses for Top N consensus reports.')
    parser.add_argument('--min_segment_size', type=int, default=15,
                       help='Minimum participant size for a segment to be included in analysis (per question).')
    # Major Segment Consensus specific
    parser.add_argument('--top_n_major_consensus', type=int, default=10,
                       help='Number of top responses per question to report based on minimum major segment agreement.')

    # Debug flag
    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')


    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Determine File Paths ---
    std_csv_path = None
    counts_csv_path = None
    output_path = None

    if args.gd_number:
        gd_num = args.gd_number
        gd_identifier = f"GD{gd_num}"
        data_dir = os.path.join("Data", gd_identifier)
        output_base_dir = os.path.join("analysis_output", gd_identifier) # Default output location

        std_csv_path = os.path.join(data_dir, f"{gd_identifier}_aggregate_standardized.csv")
        counts_csv_path = os.path.join(data_dir, f"{gd_identifier}_segment_counts_by_question.csv")
        output_path = args.output_dir if args.output_dir else os.path.join(output_base_dir, "consensus") # Default output subfolder

        logging.info(f"Using GD number {gd_num} to determine paths:")
        # Validate constructed input paths
        if not os.path.exists(std_csv_path):
            parser.error(f"Standardized input file not found for GD{gd_num}. Expected at: {std_csv_path}")
        if not os.path.exists(counts_csv_path):
            parser.error(f"Segment counts input file not found for GD{gd_num}. Expected at: {counts_csv_path}")

    else: # Explicit paths provided
        if not args.standardized_csv or not args.segment_counts_csv or not args.output_dir:
             parser.error("--standardized_csv, --segment_counts_csv, and --output_dir are required when --gd_number is not used.")
        std_csv_path = args.standardized_csv
        counts_csv_path = args.segment_counts_csv
        output_path = args.output_dir # Output dir is required in this case

        logging.info(f"Using explicitly provided paths:")

    logging.info(f"  Standardized Data: {std_csv_path}")
    logging.info(f"  Segment Counts: {counts_csv_path}")
    logging.info(f"  Output Directory: {output_path}")

    os.makedirs(output_path, exist_ok=True)

    # --- Load Data ---
    logging.info(f"Loading standardized data from: {std_csv_path}")
    try:
        standardized_data = pd.read_csv(std_csv_path, low_memory=False)
        logging.info(f"Loaded standardized data with shape: {standardized_data.shape}")
    except FileNotFoundError:
        logging.error(f"Standardized data file not found: {std_csv_path}")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading standardized data: {e}")
        exit(1)

    logging.info(f"Loading segment counts data from: {counts_csv_path}")
    try:
        segment_counts_data = pd.read_csv(counts_csv_path)
        logging.info(f"Loaded segment counts data with shape: {segment_counts_data.shape}")
    except FileNotFoundError:
        logging.error(f"Segment counts data file not found: {counts_csv_path}")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading segment counts data: {e}")
        exit(1)

    # --- Prepare Segment Details Map (for major segment O-code lookup) ---
    # Extract O-code directly from the core segment names in the standardized header
    all_std_columns = standardized_data.columns.tolist()
    base_cols = [
        "Question ID", "Question Type", "Question", "Response", "OriginalResponse",
        "Star", "Categories", "Sentiment", "Submitted By", "Language", "Sample ID", "Participant ID"
    ]
    all_segment_columns_std = [col for col in all_std_columns if col not in base_cols]
    
    segment_details_map = {}
    o_code_pattern = re.compile(r'^O(\d+):') # Pattern to find O-code at the start
    for core_name in all_segment_columns_std:
        o_code = None
        match = o_code_pattern.match(core_name)
        if match:
            o_code = f"O{match.group(1)}" # Construct O-code like O1, O2
        # Store minimal details needed (just o_code) keyed by core name
        segment_details_map[core_name] = {'o_code': o_code}
        
    logging.info(f"Built segment details map for {len(segment_details_map)} segments based on standardized header.")


    # --- Run Analysis Functions ---
    consensus_results = calculate_consensus_profiles(
        standardized_data.copy(), # Pass copy to avoid modifying original
        segment_counts_data.copy(),
        output_path,
        min_segment_size=args.min_segment_size,
        percentiles_to_calc=args.percentiles,
        top_n_percentiles=args.top_n_percentiles,
        top_n_count=args.top_n_count
    )

    major_segment_results = calculate_major_segment_consensus(
        standardized_data.copy(),
        segment_counts_data.copy(),
        segment_details_map, # Pass the map for O-code lookup
        output_path,
        min_segment_size=args.min_segment_size, # Pass base min size
        top_n=args.top_n_major_consensus
    )

    # --- Optional: Summary ---
    if consensus_results is not None and not consensus_results.empty and f'MinAgree_{args.top_n_percentiles[0]}pct' in consensus_results.columns:
         highest_consensus = consensus_results.loc[consensus_results[f'MinAgree_{args.top_n_percentiles[0]}pct'].idxmax()]
         print("\n--- Overall Consensus Summary ---")
         print(f"Highest Consensus Response ({args.top_n_percentiles[0]}th Percentile Minimum: {highest_consensus[f'MinAgree_{args.top_n_percentiles[0]}pct']:.4f}):")
         print(f"  Question : {highest_consensus['Question Text'][:100]}...")
         print(f"  Response : {highest_consensus['Response Text'][:100]}...")
         print(f"  (Based on {highest_consensus['Num Valid Segments']} segments meeting size criteria for its question)")

    if major_segment_results is not None and not major_segment_results.empty:
         highest_min_overall = major_segment_results.loc[major_segment_results['Min Agreement Rate'].idxmax()]
         print("\n--- Major Segment Highest Minimum Summary ---")
         print(f"Highest Minimum Agreement Rate Across Major Segments Overall: {highest_min_overall['Min Agreement Rate']:.4f}")
         print(f"  Question : {highest_min_overall['Question Text'][:100]}...")
         print(f"  Response : {highest_min_overall['Response Text'][:100]}...")

    logging.info("Consensus analysis script finished.")


if __name__ == "__main__":
    main()