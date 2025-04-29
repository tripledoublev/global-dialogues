# Placeholder for consensus calculation script
import argparse
import logging
import os
import pandas as pd
import numpy as np
from lib.analysis_utils import load_standardized_data, parse_percentage, get_segment_columns

# --- Suppress PerformanceWarning --- 
import warnings
from pandas.errors import PerformanceWarning
# Suppress the specific PerformanceWarning related to fragmentation
warnings.filterwarnings('ignore', category=PerformanceWarning)
# --------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_consensus_profiles(questions_data, consensus_output_dir,
                               percentiles_to_calc = [100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10],
                               top_n_percentiles = [100, 95, 90],
                               top_n_count = 5):
    """
    Calculates consensus profiles for Ask Opinion questions and generates reports.

    Args:
        questions_data (list): The list of (metadata, df) tuples.
        consensus_output_dir (str): Directory to save the consensus report CSV files.
        percentiles_to_calc (list): List of percentiles to calculate consensus for.
        top_n_percentiles (list): List of percentiles to show top N responses for.
        top_n_count (int): Number of top responses to show for each percentile.

    Returns:
        pd.DataFrame: DataFrame containing all consensus results.
                    Returns empty DataFrame if no Ask Opinion questions found or processed.
    """
    print("\n--- Calculating Consensus Profiles --- ")
    # Ensure output dir exists
    os.makedirs(consensus_output_dir, exist_ok=True)
    all_consensus_results = []

    for metadata, df in questions_data:
        q_id = metadata.get('id')
        q_text = metadata.get('text')
        q_type = metadata.get('type')

        if q_type != 'Ask Opinion':
            continue

        analysis_segments = metadata.get('analysis_segment_cols', [])

        # Need at least one segment to calculate consensus
        if len(analysis_segments) < 1:
            print(f"  Skipping QID {q_id} (Ask Opinion) - No valid segments for consensus calculation.")
            continue

        # Check if expected columns exist
        response_col = 'English Response' # As per DATA_GUIDE.md
        if response_col not in df.columns:
             # Fallback check (based on notebook, might differ in actual data)
             response_col_fallback = 'English Responses'
             if response_col_fallback in df.columns:
                 response_col = response_col_fallback
             else:
                 # Check if responses are in the standard 'Responses' column
                 if 'Responses' in df.columns:
                     response_col = 'Responses'
                     print(f"  Using 'Responses' column for QID {q_id} as English responses were not found.")
                 else:
                     print(f"  Skipping QID {q_id} - Could not find response column ('{response_col}', '{response_col_fallback}', or 'Responses'). Columns: {df.columns}")
                     continue

        print(f"  Processing QID {q_id} ('{q_text[:50]}...') with {len(analysis_segments)} segments.")
        question_results = []

        # Select only the valid segment columns for calculation
        segment_data = df[analysis_segments].copy()

        # Ensure data is numeric, converting errors to NaN
        for col in analysis_segments:
            # Use parse_percentage from utils for robust conversion
            segment_data[col] = segment_data[col].apply(parse_percentage)
            # Ensure the result is numeric type for calculations
            segment_data[col] = pd.to_numeric(segment_data[col], errors='coerce')

        # Calculate average agreement rate across all segments for each response
        # skipna=True is default, handles rows with *some* valid data
        # Calculate as a Series first
        avg_agreement_series = segment_data.mean(axis=1)
        segment_data['Avg Agreement'] = avg_agreement_series
        
        # Create a copy to potentially reduce fragmentation
        segment_data = segment_data.copy()

        # Filter out rows where Avg Agreement could not be calculated (all segments were NaN)
        valid_avg_agreement = segment_data['Avg Agreement'].dropna()

        if valid_avg_agreement.empty:
            print(f"  Skipping QID {q_id} - No valid average agreement scores calculated (all segment values might be non-numeric).")
            continue

        # Calculate percentiles using only valid scores
        for percentile in percentiles_to_calc:
            # Ensure the array passed to percentile is not empty
            if valid_avg_agreement.empty:
                print(f"  Skipping percentile {percentile} for QID {q_id} - No valid scores remaining after dropna.")
                continue
                
            threshold = np.percentile(valid_avg_agreement, 100 - percentile)
            
            # Get responses above threshold (use original segment_data index)
            high_consensus_indices = valid_avg_agreement[valid_avg_agreement >= threshold].index
            high_consensus = segment_data.loc[high_consensus_indices]
            
            for index, row in high_consensus.iterrows():
                response_text = df.loc[index, response_col]
                question_results.append({
                    'Question ID': q_id,
                    'Question Text': q_text,
                    'Response Text': response_text,
                    'Percentile': percentile,
                    'Avg Agreement': row['Avg Agreement']
                })

        all_consensus_results.extend(question_results)

    if not all_consensus_results:
        print("No responses with consensus found across any Ask Opinion questions.")
        return pd.DataFrame() # Return empty DataFrame

    # Create DataFrame from all results
    results_df = pd.DataFrame(all_consensus_results)
    results_df = results_df.sort_values(by=['Question ID', 'Percentile', 'Avg Agreement'], ascending=[True, False, False])

    # --- Generate Reports ---

    # 1. Top N per Percentile Report
    for percentile in top_n_percentiles:
        top_n = results_df[results_df['Percentile'] == percentile].groupby('Question ID').head(top_n_count)
        report_path = os.path.join(consensus_output_dir, f'consensus_{percentile}th_percentile.csv')
        try:
            top_n.to_csv(report_path, index=False, float_format='%.4f')
            print(f"  Saved {percentile}th percentile consensus report to: {report_path}")
        except Exception as e:
            print(f"  Error saving {percentile}th percentile consensus report: {e}")

    # 2. All Consensus Report
    report_path_all = os.path.join(consensus_output_dir, 'consensus_all.csv')
    try:
        results_df.to_csv(report_path_all, index=False, float_format='%.4f')
        print(f"  Saved all consensus report to: {report_path_all}")
    except Exception as e:
        print(f"  Error saving all consensus report: {e}")

    print("--- Consensus Calculation Complete ---")
    return results_df # Return the full results DataFrame

def calculate_major_segment_consensus(questions_data, major_segment_column_names, output_dir):
    """
    Calculates consensus profiles for major segments (e.g., language, age, gender).

    Args:
        questions_data (list): The list of (metadata, df) tuples.
        major_segment_column_names (list): List of major segment column names to analyze.
        output_dir (str): Directory to save the major segment consensus report CSV files.

    Returns:
        pd.DataFrame: DataFrame containing all major segment consensus results.
                    Returns empty DataFrame if no Ask Opinion questions found or processed.
    """
    print("\n--- Calculating Major Segment Consensus --- ")
    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    all_major_segment_results = []

    for metadata, df in questions_data:
        q_id = metadata.get('id')
        q_text = metadata.get('text')
        q_type = metadata.get('type')

        if q_type != 'Ask Opinion':
            continue

        # Check if expected columns exist
        response_col = 'English Response' # As per DATA_GUIDE.md
        if response_col not in df.columns:
             # Fallback check (based on notebook, might differ in actual data)
             response_col_fallback = 'English Responses'
             if response_col_fallback in df.columns:
                 response_col = response_col_fallback
             else:
                 # Check if responses are in the standard 'Responses' column
                 if 'Responses' in df.columns:
                     response_col = 'Responses'
                 else:
                     print(f"  Skipping QID {q_id} - Could not find response column ('{response_col}', '{response_col_fallback}', or 'Responses'). Columns: {df.columns}")
                     continue

        print(f"  Processing QID {q_id} ('{q_text[:50]}...')")
        question_results = []

        # Select only the major segment columns for calculation
        segment_data = df[major_segment_column_names].copy()

        # Ensure data is numeric, converting errors to NaN
        for col in major_segment_column_names:
            segment_data[col] = pd.to_numeric(segment_data[col], errors='coerce')

        for index, row in segment_data.iterrows():
            response_text = df.loc[index, response_col]
            
            # Calculate average agreement rate for each major segment
            for segment_col in major_segment_column_names:
                agreement_rate = row[segment_col]
                if not pd.isna(agreement_rate):
                    question_results.append({
                        'Question ID': q_id,
                        'Question Text': q_text,
                        'Response Text': response_text,
                        'Segment': segment_col,
                        'Agreement Rate': agreement_rate
                    })

        all_major_segment_results.extend(question_results)

    if not all_major_segment_results:
        print("No major segment consensus found across any Ask Opinion questions.")
        return pd.DataFrame() # Return empty DataFrame

    # Create DataFrame from all results
    results_df = pd.DataFrame(all_major_segment_results)
    results_df = results_df.sort_values(by=['Question ID', 'Segment', 'Agreement Rate'], ascending=[True, True, False])

    # Save report
    report_path = os.path.join(output_dir, 'major_segment_consensus.csv')
    try:
        results_df.to_csv(report_path, index=False, float_format='%.4f')
        print(f"  Saved major segment consensus report to: {report_path}")
    except Exception as e:
        print(f"  Error saving major segment consensus report: {e}")

    print("--- Major Segment Consensus Calculation Complete ---")
    return results_df # Return the full results DataFrame

def main():
    parser = argparse.ArgumentParser(description='Calculate consensus analysis from standardized data.')
    parser.add_argument('standardized_csv', help='Path to the standardized aggregate CSV file.')
    parser.add_argument('output_dir', help='Directory to save consensus output files.')
    parser.add_argument('--percentiles', type=int, nargs='+', default=[100, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10],
                       help='List of percentiles to calculate consensus for.')
    parser.add_argument('--top_n_percentiles', type=int, nargs='+', default=[100, 95, 90],
                       help='List of percentiles to show top N responses for.')
    parser.add_argument('--top_n_count', type=int, default=5,
                       help='Number of top responses to show for each percentile.')
    parser.add_argument('--min_segment_size', type=int, default=15,
                       help='Minimum participant size for a segment to be included in analysis.')
    parser.add_argument('--major_segments', type=str, nargs='+', default=None,
                       help='(Optional) Explicit list of major segment column names to analyze. Overrides dynamic detection.')

    args = parser.parse_args()

    logging.info(f"Starting consensus analysis using {args.standardized_csv}")
    logging.info(f"Output will be saved to: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and preprocess data
    data = load_standardized_data(args.standardized_csv)
    if data is not None:
        # --- Determine Segment Columns and Details from Loaded Data --- 
        all_columns = data.columns.tolist()
        all_segment_cols, segment_details = get_segment_columns(all_columns)
        
        # --- Dynamically Determine Major Segments (if not provided via args) ---
        major_segment_column_names_to_use = []
        if args.major_segments:
            logging.info(f"Using user-provided major segments: {args.major_segments}")
            # Basic validation: Check if provided segments exist in the data
            major_segment_column_names_to_use = [col for col in args.major_segments if col in all_columns]
            missing_major = [col for col in args.major_segments if col not in all_columns]
            if missing_major:
                logging.warning(f"User-provided major segments not found in data columns: {missing_major}")
            if not major_segment_column_names_to_use:
                 logging.error("None of the user-provided major segments were found in the data. Aborting major segment calculation.")
        elif segment_details:
            logging.info("Dynamically determining major segments based on criteria...")
            # Find the 'All' segment to get total N for dynamic threshold (optional but good)
            all_segment_info = next((details for col, details in segment_details.items() if col.lower().startswith('all(')), None)
            total_N = 0
            min_threshold = args.min_segment_size # Use arg as the base min size

            if all_segment_info and 'size' in all_segment_info and pd.notna(all_segment_info['size']):
                total_N = all_segment_info['size']
                # Calculate dynamic threshold: max of arg or 1% of total N
                dynamic_threshold = max(1, round(total_N / 100.0)) # Ensure at least 1
                min_threshold = max(args.min_segment_size, dynamic_threshold)
                logging.info(f"  Using Total N = {total_N}. Calculated min size threshold for major segments: {min_threshold} (max({args.min_segment_size}, {dynamic_threshold}))")
            else:
                logging.warning(f"  Could not determine Total N from 'All' segment. Using min_segment_size ({args.min_segment_size}) for major segment filtering.")
            
            for col_name, details in segment_details.items():
                # Exclude 'All' segment itself
                if col_name.lower().startswith('all('):
                    continue
                o_code = details.get('o_code')
                size = details.get('size')
                # Criteria: Not O1 (Language), Not O7 (Country), Size >= threshold
                if o_code not in ['O1', 'O7'] and pd.notna(size) and size >= min_threshold:
                    major_segment_column_names_to_use.append(col_name)
            logging.info(f"  Identified {len(major_segment_column_names_to_use)} major segments meeting criteria: {major_segment_column_names_to_use}")
            if not major_segment_column_names_to_use:
                 logging.warning("  No segments met the criteria to be considered 'major' for the major segment consensus report.")
        else:
             logging.warning("Could not determine major segments: No segment details found in data.")

        # Process data into the expected format (list of (metadata, df) tuples)
        # Pass the *full* list of segment columns found in the data to metadata
        questions_data = []
        for q_id, group in data.groupby('Question ID'):
            # Get segment columns specifically present in *this group's* columns
            group_cols = group.columns.tolist()
            group_segment_cols, _ = get_segment_columns(group_cols) # Ignore details here
            metadata = {
                'id': q_id,
                'type': group['Question Type'].iloc[0],
                'text': group['Question'].iloc[0],
                'analysis_segment_cols': group_segment_cols # Use segments actually in this group for general analysis
            }
            questions_data.append((metadata, group))
        
        # Calculate consensus profiles (uses 'analysis_segment_cols' from metadata)
        consensus_results = calculate_consensus_profiles(
            questions_data,
            args.output_dir,
            percentiles_to_calc=args.percentiles,
            top_n_percentiles=args.top_n_percentiles,
            top_n_count=args.top_n_count
        )
        
        # Calculate major segment consensus (uses the dynamically determined or user-provided list)
        if major_segment_column_names_to_use: # Only run if we have major segments
            major_segment_results = calculate_major_segment_consensus(
                questions_data,
                major_segment_column_names_to_use, # Pass the determined list
                args.output_dir
            )
        else:
            logging.info("Skipping Major Segment Consensus calculation as no major segments were identified or provided.")
            major_segment_results = pd.DataFrame() # Ensure it exists as an empty DF
        
        # --- Summary --- (No change needed here)
        if not consensus_results.empty:
            highest_consensus = consensus_results.loc[consensus_results['Avg Agreement'].idxmax()]
            print("\n--- Overall Summary ---")
            print(f"Highest Consensus Response (Avg Agreement: {highest_consensus['Avg Agreement']:.4f}):")
            print(f"  Question : {highest_consensus['Question Text'][:100]}...")
            print(f"  Response : {highest_consensus['Response Text'][:100]}...")
            print(f"  Percentile: {highest_consensus['Percentile']}th")
        
        logging.info("Consensus analysis complete.")
    else:
        logging.error("Failed to load data, aborting consensus analysis.")

if __name__ == "__main__":
    main() 