# divergence calculation script
import argparse
import logging
import os
import math
import pandas as pd
import numpy as np
from lib.analysis_utils import parse_percentage # Keep parse_percentage

# --- Suppress PerformanceWarning if needed ---
import warnings
from pandas.errors import PerformanceWarning
warnings.filterwarnings('ignore', category=PerformanceWarning)
# --------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_divergence_report(standardized_df, segment_counts_df, output_dir,
                                min_segment_size, # Now required for per-question filtering
                                top_n_per_question=20, top_n_overall=50):
    """
    Calculates divergence for Ask Opinion questions using standardized data.
    Filters segments per question based on counts from segment_counts_df.

    Args:
        standardized_df (pd.DataFrame): DataFrame from _aggregate_standardized.csv.
        segment_counts_df (pd.DataFrame): DataFrame from _segment_counts_by_question.csv.
        output_dir (str): Directory to save the divergence report CSV files.
        min_segment_size (int): Minimum participant count for a segment to be included.
        top_n_per_question (int): Number of top divergent responses to show per question.
        top_n_overall (int): Number of top divergent responses to show overall.

    Returns:
        pd.DataFrame: DataFrame containing all divergence results (score > 0).
                    Returns empty DataFrame if no Ask Opinion questions found or processed.
    """
    print("\n--- Calculating Divergence Report (using standardized data) --- ")
    os.makedirs(output_dir, exist_ok=True)
    all_divergence_results = []

    # Identify all potential segment columns from the standardized data
    base_cols = ["Question ID", "Question Type", "Question", "Response", "OriginalResponse",
                 "Star", "Categories", "Sentiment", "Submitted By", "Language", "Sample ID", "Participant ID"]
    all_segment_columns = [col for col in standardized_df.columns if col not in base_cols]

    if not all_segment_columns:
         print("  Error: No segment columns found in the standardized data header.")
         return pd.DataFrame()

    print(f"  Identified {len(all_segment_columns)} potential segment columns in standardized data.")

    # Pre-process segment counts for easier lookup
    if segment_counts_df.index.name != 'Question ID':
        segment_counts_df = segment_counts_df.set_index('Question ID')
    # Convert counts columns to numeric, coercing errors to NaN
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
             
        # Need at least two segments to calculate divergence
        if len(valid_segments_for_q) < 2:
            print(f"  Skipping QID {q_id} (Ask Opinion) - Fewer than 2 segments met min size ({min_segment_size}) for divergence calculation.")
            continue

        print(f"  Processing QID {q_id} ('{q_text[:50]}...') with {len(valid_segments_for_q)} valid segments (>= {min_segment_size} participants).")
        question_results = []

        # Select only the valid segment columns for calculation from this group
        segment_data = group[valid_segments_for_q].copy()

        # Parse percentages and ensure numeric type
        for col in valid_segments_for_q:
            segment_data[col] = segment_data[col].apply(parse_percentage)
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
                min_segment = valid_rates.idxmin() if pd.notna(min_rate) else 'N/A'
                max_segment = valid_rates.idxmax() if pd.notna(max_rate) else 'N/A'

                # Get response text using the original DataFrame index (from the group)
                response_text = group.loc[index, 'Response'] # Use standard name

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
        print(f"  Saved divergence per question report to: {report_path_per_q}")
    except Exception as e:
        print(f"  Error saving per-question divergence report: {e}")

    # 2. Top N Overall Report
    top_overall = results_df.head(top_n_overall)
    report_path_overall = os.path.join(output_dir, 'divergence_overall.csv')
    try:
        top_overall.to_csv(report_path_overall, index=False, float_format='%.4f')
        print(f"  Saved overall divergence report to: {report_path_overall}")
    except Exception as e:
        print(f"  Error saving overall divergence report: {e}")

    print("--- Divergence Calculation Complete ---")
    return results_df # Return the full results DataFrame

def main():
    parser = argparse.ArgumentParser(description='Calculate divergence analysis from standardized data.')
    
    # Input specification
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--gd_number", type=int, help="Global Dialogue cadence number (e.g., 1, 2, 3). Constructs default paths.")
    input_group.add_argument("--standardized_csv", help="Explicit path to the standardized aggregate CSV file.")

    # Conditionally required segment counts path
    parser.add_argument('--segment_counts_csv', help='Path to the segment counts per question CSV file (required if --standardized_csv is used).')
    
    # Output directory
    parser.add_argument('-o', '--output_dir', help='Directory to save divergence output files (required if --standardized_csv is used).')

    # Analysis parameters
    parser.add_argument('--top_n_per_question', type=int, default=20,
                       help='Number of top divergent responses to show per question.')
    parser.add_argument('--top_n_overall', type=int, default=50,
                       help='Number of top divergent responses to show overall.')
    parser.add_argument('--min_segment_size', type=int, default=15,
                       help='Minimum participant size for a segment to be included in analysis (per question).')
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
        output_base_dir = os.path.join("analysis_output", gd_identifier) 

        std_csv_path = os.path.join(data_dir, f"{gd_identifier}_aggregate_standardized.csv")
        counts_csv_path = os.path.join(data_dir, f"{gd_identifier}_segment_counts_by_question.csv")
        output_path = args.output_dir if args.output_dir else os.path.join(output_base_dir, "divergence") # Default output subfolder

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
        output_path = args.output_dir 
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
        logging.error(f"Standardized data file not found: {std_csv_path}"); exit(1)
    except Exception as e:
        logging.error(f"Error loading standardized data: {e}"); exit(1)

    logging.info(f"Loading segment counts data from: {counts_csv_path}")
    try:
        segment_counts_data = pd.read_csv(counts_csv_path)
        logging.info(f"Loaded segment counts data with shape: {segment_counts_data.shape}")
    except FileNotFoundError:
        logging.error(f"Segment counts data file not found: {counts_csv_path}"); exit(1)
    except Exception as e:
        logging.error(f"Error loading segment counts data: {e}"); exit(1)

    # --- Calculate Divergence --- 
    results_df = calculate_divergence_report(
        standardized_data.copy(), # Pass copy to avoid modifying original
        segment_counts_data.copy(),
        output_path, 
        min_segment_size=args.min_segment_size,
        top_n_per_question=args.top_n_per_question,
        top_n_overall=args.top_n_overall
    )
        
    # --- Summary --- 
    if results_df is not None and not results_df.empty:
        most_divergent = results_df.iloc[0] # Already sorted descending
        print("\n--- Overall Divergence Summary ---")
        print(f"Most Divergent Response Overall (Score: {most_divergent['Divergence Score']:.4f}):")
        print(f"  Question : {most_divergent['Question Text'][:100]}...")
        print(f"  Response : {most_divergent['Response Text'][:100]}...")
        print(f"  Segments : {most_divergent['Min Segment']} ({most_divergent['Min Agreement']:.1%}) vs {most_divergent['Max Segment']} ({most_divergent['Max Agreement']:.1%})")
        
    logging.info("Divergence analysis script finished.")

if __name__ == "__main__":
    main() 