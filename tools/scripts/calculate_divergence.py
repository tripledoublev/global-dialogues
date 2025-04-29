# Placeholder for divergence calculation script
import argparse
import logging
import os
import math
import pandas as pd
import numpy as np
from lib.analysis_utils import load_standardized_data, parse_percentage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def main():
    parser = argparse.ArgumentParser(description='Calculate divergence analysis from standardized data.')
    parser.add_argument('standardized_csv', help='Path to the standardized aggregate CSV file.')
    parser.add_argument('output_dir', help='Directory to save divergence output files.')
    parser.add_argument('--top_n_per_question', type=int, default=20,
                       help='Number of top divergent responses to show per question.')
    parser.add_argument('--top_n_overall', type=int, default=50,
                       help='Number of top divergent responses to show overall.')
    parser.add_argument('--min_segment_size', type=int, default=15,
                       help='Minimum participant size for a segment to be included in analysis.')

    args = parser.parse_args()

    logging.info(f"Starting divergence analysis using {args.standardized_csv}")
    logging.info(f"Output will be saved to: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and preprocess data
    data = load_standardized_data(args.standardized_csv)
    if data is not None:
        # Process data into the expected format (list of (metadata, df) tuples)
        questions_data = []
        for q_id, group in data.groupby('Question ID'):
            metadata = {
                'id': q_id,
                'type': group['Question Type'].iloc[0],
                'text': group['Question'].iloc[0],
                'analysis_segment_cols': [col for col in group.columns if col.startswith('All(') or col.startswith('O') and '(' in col]
            }
            questions_data.append((metadata, group))
        
        # Calculate divergence
        results_df = calculate_divergence_report(
            questions_data,
            args.output_dir,
            top_n_per_question=args.top_n_per_question,
            top_n_overall=args.top_n_overall
        )
        
        if not results_df.empty:
            # Print summary of most divergent response
            most_divergent = results_df.loc[results_df['Divergence Score'].idxmax()]
            print("\n--- Overall Summary ---")
            print(f"Most Divergent Response Overall (Score: {most_divergent['Divergence Score']:.4f}):")
            print(f"  Question : {most_divergent['Question Text'][:100]}...")
            print(f"  Response : {most_divergent['Response Text'][:100]}...")
            print(f"  Segments : {most_divergent['Min Segment']} ({most_divergent['Min Agreement']:.1%}) vs {most_divergent['Max Segment']} ({most_divergent['Max Agreement']:.1%})")
        
        logging.info("Divergence analysis complete.")
    else:
        logging.error("Failed to load data, aborting divergence analysis.")

if __name__ == "__main__":
    main() 