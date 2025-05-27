#!/usr/bin/env python3
"""
Export Unreliable Participants Script

This script creates CSV files of unreliable participants based on existing PRI scores,
without recalculating the entire PRI analysis. It extracts their open-ended responses
for manual review.

Usage:
    python export_unreliable_participants.py <gd_number> [--method METHOD] [--threshold THRESHOLD] [--debug]

Arguments:
    gd_number   The Global Dialogue number (e.g., 1, 2, 3)
    --method    Method to identify unreliable participants: 'outliers', 'percentile', 'threshold' (default: outliers)
    --threshold Threshold value for percentile or threshold methods (default: 10 for percentile, 2.5 for threshold)
    --debug     Enable verbose debug output

Output:
    CSV file with unreliable participant IDs, PRI scores, and all open-ended responses
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Export unreliable participants based on existing PRI scores")
    parser.add_argument("gd_number", type=int, help="Global Dialogue number (e.g., 1, 2, 3)")
    parser.add_argument("--method", type=str, default="outliers", 
                       choices=["outliers", "percentile", "threshold"],
                       help="Method to identify unreliable participants (default: outliers)")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Threshold value (default: 10 for percentile, 2.5 for threshold)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    return parser.parse_args()


def get_config(gd_number):
    """Get file paths for the specified Global Dialogue"""
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "Data" / f"GD{gd_number}"
    output_dir = base_dir / "analysis_output" / f"GD{gd_number}" / "pri"
    
    # Verify directories exist
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    config = {
        'DATA_DIR': str(data_dir),
        'OUTPUT_DIR': str(output_dir),
        'PRI_SCORES_PATH': str(output_dir / f"GD{gd_number}_pri_scores.csv"),
        'VERBATIM_MAP_PATH': str(data_dir / f"GD{gd_number}_verbatim_map.csv"),
        'DISCUSSION_GUIDE_PATH': str(data_dir / f"GD{gd_number}_discussion_guide.csv"),
    }
    
    # Verify required files exist
    for key, path in config.items():
        if key.endswith('_PATH') and not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
    
    return config


def extract_open_ended_responses(verbatim_map_df, discussion_guide_df, participant_ids, debug=False):
    """
    Extract all open-ended (Ask Opinion, Ask Experience) responses for specific participants.
    
    Args:
        verbatim_map_df: DataFrame with participant responses
        discussion_guide_df: DataFrame with question types  
        participant_ids: List of participant IDs to extract responses for
        debug: Whether to print debug information
        
    Returns:
        DataFrame with participant responses pivoted into columns
    """
    if debug:
        print(f"Extracting open-ended responses for {len(participant_ids)} participants...")
    
    # Find Ask Opinion and Ask Experience questions from discussion guide
    open_ended_questions = discussion_guide_df[
        discussion_guide_df['Item type (dropdown)'].str.contains('ask opinion|ask experience', case=False, na=False)
    ]['Cross Conversation Tag - Polls and Opinions only (Optional)'].tolist()
    
    if debug:
        print(f"Found {len(open_ended_questions)} open-ended questions from discussion guide: {open_ended_questions}")
    
    # Get unique question IDs from verbatim map to see what's actually available
    available_question_ids = verbatim_map_df['Question ID'].unique()
    if debug:
        print(f"Available question IDs in verbatim map: {len(available_question_ids)} total")
        print(f"Sample available question IDs: {list(available_question_ids)[:10]}")
    
    # Find intersection of expected open-ended questions and available questions
    matching_questions = [q for q in open_ended_questions if q in available_question_ids]
    if debug:
        print(f"Matching open-ended questions: {matching_questions}")
    
    # If no matches, use a broader approach - get all responses for participants and filter by question text patterns
    if len(matching_questions) == 0:
        if debug:
            print("No direct question ID matches found. Trying to match by question text patterns...")
        
        # Filter responses for target participants first
        participant_responses = verbatim_map_df[
            verbatim_map_df['Participant ID'].isin(participant_ids)
        ].copy()
        
        # Look for question texts that indicate open-ended questions
        open_ended_patterns = [
            'explain why', 'can you explain', 'please explain', 'would you want', 
            'do you think', 'would you prefer', 'what has been', 'is there anything'
        ]
        
        mask = participant_responses['Question Text'].str.contains(
            '|'.join(open_ended_patterns), case=False, na=False
        )
        
        responses_df = participant_responses[mask].copy()
        
        if debug:
            print(f"Found {len(responses_df)} responses using text pattern matching")
            if len(responses_df) > 0:
                print(f"Sample question texts: {responses_df['Question Text'].unique()[:3]}")
    else:
        # Use the matching question IDs
        responses_df = verbatim_map_df[
            (verbatim_map_df['Participant ID'].isin(participant_ids)) &
            (verbatim_map_df['Question ID'].isin(matching_questions))
        ].copy()
        
        if debug:
            print(f"Found {len(responses_df)} responses using question ID matching")
    
    if len(responses_df) == 0:
        if debug:
            print("No open-ended responses found for target participants")
        return pd.DataFrame()
    
    # Create a mapping of Question ID to Question Text for column names
    question_mapping = dict(zip(responses_df['Question ID'], responses_df['Question Text']))
    
    # Group by participant and question, concatenate multiple thoughts if any
    grouped_responses = responses_df.groupby(['Participant ID', 'Question ID'])['Thought Text'].agg(
        lambda x: ' | '.join(x.astype(str)) if len(x) > 1 else x.iloc[0] if len(x) == 1 else ''
    ).reset_index()
    
    # Pivot to get one column per question
    pivoted_df = grouped_responses.pivot(index='Participant ID', columns='Question ID', values='Thought Text')
    
    # Rename columns to use question text instead of question ID
    pivoted_df.columns = [question_mapping.get(col, col) for col in pivoted_df.columns]
    
    # Reset index to make Participant ID a regular column
    pivoted_df = pivoted_df.reset_index()
    
    # Fill NaN values with empty strings
    pivoted_df = pivoted_df.fillna('')
    
    if debug:
        print(f"Extracted responses for {len(pivoted_df)} participants across {len(pivoted_df.columns)-1} questions")
    
    return pivoted_df


def identify_unreliable_participants(pri_scores_df, method='outliers', threshold=None, debug=False):
    """
    Identify participants who should be considered unreliable based on PRI scores.
    
    Args:
        pri_scores_df: DataFrame with calculated PRI scores
        method: Method to use ('outliers', 'percentile', 'threshold')
        threshold: Specific threshold value (used with 'threshold' or 'percentile' methods)
        debug: Whether to print debug information
        
    Returns:
        Tuple of (unreliable_participant_ids, effective_threshold_used)
    """
    valid_scores = pri_scores_df['PRI_Scale_1_5'].dropna()
    
    if len(valid_scores) == 0:
        print("Warning: No valid PRI scores to analyze")
        return [], None
    
    unreliable_participants = []
    effective_threshold = None
    
    if method == 'outliers':
        # Use boxplot outlier calculation (1.5 * IQR rule)
        q1 = valid_scores.quantile(0.25)
        q3 = valid_scores.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Identify outliers (focusing on low scores as unreliable)
        outlier_mask = (pri_scores_df['PRI_Scale_1_5'] < lower_bound) | (pri_scores_df['PRI_Scale_1_5'] > upper_bound)
        low_outlier_mask = pri_scores_df['PRI_Scale_1_5'] < lower_bound
        
        unreliable_participants = pri_scores_df[low_outlier_mask]['Participant ID'].tolist()
        
        # Set effective threshold to the lower bound (cutoff for outliers)
        effective_threshold = lower_bound
        
        if debug:
            print(f"Outlier analysis: Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f}")
            print(f"Lower bound: {lower_bound:.3f}, Upper bound: {upper_bound:.3f}")
            print(f"Total outliers: {outlier_mask.sum()}, Low outliers (unreliable): {low_outlier_mask.sum()}")
            print(f"Effective threshold (lower bound): {effective_threshold:.3f}")
    
    elif method == 'percentile':
        if threshold is None:
            threshold = 10  # Default to bottom 10th percentile
        
        percentile_threshold = valid_scores.quantile(threshold / 100)
        unreliable_mask = pri_scores_df['PRI_Scale_1_5'] <= percentile_threshold
        unreliable_participants = pri_scores_df[unreliable_mask]['Participant ID'].tolist()
        
        effective_threshold = percentile_threshold
        
        if debug:
            print(f"Percentile analysis: Bottom {threshold}th percentile threshold = {percentile_threshold:.3f}")
            print(f"Participants below threshold: {unreliable_mask.sum()}")
    
    elif method == 'threshold':
        if threshold is None:
            threshold = 2.5  # Default threshold for "Low Reliability" 
        
        unreliable_mask = pri_scores_df['PRI_Scale_1_5'] <= threshold
        unreliable_participants = pri_scores_df[unreliable_mask]['Participant ID'].tolist()
        
        effective_threshold = threshold
        
        if debug:
            print(f"Hard threshold analysis: Threshold = {threshold}")
            print(f"Participants below threshold: {unreliable_mask.sum()}")
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'outliers', 'percentile', or 'threshold'")
    
    if debug:
        print(f"Identified {len(unreliable_participants)} unreliable participants using method '{method}'")
    
    return unreliable_participants, effective_threshold


def main():
    """Main execution function"""
    # Parse command-line arguments
    args = parse_args()
    gd_number = args.gd_number
    method = args.method
    threshold = args.threshold
    debug = args.debug
    
    print(f"Exporting unreliable participants for Global Dialogue {gd_number}")
    print(f"Method: {method}")
    if threshold is not None:
        print(f"Threshold: {threshold}")
    print(f"Debug mode: {'Enabled' if debug else 'Disabled'}")
    
    # Get configuration for this GD
    try:
        config = get_config(gd_number)
    except Exception as e:
        print(f"Error in configuration: {e}")
        sys.exit(1)
    
    # 1. Load existing PRI scores
    try:
        print(f"Loading PRI scores from {config['PRI_SCORES_PATH']}...")
        pri_scores_df = pd.read_csv(config['PRI_SCORES_PATH'])
        print(f"Loaded PRI scores for {len(pri_scores_df)} participants")
    except Exception as e:
        print(f"Error loading PRI scores: {e}")
        sys.exit(1)
    
    # 2. Load verbatim map data
    try:
        print(f"Loading verbatim responses from {config['VERBATIM_MAP_PATH']}...")
        verbatim_map_df = pd.read_csv(config['VERBATIM_MAP_PATH'], quotechar='"', engine='python')
        print(f"Loaded verbatim map data with shape: {verbatim_map_df.shape}")
    except Exception as e:
        print(f"Error loading verbatim map data: {e}")
        sys.exit(1)
    
    # 3. Load discussion guide
    try:
        print(f"Loading discussion guide from {config['DISCUSSION_GUIDE_PATH']}...")
        discussion_guide_df = pd.read_csv(config['DISCUSSION_GUIDE_PATH'], encoding='utf-8-sig', quotechar='"', 
                                        engine='python', on_bad_lines='skip')
        print(f"Loaded discussion guide with {len(discussion_guide_df)} questions")
    except Exception as e:
        print(f"Error loading discussion guide: {e}")
        sys.exit(1)
    
    # 4. Identify unreliable participants
    unreliable_participant_ids, effective_threshold = identify_unreliable_participants(
        pri_scores_df, method=method, threshold=threshold, debug=debug
    )
    
    if len(unreliable_participant_ids) == 0:
        print("No unreliable participants identified.")
        return
    
    # 5. Get PRI score information for unreliable participants
    # Include final normalized component scores and PRI scores
    # Check if LLM judge columns are available
    base_columns = ['Participant ID', 'Duration_Norm', 'LowQualityTag_Norm', 
                   'UniversalDisagreement_Norm', 'ASC_Norm', 'PRI_Scale_1_5']
    
    # Look for LLM judge columns (they might have different names)
    available_columns = pri_scores_df.columns.tolist()
    llm_columns = [col for col in available_columns if 'llm' in col.lower() and 'norm' in col.lower()]
    
    if llm_columns:
        columns_to_include = base_columns[:-1] + llm_columns + [base_columns[-1]]  # Insert LLM columns before PRI_Scale_1_5
        if debug:
            print(f"Found LLM judge columns: {llm_columns}")
    else:
        columns_to_include = base_columns
        if debug:
            print("No LLM judge columns found in PRI scores")
    
    unreliable_pri_df = pri_scores_df[
        pri_scores_df['Participant ID'].isin(unreliable_participant_ids)
    ][columns_to_include].copy()
    
    # 6. Extract open-ended responses for these participants
    responses_df = extract_open_ended_responses(
        verbatim_map_df, discussion_guide_df, unreliable_participant_ids, debug=debug
    )
    
    # 7. Merge PRI scores with responses
    if len(responses_df) > 0:
        final_df = unreliable_pri_df.merge(responses_df, on='Participant ID', how='left')
    else:
        final_df = unreliable_pri_df
        print("Warning: No open-ended responses found for unreliable participants")
    
    # 8. Sort by PRI score (lowest first)
    final_df = final_df.sort_values('PRI_Scale_1_5', ascending=True)
    
    # 9. Add metadata columns
    final_df.insert(0, 'Recommended_Action', 'IGNORE')
    final_df.insert(1, 'Identification_Method', method)
    if effective_threshold is not None:
        final_df.insert(2, 'Threshold_Used', round(effective_threshold, 3))
    else:
        final_df.insert(2, 'Threshold_Used', 'N/A')
    
    # 10. Generate output filename
    if method == 'outliers':
        output_filename = f"GD{gd_number}_unreliable_participants_outliers.csv"
    elif method == 'percentile':
        thresh_str = f"{threshold}pct" if threshold else "10pct"
        output_filename = f"GD{gd_number}_unreliable_participants_bottom{thresh_str}.csv"
    elif method == 'threshold':
        thresh_str = f"{threshold}" if threshold else "2.5"
        output_filename = f"GD{gd_number}_unreliable_participants_threshold{thresh_str}.csv"
    
    output_path = os.path.join(config['OUTPUT_DIR'], output_filename)
    
    # 11. Export to CSV
    final_df.to_csv(output_path, index=False)
    
    print(f"\nExported {len(final_df)} unreliable participants to {output_path}")
    print(f"Columns in export: {list(final_df.columns)}")
    
    # 12. Print summary statistics
    print(f"\nSummary:")
    print(f"- Total participants with PRI scores: {len(pri_scores_df)}")
    print(f"- Unreliable participants identified: {len(final_df)}")
    print(f"- Percentage identified as unreliable: {len(final_df)/len(pri_scores_df)*100:.1f}%")
    print(f"- PRI score range for unreliable: {final_df['PRI_Scale_1_5'].min():.3f} - {final_df['PRI_Scale_1_5'].max():.3f}")


if __name__ == "__main__":
    main()