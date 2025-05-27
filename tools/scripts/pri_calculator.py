#!/usr/bin/env python3
"""
Participant Reliability Index (PRI) Calculator

This script calculates a reliability score for each participant in a Global Dialogue survey,
combining multiple signals to create a comprehensive measure of participant engagement 
and response quality.

Usage:
    python pri_calculator.py <gd_number> [--debug] [--limit N]

Arguments:
    gd_number   The Global Dialogue number (e.g., 1, 2, 3)
    --debug     Enable verbose debug output
    --limit     Limit processing to first N participants (for testing)

Output:
    CSV file with participant IDs and calculated metrics
"""

import pandas as pd
import numpy as np
import argparse
import time
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Calculate Participant Reliability Index (PRI) scores.')
    parser.add_argument('gd_number', type=int, help='The Global Dialogue number (e.g. 1, 2, 3)')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debug output')
    parser.add_argument('--limit', type=int, help='Limit processing to first N participants (for testing)', default=None)
    return parser.parse_args()


def get_config(gd_number):
    """Define file paths and PRI parameters based on GD number."""
    data_dir = Path(f"Data/GD{gd_number}")
    tags_dir = data_dir / "tags"
    
    # Ensure directories exist
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # File paths
    config = {
        'DATA_DIR': str(data_dir),
        'TAGS_DIR': str(tags_dir),
        'VERBATIM_MAP_PATH': str(data_dir / f"GD{gd_number}_verbatim_map.csv"),
        'BINARY_PATH': str(data_dir / f"GD{gd_number}_binary.csv"),
        'PREFERENCE_PATH': str(data_dir / f"GD{gd_number}_preference.csv"),
        'AGGREGATE_STD_PATH': str(data_dir / f"GD{gd_number}_aggregate_standardized.csv"),
        'SEGMENT_COUNTS_PATH': str(data_dir / f"GD{gd_number}_segment_counts_by_question.csv"),
        'THOUGHT_LABELS_PATH': str(tags_dir / "all_thought_labels.csv"),
        'OUTPUT_PATH': str(data_dir / f"GD{gd_number}_pri_scores.csv"),
        
        # PRI Parameters (per documentation)
        'ASC_HIGH_THRESHOLD': 0.70,                         # Agreement rate for strong agreement
        'ASC_LOW_THRESHOLD': 0.30,                          # Agreement rate for strong disagreement
        'UNIVERSAL_DISAGREEMENT_THRESHOLD_ALL': 0.30,       # Max agreement for "disagreed" responses for 'All'
        'UNIVERSAL_DISAGREEMENT_THRESHOLD_SEGMENTS': 0.40,  # Max agreement for "disagreed" responses for all individual Segments
        'MAJOR_SEGMENT_MIN_PARTICIPANTS': 20,               # Min participants required for a segment to be considered "major"
        'DURATION_REASONABLE_MAX': 60*90,                   # Reasonable max duration to complete survey in seconds

        
        # Component weights for final PRI score
        'DURATION_WEIGHT': 0.30,
        'LOW_QUALITY_TAG_WEIGHT': 0.30,
        'UNIVERSAL_DISAGREEMENT_WEIGHT': 0.20,
        'ASC_WEIGHT': 0.20,
    }
    
    return config


def load_data(config, debug=False):
    """
    Load and preprocess all necessary data files.
    
    Returns:
        Tuple of DataFrames, participant IDs, and major segments:
        (binary_df, preference_df, thought_labels_df, verbatim_map_df, aggregate_std_df, all_participant_ids, major_segments)
    """
    print(f"Loading data from {config['DATA_DIR']}...")
    
    # Set pandas display options for debugging
    if debug:
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.width', 120)
    
    # 1. Load binary vote data
    try:
        binary_df = pd.read_csv(config['BINARY_PATH'], quotechar='"', low_memory=False)
        
        # Convert timestamps to datetime
        if 'Timestamp' in binary_df.columns:
            binary_df['Timestamp'] = pd.to_datetime(
                binary_df['Timestamp'], 
                format='%B %d, %Y at %I:%M %p (GMT)',
                errors='coerce'
            )
        
        # Normalize vote values
        if 'Vote' in binary_df.columns:
            binary_df['VoteNumeric'] = binary_df['Vote'].map(
                {'Agree': 1, 'agree': 1, 'Disagree': 0, 'disagree': 0}
            ).astype(float)
        
        print(f"Loaded binary data with shape: {binary_df.shape}")
    except Exception as e:
        print(f"Error loading binary data: {e}")
        sys.exit(1)
    
    # 2. Load preference judgment data
    try:
        preference_df = pd.read_csv(config['PREFERENCE_PATH'], quotechar='"', low_memory=False)
        
        # Convert timestamps to datetime
        if 'Timestamp' in preference_df.columns:
            preference_df['Timestamp'] = pd.to_datetime(
                preference_df['Timestamp'], 
                format='%B %d, %Y at %I:%M %p (GMT)',
                errors='coerce'
            )
        
        print(f"Loaded preference data with shape: {preference_df.shape}")
    except Exception as e:
        print(f"Error loading preference data: {e}")
        sys.exit(1)
    
    # 3. Load thought labels data (for quality tags)
    try:
        thought_labels_df = pd.read_csv(config['THOUGHT_LABELS_PATH'], encoding='utf-8-sig')
        print(f"Loaded thought labels data with shape: {thought_labels_df.shape}")
    except Exception as e:
        print(f"Error loading thought labels data: {e}")
        # Non-fatal error - can continue without tags
        thought_labels_df = pd.DataFrame(columns=['Participant ID', 'Question ID'])
    
    # 4. Load verbatim map data
    try:
        verbatim_map_df = pd.read_csv(config['VERBATIM_MAP_PATH'], quotechar='"', engine='python')
        print(f"Loaded verbatim map data with shape: {verbatim_map_df.shape}")
    except Exception as e:
        print(f"Error loading verbatim map data: {e}")
        sys.exit(1)
    
    # 5. Load aggregate standardized data
    try:
        # Suppress dtype warnings for mixed columns
        with pd.option_context('mode.chained_assignment', None):
            aggregate_std_df = pd.read_csv(config['AGGREGATE_STD_PATH'], low_memory=False)
            
            # Convert percentage columns to numeric values
            if 'All' in aggregate_std_df.columns:
                aggregate_std_df['All_Agreement'] = aggregate_std_df['All'].apply(
                    lambda x: parse_percentage(x, debug)
                )
            
            # Load major segments from segment counts file
            major_segments = load_major_segments(config, debug)
            
            # Convert percentage columns to numeric values for major segment columns
            segment_cols = [col for col in aggregate_std_df.columns if col in major_segments]
            if not segment_cols:
                if debug:
                    print(f"No major segment columns found for analysis")
            else:
                for col in segment_cols:
                    segment_agreement_col = f'{col}_Agreement'
                    aggregate_std_df[segment_agreement_col] = aggregate_std_df[col].apply(lambda x: parse_percentage(x, debug))
                    
                if debug:
                    print(f"Processed {len(segment_cols)} major segment columns for agreement calculation")

            print(f"Loaded aggregate data with shape: {aggregate_std_df.shape}")
    except Exception as e:
        print(f"Error loading aggregate data: {e}")
        sys.exit(1)
    
    # Get unique participant IDs from binary votes
    all_participant_ids = binary_df['Participant ID'].unique()
    print(f"Found {len(all_participant_ids)} unique participants")
    
    # Data validation checks
    if debug:
        # Validate thought IDs exist in both files
        verbatim_thoughts = set(verbatim_map_df['Thought ID']) if 'Thought ID' in verbatim_map_df.columns else set()
        aggregate_thoughts = set(aggregate_std_df['Thought ID']) if 'Thought ID' in aggregate_std_df.columns else set()
        missing_thoughts = verbatim_thoughts - aggregate_thoughts
        if missing_thoughts:
            print(f"Warning: {len(missing_thoughts)} thoughts in verbatim_map not found in aggregate data")
        
        # Handle participants with no authored thoughts
        if 'Participant ID' in verbatim_map_df.columns:
            participant_thought_counts = verbatim_map_df['Participant ID'].value_counts()
            zero_thought_participants = [pid for pid in all_participant_ids 
                                       if pid not in participant_thought_counts.index]
            if zero_thought_participants:
                print(f"Info: {len(zero_thought_participants)} participants have no authored thoughts")
    
    return (
        binary_df, preference_df, thought_labels_df, 
        verbatim_map_df, aggregate_std_df, all_participant_ids, major_segments
    )


def parse_percentage(value, debug=False):
    """Convert percentage strings to float values (0-1 range)."""
    if pd.isna(value):
        return np.nan
    
    try:
        # Handle numeric values
        if isinstance(value, (int, float)):
            return float(value) / 100.0 if value > 1 else float(value)
        
        # Handle string values
        value_str = str(value).strip()
        if value_str.endswith('%'):
            return float(value_str.rstrip('%')) / 100.0
        elif value_str in ['-', ' - ']:
            return np.nan
        else:
            # Direct conversion with range check
            val = float(value_str)
            return val / 100.0 if val > 1 else val
    except (ValueError, TypeError) as e:
        if debug:
            print(f"Warning: Could not convert '{value}' to float: {e}")
        return np.nan


def load_major_segments(config, debug=False):
    """
    Load and identify major segments based on participation counts from segment counts file.
    
    Args:
        config: Dictionary with configuration values including SEGMENT_COUNTS_PATH
        debug: Whether to print debug information
        
    Returns:
        List of segment names that qualify as "major" (have sufficient participation)
    """
    try:
        segment_counts_df = pd.read_csv(config['SEGMENT_COUNTS_PATH'])
        
        # Get segment columns (exclude metadata columns)
        excluded_prefixes = ('Question ID', 'Question Text', 'All', '44+', '55+')
        segment_cols = [col for col in segment_counts_df.columns 
                       if not col.startswith(excluded_prefixes) and not col.startswith('O7:')]  # Exclude countries
        
        if debug:
            print(f"Found {len(segment_cols)} potential segment columns")
        
        # Calculate average participation per segment across all questions
        avg_participation = segment_counts_df[segment_cols].mean()
        
        # Identify major segments (those with >= min_participants average)
        min_participants = config['MAJOR_SEGMENT_MIN_PARTICIPANTS']
        major_segments = avg_participation[avg_participation >= min_participants].index.tolist()
        
        if debug:
            print(f"Identified {len(major_segments)} major segments with >= {min_participants} avg participants")
            print(f"Major segments: {major_segments}")
        
        return major_segments
        
    except Exception as e:
        print(f"Warning: Could not load major segments from {config['SEGMENT_COUNTS_PATH']}: {e}")
        # Fallback to hardcoded segments if file loading fails
        return [
            # Regions
            'Africa','Asia','Caribbean','Central America','Central Asia','Eastern Africa','Eastern Asia','Eastern Europe','Europe','Middle Africa','North America','Norther Europe','Northern Africa','Northern America','Oceania','South America','South Eastern Asia','Souther Asia','Southern Africa','Southern Europe','Western Africa','Western Asia','Western Europe',
            # Ages (18+)
            'O2: 18-25','O2: 26-35','O2: 36-45','O2: 46-55','O2: 56-65','O2: 65+',
            # Sex
            'O3: Female','O3: Male','O3: Non-binary',
            # Environment
            'O4: Rural','O4: Suburban','O4: Urban',
            # Excitement for AI
            'O5: Equally concerned and excited','O5: More concerned than excited','O5: More excited than concerned',
            # Religion
            'O6: Buddhism','O6: Christianity','O6: Hinduism','O6: I do not identify with any religious group or faith','O6: Islam','O6: Judaism','O6: Other religious group','O6: Sikhism',
        ]


# --- Signal Calculation Functions ---

def calculate_duration(participant_id, binary_df, preference_df, debug=False):
    """
    Calculate the total time a participant spent in the survey based on timestamps.
    
    Args:
        participant_id: Unique ID of the participant
        binary_df: DataFrame with binary vote timestamps
        preference_df: DataFrame with preference judgment timestamps
        debug: Whether to print debug information
        
    Returns:
        pd.Timedelta: Duration between first and last recorded activity
    """
    if debug:
        print(f"[Duration {participant_id}] Calculating duration...")
    
    # Filter timestamps for this participant
    participant_binary_times = binary_df[binary_df['Participant ID'] == participant_id]['Timestamp'].dropna()
    participant_pref_times = preference_df[preference_df['Participant ID'] == participant_id]['Timestamp'].dropna()
    
    # Combine all timestamps
    all_times = pd.concat([participant_binary_times, participant_pref_times])
    
    # Check if we have enough timestamps
    if len(all_times) < 2:
        if debug:
            print(f"[Duration {participant_id}] Insufficient timestamps (found {len(all_times)})")
        return pd.Timedelta(seconds=0)
    
    # Calculate time difference between first and last activity
    duration = all_times.max() - all_times.min()
    
    if debug:
        print(f"[Duration {participant_id}] First activity: {all_times.min()}")
        print(f"[Duration {participant_id}] Last activity: {all_times.max()}")
        print(f"[Duration {participant_id}] Duration: {duration}")
    
    return duration


def calculate_low_quality_tag_percentage(participant_id, thought_labels_df, debug=False):
    """
    Calculate the percentage of a participant's responses tagged as 'Uninformative answer'.
    
    Args:
        participant_id: Unique ID of the participant
        thought_labels_df: DataFrame with thought labels
        debug: Whether to print debug information
        
    Returns:
        float: Ratio of low quality responses to total responses (0-1)
    """
    if debug:
        print(f"[LowQuality {participant_id}] Calculating low quality percentage...")
    
    # If no thought labels data is available, return 0
    if thought_labels_df.empty:
        if debug:
            print(f"[LowQuality {participant_id}] No thought labels data available")
        return 0.0
    
    # Filter for this participant's labeled responses
    participant_labels = thought_labels_df[thought_labels_df['Participant ID'] == participant_id]
    
    if participant_labels.empty:
        if debug:
            print(f"[LowQuality {participant_id}] No labeled responses found")
        return 0.0  # No labeled responses, assume perfect quality
    
    # Find tag columns
    tag_cols = [col for col in thought_labels_df.columns if col.startswith('Tag ')]
    
    if debug:
        print(f"[LowQuality {participant_id}] Found {len(tag_cols)} tag columns")
    
    if not tag_cols:
        if debug:
            print(f"[LowQuality {participant_id}] No tag columns found")
        return 0.0  # No tag columns, cannot evaluate
    
    # Check for 'Uninformative answer' tag in any tag column
    is_low_quality = participant_labels[tag_cols].apply(
        lambda row: any('Uninformative answer' == val for val in row.values if pd.notna(val)), 
        axis=1
    )
    
    num_low_quality = is_low_quality.sum()
    total_responses = len(participant_labels)
    
    if debug:
        print(f"[LowQuality {participant_id}] Low quality responses: {num_low_quality}/{total_responses}")
    
    return num_low_quality / total_responses if total_responses > 0 else 0.0


def calculate_universal_disagreement_percentage(participant_id, verbatim_map_df, aggregate_std_df, major_segments, config, debug=False):
    """
    Calculate the percentage of a participant's responses that received widespread disagreement
    across major demographic segments.
    
    Args:
        participant_id: Unique ID of the participant
        verbatim_map_df: DataFrame mapping thoughts to participants
        aggregate_std_df: DataFrame with agreement scores
        major_segments: List of major segment names to evaluate
        config: Dictionary with configuration values
        debug: Whether to print debug information
        
    Returns:
        float: Ratio of universally disagreed responses to total responses (0-1)
    """
    if debug:
        print(f"[UniversalDisagreement {participant_id}] Calculating universal disagreement...")
    
    # Get thoughts authored by this participant
    authored_thought_ids = verbatim_map_df[verbatim_map_df['Participant ID'] == participant_id]['Thought ID'].unique()
    
    if debug:
        print(f"[UniversalDisagreement {participant_id}] Found {len(authored_thought_ids)} authored thoughts")
    
    if len(authored_thought_ids) == 0:
        return 0.0  # No authored thoughts, cannot evaluate
    
    # Get agreement rates for these thoughts from aggregate data
    authored_thoughts_data = []
    
    
    # Process each authored thought
    for thought_id in authored_thought_ids:
        # Get question ID for this thought
        question_row = verbatim_map_df[verbatim_map_df['Thought ID'] == thought_id]
        if question_row.empty:
            continue
            
        question_id = question_row['Question ID'].iloc[0]
        
        # Find agreement data for the response from this Participant on this question in aggregate
        # TODO: Don't we actually need the agreement data for this specific Thought for this Question? Not the agreement rates across all Thoughts on this Question?
        # TODO: We can get the authoring Participant Id from aggregate_standardized, but not the Thought ID. So that's what we should be filtering on
        question_data = aggregate_std_df[
            (aggregate_std_df['Question ID'] == question_id) &
            (aggregate_std_df['Participant ID'] == participant_id)
        ]

        if question_data.empty:
            continue

        # Use major segments for agreement calculation
        segment_agreement_cols = [f'{col}_Agreement' for col in major_segments 
                                if f'{col}_Agreement' in question_data.columns]

        # Get the Maximum agreement rate found among any major segment for responses from this Participant on this question
        if segment_agreement_cols:
            segment_agreements = question_data[segment_agreement_cols]
            max_segment_agreement_value = segment_agreements.max(axis=1).iloc[0] if not segment_agreements.empty else np.nan
        else:
            max_segment_agreement_value = np.nan
        
        if debug and len(authored_thought_ids) <= 5:  # Only show debug for first few participants to avoid spam
            print(f"[UniversalDisagreement {participant_id}] max_segment_agreement_value: {max_segment_agreement_value} for thought {thought_id}")

            
        # For now, we'll use the 'All_Agreement' as a proxy for simplicity
        # In a complete implementation, we would check agreement across segments (started implementation, TODO: use only 'major' segments rather than all Segments)
        agreement_value = question_data['All_Agreement'].iloc[0] if 'All_Agreement' in question_data.columns else None
        
        if agreement_value is not None and not pd.isna(agreement_value):
            authored_thoughts_data.append({
                'Thought ID': thought_id,
                'Question ID': question_id,
                'All_Agreement': agreement_value,
                'Max_Segment_Agreement': max_segment_agreement_value,
            })
    
    if not authored_thoughts_data:
        if debug:
            print(f"[UniversalDisagreement {participant_id}] No agreement data found for authored thoughts")
        return 0.0
        
    # Create DataFrame with thought agreement data
    authored_aggr_df = pd.DataFrame(authored_thoughts_data)
    
    # Check how many thoughts from this prticipant have universal disagreement (below threshold)
    threshold_all = config['UNIVERSAL_DISAGREEMENT_THRESHOLD_ALL']
    threshold_segments = config['UNIVERSAL_DISAGREEMENT_THRESHOLD_SEGMENTS']
    # consider the Participant's Thought (Response) 'Universally disagreed' if EITHER agreement across 'ALL' participants is below some threshold,
    # OR if no single segment has an agreement rate above some threshold
    is_universally_disagreed = (authored_aggr_df['All_Agreement'] < threshold_all) | (authored_aggr_df['Max_Segment_Agreement'] < threshold_segments)
    num_universally_disagreed = is_universally_disagreed.sum()
    total_evaluated = len(authored_aggr_df)
    
    if debug:
        print(f"[UniversalDisagreement {participant_id}] Universally disagreed: {num_universally_disagreed}/{total_evaluated}")
    
    return num_universally_disagreed / total_evaluated if total_evaluated > 0 else 0.0


def precompute_consensus_data(binary_df, verbatim_map_df, aggregate_std_df, config, debug=False):
    """
    Pre-compute consensus data for all thoughts to optimize ASC score calculation.
    
    Args:
        binary_df: DataFrame with binary votes
        verbatim_map_df: DataFrame mapping thoughts to participants
        aggregate_std_df: DataFrame with agreement scores
        config: Dictionary with configuration values
        debug: Whether to print debug information
        
    Returns:
        dict: Dictionary with sets of strong agreement and disagreement thought IDs
    """
    start_time = time.time()
    print("Pre-computing consensus data for ASC calculation...")
    
    high_threshold = config['ASC_HIGH_THRESHOLD']
    low_threshold = config['ASC_LOW_THRESHOLD']
    
    # Create a mapping of Question ID to Agreement Score
    question_agreement_map = {}
    
    # Process each question in aggregate data
    agreement_count = 0
    for _, row in aggregate_std_df.iterrows():
        if 'Question ID' in row and 'All_Agreement' in row and pd.notna(row['All_Agreement']):
            question_id = row['Question ID']
            agreement = row['All_Agreement']
            agreement_count += 1
            
            # For each question, store the agreement score
            # If we have multiple rows for the same question, use the highest agreement
            if question_id in question_agreement_map:
                question_agreement_map[question_id] = max(question_agreement_map[question_id], agreement)
            else:
                question_agreement_map[question_id] = agreement
    
    if debug:
        print(f"Found {agreement_count} rows with valid agreement data")
        print(f"Mapped agreement scores for {len(question_agreement_map)} unique questions")
        
        # Show distribution of agreement scores
        if question_agreement_map:
            scores = list(question_agreement_map.values())
            print(f"Agreement score distribution:")
            print(f"  Min: {min(scores):.3f}")
            print(f"  Max: {max(scores):.3f}")
            print(f"  Mean: {sum(scores)/len(scores):.3f}")
            print(f"  Scores ≥ {high_threshold}: {sum(1 for s in scores if s >= high_threshold)}")
            print(f"  Scores ≤ {low_threshold}: {sum(1 for s in scores if s <= low_threshold)}")
    
    # Create sets for strong consensus thoughts
    strong_agree_thoughts = set()
    strong_disagree_thoughts = set()
    
    # Create a mapping of thought ID to question ID for quick lookups
    thought_to_question = dict(zip(verbatim_map_df['Thought ID'], verbatim_map_df['Question ID']))
    
    if debug:
        print(f"Verbatim map contains {len(thought_to_question)} thought-to-question mappings")
    
    # Process each thought and assign it to the appropriate consensus category
    mapped_thoughts = 0
    for thought_id, question_id in thought_to_question.items():
        if question_id in question_agreement_map:
            mapped_thoughts += 1
            agreement = question_agreement_map[question_id]
            
            if agreement >= high_threshold:
                strong_agree_thoughts.add(thought_id)
            elif agreement <= low_threshold:
                strong_disagree_thoughts.add(thought_id)
    
    if debug and mapped_thoughts < len(thought_to_question):
        print(f"Warning: Only {mapped_thoughts}/{len(thought_to_question)} thoughts could be mapped to agreement scores")
    
    # Create a reusable dictionary of consensus data
    consensus_data = {
        'strong_agree_thoughts': strong_agree_thoughts,
        'strong_disagree_thoughts': strong_disagree_thoughts
    }
    
    print(f"Identified {len(strong_agree_thoughts)} thoughts with strong agreement (≥{high_threshold})")
    print(f"Identified {len(strong_disagree_thoughts)} thoughts with strong disagreement (≤{low_threshold})")
    print(f"Total consensus thoughts: {len(strong_agree_thoughts) + len(strong_disagree_thoughts)}")
    print(f"Consensus data computation completed in {time.time() - start_time:.2f} seconds")
    
    return consensus_data


def calculate_asc_score(participant_id, binary_df, consensus_data, debug=False):
    """
    Calculate the Anti-Social Consensus (ASC) score - the rate at which the participant votes
    against strong consensus items.
    
    Args:
        participant_id: Unique ID of the participant
        binary_df: DataFrame with binary votes
        consensus_data: Pre-computed consensus data dictionary
        debug: Whether to print debug information
        
    Returns:
        float: Ratio of votes against consensus to total votes on consensus items (0-1)
    """
    if debug:
        print(f"[ASC {participant_id}] Calculating Anti-Social Consensus score...")
    
    # Extract pre-computed consensus data
    strong_agree_thoughts = consensus_data['strong_agree_thoughts']
    strong_disagree_thoughts = consensus_data['strong_disagree_thoughts']
    
    # Create a set of all consensus thoughts for faster lookups
    all_consensus_thoughts = strong_agree_thoughts.union(strong_disagree_thoughts)
    
    if not all_consensus_thoughts:
        if debug:
            print(f"[ASC {participant_id}] No consensus thoughts found for analysis")
        return np.nan  # Cannot evaluate ASC without consensus thoughts
    
    # Get participant's votes on consensus thoughts
    participant_binary_df = binary_df[binary_df['Participant ID'] == participant_id]
    
    # Only keep votes on consensus thoughts
    consensus_votes = participant_binary_df[participant_binary_df['Thought ID'].isin(all_consensus_thoughts)]
    
    if consensus_votes.empty:
        if debug:
            print(f"[ASC {participant_id}] Participant did not vote on any consensus thoughts")
        return np.nan  # Cannot evaluate if no votes on consensus thoughts
    
    # Count votes AGAINST consensus
    against_consensus_count = 0
    total_consensus_votes = 0
    
    # Process each vote
    for _, vote_row in consensus_votes.iterrows():
        thought_id = vote_row['Thought ID']
        vote_value = vote_row['VoteNumeric']
        
        # Skip if vote is not clearly agree (1) or disagree (0)
        if pd.isna(vote_value):
            continue
            
        total_consensus_votes += 1
            
        # Check if vote is against consensus
        if thought_id in strong_agree_thoughts and vote_value == 0:  # Disagreed with high consensus
            against_consensus_count += 1
        elif thought_id in strong_disagree_thoughts and vote_value == 1:  # Agreed with low consensus
            against_consensus_count += 1
    
    if total_consensus_votes == 0:
        if debug:
            print(f"[ASC {participant_id}] No valid consensus votes found")
        return np.nan
    
    asc_score = against_consensus_count / total_consensus_votes
    
    if debug:
        print(f"[ASC {participant_id}] Against consensus votes: {against_consensus_count}/{total_consensus_votes}")
        print(f"[ASC {participant_id}] ASC score: {asc_score}")
    
    return asc_score


def calculate_all_pri_signals(data_tuple, config, participant_limit=None, debug=False):
    """
    Calculate all PRI signals for all participants.
    
    Args:
        data_tuple: Tuple of DataFrames from load_data()
        config: Dictionary with configuration values
        participant_limit: Limit processing to first N participants (for testing)
        debug: Whether to print debug information
        
    Returns:
        DataFrame containing calculated PRI signals for each participant
    """
    print("\nCalculating PRI signals for all participants...")
    
    binary_df, preference_df, thought_labels_df, verbatim_map_df, aggregate_std_df, all_participant_ids, major_segments = data_tuple
    
    # Apply limit if specified
    if participant_limit is not None:
        participant_limit = min(participant_limit, len(all_participant_ids))
        all_participant_ids = all_participant_ids[:participant_limit]
        print(f"Limited to first {participant_limit} participants")
    else:
        participant_limit = len(all_participant_ids)
        
    print(f"Processing {participant_limit} participants...")
    
    # Pre-compute consensus data once for all participants
    consensus_data = precompute_consensus_data(binary_df, verbatim_map_df, aggregate_std_df, config, debug)
    
    # Pre-filter timestamp data for efficiency
    binary_times_df = binary_df[['Participant ID', 'Timestamp']].copy()
    preference_times_df = preference_df[['Participant ID', 'Timestamp']].copy()
    
    # Process each participant
    results = []
    
    # Use for progress reporting
    progress_step = max(1, participant_limit // 10)
    start_time = time.time()
    
    for i, participant_id in enumerate(all_participant_ids):
        if i % progress_step == 0 or i == participant_limit - 1:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1) if i > 0 else 0
            est_remaining = avg_time * (participant_limit - i - 1)
            print(f"Processing participant {i+1}/{participant_limit} ({i/participant_limit*100:.1f}%)... " +
                  f"(Est. remaining: {est_remaining/60:.1f} minutes)")
        
        # Calculate metrics
        try:
            # 1. Duration
            duration = calculate_duration(participant_id, binary_times_df, preference_times_df, debug)
            
            # 2. Low Quality Tags Percentage
            low_quality_perc = calculate_low_quality_tag_percentage(participant_id, thought_labels_df, debug)
            
            # 3. Universal Disagreement Percentage
            universal_disagreement_perc = calculate_universal_disagreement_percentage(
                participant_id, verbatim_map_df, aggregate_std_df, major_segments, config, debug
            )
            
            # 4. Anti-Social Consensus Score (raw - lower is better)
            asc_raw = calculate_asc_score(participant_id, binary_df, consensus_data, debug)
            
            # Add results
            results.append({
                'Participant ID': participant_id,
                'Duration_seconds': duration.total_seconds() if pd.notna(duration) else np.nan,
                'LowQualityTag_Perc': low_quality_perc,
                'UniversalDisagreement_Perc': universal_disagreement_perc,
                'ASC_Score_Raw': asc_raw,
            })
        except Exception as e:
            print(f"Error processing participant {participant_id}: {e}")
            # Add empty results to maintain participant count
            results.append({
                'Participant ID': participant_id,
                'Duration_seconds': np.nan,
                'LowQualityTag_Perc': np.nan,
                'UniversalDisagreement_Perc': np.nan,
                'ASC_Score_Raw': np.nan,
            })
    
    results_df = pd.DataFrame(results)
    print("Signal calculation complete.")
    return results_df


def normalize_and_calculate_pri(pri_signals_df, config, debug=False):
    """
    Normalize the raw PRI signals and calculate the final PRI score.
    
    Args:
        pri_signals_df: DataFrame with raw PRI metrics
        config: Dictionary with configuration values
        debug: Whether to print debug information
        
    Returns:
        DataFrame with normalized metrics and final PRI score
    """
    print("\nNormalizing metrics and calculating final PRI scores...")
    
    # Check how many NaN values we have in each column
    if debug:
        print("\nNaN counts in raw signals:")
        print(pri_signals_df[['Duration_seconds', 'LowQualityTag_Perc', 
                              'UniversalDisagreement_Perc', 'ASC_Score_Raw']].isna().sum())
    
    # Simple min-max normalization function
    def min_max_normalize(series, invert=False, reasonable_max=None):
        """
        Min-max normalization with optional inversion and reasonable maximum cap.
        
        For reasonable_max: values above this threshold get normalized score of 1.0,
        values below get scaled between min_val and reasonable_max.
        """
        if series.isna().all():
            return series  # Return as-is if all NaN
        
        # Replace NaN with median for normalization purposes
        median_val = series.median()
        filled_series = series.fillna(median_val)
        
        min_val = filled_series.min()
        max_val = filled_series.max()
        
        # Avoid division by zero
        if min_val == max_val:
            normalized = pd.Series(0.5, index=series.index)
        else:
            if reasonable_max is not None and max_val > reasonable_max:
                # Cap values above reasonable_max to 1.0, normalize others between min and reasonable_max
                normalized = (filled_series - min_val) / (reasonable_max - min_val)
                normalized[filled_series > reasonable_max] = 1.0
            else:
                normalized = (filled_series - min_val) / (max_val - min_val)
            
        # Invert if needed (for metrics where lower raw value is better)
        if invert:
            normalized = 1 - normalized
            
        # Restore NaN values
        normalized[series.isna()] = np.nan
        
        return normalized
    
    # Normalize metrics
    
    # 1. Duration (longer duration is better)
    pri_signals_df['Duration_Norm'] = min_max_normalize(pri_signals_df['Duration_seconds'], reasonable_max=config['DURATION_REASONABLE_MAX'])
    
    # 2. Low Quality Tags (lower percentage is better, so invert)
    pri_signals_df['LowQualityTag_Norm'] = min_max_normalize(pri_signals_df['LowQualityTag_Perc'], invert=True)
    
    # 3. Universal Disagreement (lower percentage is better, so invert)
    pri_signals_df['UniversalDisagreement_Norm'] = min_max_normalize(pri_signals_df['UniversalDisagreement_Perc'], invert=True)
    
    # 4. Anti-Social Consensus (lower score is better, so invert)
    asc_available = not pri_signals_df['ASC_Score_Raw'].isna().all()
    
    if asc_available:
        # Normal calculation with ASC
        pri_signals_df['ASC_Norm'] = min_max_normalize(pri_signals_df['ASC_Score_Raw'], invert=True)
        
        # Define weights for each component
        weights = {
            'Duration_Norm': config['DURATION_WEIGHT'],
            'LowQualityTag_Norm': config['LOW_QUALITY_TAG_WEIGHT'],
            'UniversalDisagreement_Norm': config['UNIVERSAL_DISAGREEMENT_WEIGHT'],
            'ASC_Norm': config['ASC_WEIGHT']
        }
        
        # Calculate final weighted PRI score with all components
        pri_signals_df['PRI_Score'] = (
            pri_signals_df['Duration_Norm'] * weights['Duration_Norm'] +
            pri_signals_df['LowQualityTag_Norm'] * weights['LowQualityTag_Norm'] +
            pri_signals_df['UniversalDisagreement_Norm'] * weights['UniversalDisagreement_Norm'] +
            pri_signals_df['ASC_Norm'] * weights['ASC_Norm']
        )
    else:
        # Adjusted calculation without ASC
        print("Warning: No valid ASC scores available. Calculating PRI without ASC component.")
        
        # Adjust weights to distribute ASC's weight to other components
        total_weight = config['DURATION_WEIGHT'] + config['LOW_QUALITY_TAG_WEIGHT'] + config['UNIVERSAL_DISAGREEMENT_WEIGHT']
        
        adjusted_weights = {
            'Duration_Norm': config['DURATION_WEIGHT'] / total_weight,
            'LowQualityTag_Norm': config['LOW_QUALITY_TAG_WEIGHT'] / total_weight,
            'UniversalDisagreement_Norm': config['UNIVERSAL_DISAGREEMENT_WEIGHT'] / total_weight
        }
        
        # Calculate final weighted PRI score without ASC
        pri_signals_df['PRI_Score'] = (
            pri_signals_df['Duration_Norm'] * adjusted_weights['Duration_Norm'] +
            pri_signals_df['LowQualityTag_Norm'] * adjusted_weights['LowQualityTag_Norm'] +
            pri_signals_df['UniversalDisagreement_Norm'] * adjusted_weights['UniversalDisagreement_Norm']
        )
    
    # Create a 1-5 scale version for easier interpretation
    pri_signals_df['PRI_Scale_1_5'] = pri_signals_df['PRI_Score'] * 4 + 1
    
    print("PRI score calculation complete.")
    return pri_signals_df


def create_pri_distribution_chart(pri_signals_df, gd_number, config, debug=False):
    """
    Create a comprehensive PRI score distribution visualization.
    
    Args:
        pri_signals_df: DataFrame with calculated PRI scores
        gd_number: Global Dialogue number for labeling
        config: Configuration dictionary with file paths
        debug: Whether to print debug information
        
    Returns:
        str: Path to the saved visualization file
    """
    # Filter out NaN values for visualization
    valid_scores = pri_signals_df['PRI_Scale_1_5'].dropna()
    
    if len(valid_scores) == 0:
        print("Warning: No valid PRI scores to visualize")
        return None
    
    # Calculate key statistics
    stats = {
        'count': len(valid_scores),
        'mean': valid_scores.mean(),
        'median': valid_scores.median(),
        'std': valid_scores.std(),
        'min': valid_scores.min(),
        'max': valid_scores.max(),
        'q25': valid_scores.quantile(0.25),
        'q75': valid_scores.quantile(0.75)
    }
    
    if debug:
        print(f"PRI visualization stats: {stats}")
    
    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
    fig.suptitle(f'Participant Reliability Index (PRI) Distribution - Global Dialogue {gd_number}', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Main histogram
    n_bins = min(50, max(10, int(len(valid_scores) / 10)))  # Adaptive bin count
    n, bins, patches = ax1.hist(valid_scores, bins=n_bins, alpha=0.7, color='skyblue', 
                               edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for key statistics
    ax1.axvline(stats['median'], color='red', linestyle='-', linewidth=2, label=f"Median: {stats['median']:.2f}")
    ax1.axvline(stats['mean'], color='orange', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.2f}")
    ax1.axvline(stats['q25'], color='green', linestyle=':', linewidth=2, label=f"25th Percentile: {stats['q25']:.2f}")
    ax1.axvline(stats['q75'], color='purple', linestyle=':', linewidth=2, label=f"75th Percentile: {stats['q75']:.2f}")
    
    # Formatting for main plot
    ax1.set_xlabel('PRI Score (1-5 Scale)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Participants', fontsize=12, fontweight='bold')
    ax1.set_xlim(1, 5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    
    # Add text box with key statistics
    stats_text = f"""Summary Statistics:
    Participants: {stats['count']:,}
    Mean: {stats['mean']:.3f}
    Median: {stats['median']:.3f}
    Std Dev: {stats['std']:.3f}
    Range: {stats['min']:.2f} - {stats['max']:.2f}
    IQR: {stats['q25']:.2f} - {stats['q75']:.2f}"""
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Box plot
    bp = ax2.boxplot(valid_scores, vert=False, patch_artist=True, 
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    
    ax2.set_xlabel('PRI Score (1-5 Scale)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlim(1, 5)
    ax2.grid(True, alpha=0.3)
    ax2.set_yticks([])  # Remove y-axis ticks for box plot
    
    # Add percentile labels on box plot
    ax2.text(stats['q25'], 1.3, f"Q1\n{stats['q25']:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.text(stats['median'], 1.3, f"Median\n{stats['median']:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')
    ax2.text(stats['q75'], 1.3, f"Q3\n{stats['q75']:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add interpretation guide
    interpretation_text = """PRI Score Interpretation:
    4.5-5.0: Highly Reliable  |  3.5-4.5: Reliable  |  2.5-3.5: Moderately Reliable  |  1.5-2.5: Low Reliability  |  1.0-1.5: Very Low Reliability"""
    
    fig.text(0.5, 0.02, interpretation_text, ha='center', va='bottom', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12)
    
    # Save the chart
    chart_path = f"Data/GD{gd_number}/GD{gd_number}_pri_distribution.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    if debug:
        print(f"PRI distribution chart saved to: {chart_path}")
    
    plt.close()  # Close the figure to free memory
    
    return chart_path


def main():
    """Main execution function"""
    start_time = time.time()
    
    # Parse command-line arguments
    args = parse_args()
    gd_number = args.gd_number
    debug = args.debug
    participant_limit = args.limit
    
    print(f"Calculating PRI for Global Dialogue {gd_number}")
    print(f"Debug mode: {'Enabled' if debug else 'Disabled'}")
    if participant_limit:
        print(f"Limiting to first {participant_limit} participants for testing")
    
    # Get configuration for this GD
    try:
        config = get_config(gd_number)
    except Exception as e:
        print(f"Error in configuration: {e}")
        sys.exit(1)
    
    # 1. Load and clean all necessary data
    try:
        data_tuple = load_data(config, debug)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # 2. Calculate raw PRI signals for all participants
    pri_signals_df = calculate_all_pri_signals(data_tuple, config, participant_limit, debug)
    
    # 3. Normalize and calculate final PRI score
    pri_signals_df = normalize_and_calculate_pri(pri_signals_df, config, debug)
    
    # 4. Print summary statistics
    print("\nPRI Score Statistics:")
    print(pri_signals_df[['PRI_Score', 'PRI_Scale_1_5']].describe())
    
    # 5. Show top/bottom participants
    print("\nTop 5 Most Reliable Participants:")
    print(pri_signals_df.sort_values('PRI_Score', ascending=False).head(5)[['Participant ID', 'PRI_Score', 'PRI_Scale_1_5']])
    
    print("\nBottom 5 Least Reliable Participants:")
    print(pri_signals_df.sort_values('PRI_Score', ascending=True).head(5)[['Participant ID', 'PRI_Score', 'PRI_Scale_1_5']])
    
    # 6. Save results to CSV
    output_path = config['OUTPUT_PATH']
    pri_signals_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # 7. Generate PRI distribution visualization
    try:
        chart_path = create_pri_distribution_chart(pri_signals_df, gd_number, config, debug)
        if chart_path:
            print(f"PRI distribution chart saved to {chart_path}")
    except Exception as e:
        print(f"Warning: Could not generate PRI distribution chart: {e}")
        if debug:
            import traceback
            traceback.print_exc()
    
    # 8. Print execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nExecution completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()