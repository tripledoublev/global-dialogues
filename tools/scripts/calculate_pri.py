# calculate_pri.py

import pandas as pd
import numpy as np
import warnings

# --- Constants and Configuration ---
# TODO: Define paths to input files for a specific GD (e.g., GD3)
GD_NUMBER = 3
DATA_DIR = f"Data/GD{GD_NUMBER}"
TAGS_DIR = f"{DATA_DIR}/tags"

# File paths
AGGREGATE_STD_PATH = f"{DATA_DIR}/GD{GD_NUMBER}_aggregate_standardized.csv"
BINARY_PATH = f"{DATA_DIR}/GD{GD_NUMBER}_binary.csv"
PREFERENCE_PATH = f"{DATA_DIR}/GD{GD_NUMBER}_preference.csv"
VERBATIM_MAP_PATH = f"{DATA_DIR}/GD{GD_NUMBER}_verbatim_map.csv"
THOUGHT_LABELS_PATH = f"{TAGS_DIR}/all_thought_labels.csv"
# SEGMENT_COUNTS_PATH = f"{DATA_DIR}/GD{GD_NUMBER}_segment_counts_by_question.csv" # May be needed for universal disagreement

# Signal Thresholds (examples, need tuning)
ASC_HIGH_THRESHOLD = 0.80 # Agreement rate for strong agreement
ASC_LOW_THRESHOLD = 0.20  # Agreement rate for strong disagreement
UNIVERSAL_DISAGREEMENT_THRESHOLD = 0.20 # Max agreement rate for a response to be considered 'disagreed'
UNIVERSAL_DISAGREEMENT_COVERAGE = 0.90 # Minimum proportion of population needed to form 'universal' disagreement

# --- Data Loading and Cleaning ---

def load_and_clean_data():
    """Loads and performs initial cleaning on necessary CSV files."""
    print("Loading data...")

    # --- Load Binary ---
    # Anticipating large file, load only necessary columns
    try:
        binary_df = pd.read_csv(
            BINARY_PATH,
            usecols=['Participant ID', 'Thought ID', 'Vote', 'Timestamp'],
            low_memory=False,
            skiprows=3 # Skip metadata headers based on preview
        )
        # Convert timestamp (format might need adjustment based on actual data)
        binary_df['Timestamp'] = pd.to_datetime(binary_df['Timestamp'], errors='coerce')
        # Convert Vote to numeric for easier comparison (Agree=1, Disagree=0, Neutral=NaN or 0.5?)
        # For ASC, we only care about Agree/Disagree
        binary_df['VoteNumeric'] = binary_df['Vote'].map({'agree': 1, 'disagree': 0}).astype(float)

    except Exception as e:
        print(f"Error loading or processing {BINARY_PATH}: {e}")
        binary_df = pd.DataFrame() # Return empty df on error


    # --- Load Preference ---
    # Only need Timestamp and Participant ID for duration calculation
    try:
        preference_df = pd.read_csv(
            PREFERENCE_PATH,
            usecols=['Participant ID', 'Timestamp'],
            low_memory=False,
            skiprows=3 # Skip metadata headers
        )
        preference_df['Timestamp'] = pd.to_datetime(preference_df['Timestamp'], errors='coerce')
    except Exception as e:
        print(f"Error loading or processing {PREFERENCE_PATH}: {e}")
        preference_df = pd.DataFrame()

    # --- Load Thought Labels ---
    try:
        thought_labels_df = pd.read_csv(
            THOUGHT_LABELS_PATH,
            low_memory=False,
            # Skip metadata if present (preview showed header on first line)
            # skiprows=1 ? Check file
        )
        # We might need Participant ID and all Tag columns
        # Ensure Participant ID type matches other files if necessary
    except Exception as e:
        print(f"Error loading or processing {THOUGHT_LABELS_PATH}: {e}")
        thought_labels_df = pd.DataFrame()


    # --- Load Verbatim Map ---
    try:
        verbatim_map_df = pd.read_csv(
            VERBATIM_MAP_PATH,
            usecols=['Participant ID', 'Thought ID', 'Question ID'], # Add 'Thought Text' if needed later
            low_memory=False,
            skiprows=3 # Skip metadata headers
        )
        # Ensure IDs match types in other DFs
    except Exception as e:
        print(f"Error loading or processing {VERBATIM_MAP_PATH}: {e}")
        verbatim_map_df = pd.DataFrame()


    # --- Load Aggregate Standardized ---
    # This is complex due to the wide format and percentages
    # We need Thought ID (or way to join), 'All' agreement, maybe segment columns
    try:
        # May need to load iteratively or handle dtype warnings
        warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
        aggregate_std_df = pd.read_csv(
            AGGREGATE_STD_PATH,
            low_memory=False,
            skiprows=3 # Skip metadata headers
        )
        warnings.simplefilter(action='default', category=pd.errors.DtypeWarning)

        # TODO: Identify the 'Thought ID' equivalent column if not named explicitly
        # It might be 'Response' or 'OriginalResponse' or require joining with verbatim_map

        # Convert 'All' agreement column to numeric (assuming it's named 'All')
        # Handle potential '%' sign and errors
        if 'All' in aggregate_std_df.columns:
             aggregate_std_df['All_Agreement'] = aggregate_std_df['All'].astype(str).str.rstrip('%').astype(float) / 100.0
        else:
             print("Warning: 'All' column not found in aggregate_std_df for agreement rates.")
             aggregate_std_df['All_Agreement'] = np.nan

        # TODO: Parse other segment columns if needed for Universal Disagreement
        # This would involve selecting relevant columns and converting % to numeric


    except Exception as e:
        print(f"Error loading or processing {AGGREGATE_STD_PATH}: {e}")
        aggregate_std_df = pd.DataFrame()


    print("Data loading complete.")
    return binary_df, preference_df, thought_labels_df, verbatim_map_df, aggregate_std_df

# --- Signal Calculation Functions ---

def calculate_duration(participant_id, binary_times_df, preference_times_df):
    """Calculates survey duration for a participant."""
    # Filter timestamps for the participant
    participant_binary_times = binary_times_df[binary_times_df['Participant ID'] == participant_id]['Timestamp'].dropna()
    participant_pref_times = preference_times_df[preference_times_df['Participant ID'] == participant_id]['Timestamp'].dropna()

    all_times = pd.concat([participant_binary_times, participant_pref_times])

    if len(all_times) < 2:
        return pd.Timedelta(seconds=0) # Or np.nan?

    duration = all_times.max() - all_times.min()
    return duration


def calculate_low_quality_tag_perc(participant_id, thought_labels_df):
    """Calculates the percentage of a participant's responses tagged as 'Uninformative Answer'."""
    participant_labels = thought_labels_df[thought_labels_df['Participant ID'] == participant_id]

    if participant_labels.empty:
        return 0.0 # Or np.nan? Participant might not have submitted Ask Opinion answers

    tag_cols = [col for col in thought_labels_df.columns if col.startswith('Tag ')]
    # Check if 'Uninformative Answer' exists in any tag column for each row
    is_low_quality = participant_labels[tag_cols].apply(lambda row: 'Uninformative Answer' in row.values, axis=1)

    num_low_quality = is_low_quality.sum()
    total_responses = len(participant_labels)

    return num_low_quality / total_responses


def calculate_universal_disagreement_perc(participant_id, verbatim_map_df, aggregate_std_df):
    """
    Calculates the percentage of a participant's authored responses
    that received 'universal disagreement'.
    """
    # Get thoughts authored by the participant
    authored_thought_ids = verbatim_map_df[verbatim_map_df['Participant ID'] == participant_id]['Thought ID'].unique()

    if len(authored_thought_ids) == 0:
        return 0.0 # Or np.nan?

    # Filter aggregate data for these thoughts
    # TODO: This assumes aggregate_std_df has a 'Thought ID' column matching verbatim_map_df
    authored_aggr_df = aggregate_std_df[aggregate_std_df['Thought ID'].isin(authored_thought_ids)]

    if authored_aggr_df.empty:
         print(f"Warning: No aggregate data found for thoughts by participant {participant_id}")
         return 0.0 # Or np.nan?

    # --- Check for Universal Disagreement ---
    # This is simplified: uses only the 'All' column for now.
    # A more robust version would check across multiple segments.
    # TODO: Implement check across segments using UNIVERSAL_DISAGREEMENT_* thresholds
    # Requires parsing segment columns and potentially segment counts.

    is_disagreed = authored_aggr_df['All_Agreement'] < UNIVERSAL_DISAGREEMENT_THRESHOLD
    num_universally_disagreed = is_disagreed.sum()
    total_authored = len(authored_thought_ids) # Use count from verbatim map

    # Ensure we use the correct total (might differ if some thoughts lack aggregate data)
    actual_evaluated_count = len(authored_aggr_df['Thought ID'].unique())
    if actual_evaluated_count == 0:
        return 0.0

    return num_universally_disagreed / actual_evaluated_count


def calculate_asc_score(participant_id, binary_df, aggregate_std_df):
    """
    Calculates the participant's rate of voting AGAINST strong consensus.
    (Lower is better, so this raw score needs inversion later).
    """
    # Identify strong consensus thoughts
    strong_agree_ids = aggregate_std_df[aggregate_std_df['All_Agreement'] >= ASC_HIGH_THRESHOLD]['Thought ID'].unique()
    strong_disagree_ids = aggregate_std_df[aggregate_std_df['All_Agreement'] <= ASC_LOW_THRESHOLD]['Thought ID'].unique()
    strong_consensus_ids = np.concatenate([strong_agree_ids, strong_disagree_ids])

    if len(strong_consensus_ids) == 0:
        print("Warning: No strong consensus thoughts found in aggregate data.")
        return np.nan # Cannot calculate score

    # Get participant's votes on these specific thoughts
    participant_votes = binary_df[
        (binary_df['Participant ID'] == participant_id) &
        (binary_df['Thought ID'].isin(strong_consensus_ids))
    ].dropna(subset=['VoteNumeric']) # Only consider Agree/Disagree votes

    if participant_votes.empty:
        # Participant didn't vote on any strong consensus items
        return np.nan # Or a neutral score like 0.5? Needs consideration.

    # Join votes with consensus info
    votes_with_consensus = participant_votes.merge(
        aggregate_std_df[['Thought ID', 'All_Agreement']],
        on='Thought ID',
        how='left'
    )

    # Identify votes AGAINST consensus
    # Vote is 1 (Agree), but consensus was Low (<0.2)
    voted_agree_on_low = (votes_with_consensus['VoteNumeric'] == 1) & (votes_with_consensus['All_Agreement'] <= ASC_LOW_THRESHOLD)
    # Vote is 0 (Disagree), but consensus was High (>0.8)
    voted_disagree_on_high = (votes_with_consensus['VoteNumeric'] == 0) & (votes_with_consensus['All_Agreement'] >= ASC_HIGH_THRESHOLD)

    num_against_consensus = voted_agree_on_low.sum() + voted_disagree_on_high.sum()
    total_consensus_votes = len(participant_votes)

    return num_against_consensus / total_consensus_votes


# --- Main Processing ---

def calculate_all_pri_signals(binary_df, preference_df, thought_labels_df, verbatim_map_df, aggregate_std_df):
    """Calculates all PRI signals for all participants."""

    # Get unique participant IDs from a comprehensive source (e.g., binary votes)
    all_participant_ids = binary_df['Participant ID'].unique()
    print(f"Calculating signals for {len(all_participant_ids)} participants...")

    results = []

    # Pre-filter timestamp data for efficiency in duration calculation
    binary_times_df = binary_df[['Participant ID', 'Timestamp']].copy()
    preference_times_df = preference_df[['Participant ID', 'Timestamp']].copy()

    # TODO: Pre-join aggregate_std_df with verbatim_map_df if 'Thought ID' isn't directly in aggregate
    # This depends on the exact structure revealed after loading aggregate_std_df


    for participant_id in all_participant_ids:
        # 1. Duration
        duration = calculate_duration(participant_id, binary_times_df, preference_times_df)

        # 2. Low Quality Tags %
        low_quality_perc = calculate_low_quality_tag_perc(participant_id, thought_labels_df)

        # 3. Universal Disagreement %
        universal_disagreement_perc = calculate_universal_disagreement_perc(participant_id, verbatim_map_df, aggregate_std_df)

        # 4. ASC Score (raw - lower is better)
        asc_raw = calculate_asc_score(participant_id, binary_df, aggregate_std_df)

        results.append({
            'Participant ID': participant_id,
            'Duration_seconds': duration.total_seconds() if pd.notna(duration) else np.nan,
            'LowQualityTag_Perc': low_quality_perc,
            'UniversalDisagreement_Perc': universal_disagreement_perc,
            'ASC_Score_Raw': asc_raw,
        })

    results_df = pd.DataFrame(results)
    print("Signal calculation complete.")
    return results_df

# --- Script Execution ---

if __name__ == "__main__":
    # 1. Load Data
    binary_df, preference_df, thought_labels_df, verbatim_map_df, aggregate_std_df = load_and_clean_data()

    # Check if data loading was successful
    if any(df.empty for df in [binary_df, preference_df, thought_labels_df, verbatim_map_df, aggregate_std_df]):
         print("Error during data loading. Exiting.")
         # exit() # Temporarily disable exit to see partial loads/calcs
         pass


    # 2. Calculate Raw Signals
    pri_signals_df = calculate_all_pri_signals(
        binary_df, preference_df, thought_labels_df, verbatim_map_df, aggregate_std_df
    )

    print("Raw PRI Signals Head:")
    print(pri_signals_df.head())

    # --- Phase 3: Normalization and Final PRI ---
    # TODO: Implement normalization functions (e.g., percentile rank, min-max, custom bins)
    # TODO: Apply normalization to each signal (ensuring higher = better reliability)
    # TODO: Define weights and calculate final weighted PRI score

    # Example placeholder for normalization and final calculation
    # pri_signals_df['Duration_Norm'] = normalize_duration(pri_signals_df['Duration_seconds'])
    # pri_signals_df['LowQualityTag_Norm'] = 1 - pri_signals_df['LowQualityTag_Perc'] # Higher % is bad -> invert
    # pri_signals_df['UniversalDisagreement_Norm'] = 1 - pri_signals_df['UniversalDisagreement_Perc'] # Higher % is bad -> invert
    # pri_signals_df['ASC_Norm'] = 1 - pri_signals_df['ASC_Score_Raw'].fillna(0.5) # Higher raw score is bad -> invert. Fill NaNs with neutral value?

    # weights = {'Duration_Norm': 0.25, 'LowQualityTag_Norm': 0.25, 'UniversalDisagreement_Norm': 0.25, 'ASC_Norm': 0.25}
    # pri_signals_df['PRI_Score'] = (
    #     pri_signals_df['Duration_Norm'] * weights['Duration_Norm'] +
    #     pri_signals_df['LowQualityTag_Norm'] * weights['LowQualityTag_Norm'] +
    #     pri_signals_df['UniversalDisagreement_Norm'] * weights['UniversalDisagreement_Norm'] +
    #     pri_signals_df['ASC_Norm'] * weights['ASC_Norm']
    # )

    # print("PRI Scores Head (Placeholder):")
    # print(pri_signals_df.head())

    # TODO: Save results to CSV
    # output_path = f"{DATA_DIR}/GD{GD_NUMBER}_pri_scores.csv"
    # pri_signals_df.to_csv(output_path, index=False)
    # print(f"Results saved to {output_path}")
