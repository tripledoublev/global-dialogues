# calculate_pri.py

import pandas as pd
import numpy as np
import warnings

# --- Constants and Configuration ---
# TODO: Define paths to input files for a specific GD (e.g., GD3)
GD_NUMBER = 1
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
    binary_required_cols = ['Participant ID', 'Thought ID', 'Vote', 'Timestamp']
    try:
        binary_df = pd.read_csv(
            BINARY_PATH,
            delimiter=',',
            quotechar='"',
            on_bad_lines='warn',
            low_memory=False,
            header=0
        )
        print(f"Columns for {BINARY_PATH}: {binary_df.columns}")
        missing_cols = [col for col in binary_required_cols if col not in binary_df.columns]
        if missing_cols:
            print(f"Warning: Missing required columns in {BINARY_PATH}: {missing_cols}")
            for col in missing_cols:
                binary_df[col] = np.nan
        else:
            print(f"Found required columns in {BINARY_PATH}")
        if 'Timestamp' in binary_df.columns:
             binary_df['Timestamp'] = pd.to_datetime(binary_df['Timestamp'], errors='coerce')
        if 'Vote' in binary_df.columns:
            binary_df['VoteNumeric'] = binary_df['Vote'].map({'agree': 1, 'disagree': 0}).astype(float)
        else:
            binary_df['VoteNumeric'] = np.nan
    except Exception as e:
        print(f"Error loading or processing {BINARY_PATH}: {e}")
        binary_df = pd.DataFrame(columns=binary_required_cols + ['VoteNumeric'])

    # --- Load Preference ---
    preference_required_cols = ['Participant ID', 'Timestamp']
    try:
        preference_df = pd.read_csv(
            PREFERENCE_PATH,
            delimiter=',',
            quotechar='"',
            on_bad_lines='warn',
            low_memory=False,
            header=0
        )
        print(f"Columns for {PREFERENCE_PATH}: {preference_df.columns}")
        missing_cols = [col for col in preference_required_cols if col not in preference_df.columns]
        if missing_cols:
            print(f"Warning: Missing required columns in {PREFERENCE_PATH}: {missing_cols}")
            for col in missing_cols:
                 preference_df[col] = np.nan
        else:
             print(f"Found required columns in {PREFERENCE_PATH}")
        if 'Timestamp' in preference_df.columns:
            preference_df['Timestamp'] = pd.to_datetime(preference_df['Timestamp'], errors='coerce')
    except Exception as e:
        print(f"Error loading or processing {PREFERENCE_PATH}: {e}")
        preference_df = pd.DataFrame(columns=preference_required_cols)

    # --- Load Thought Labels ---
    thought_labels_required_cols = ['Participant ID', 'Question ID']
    try:
        thought_labels_df = pd.read_csv(
            THOUGHT_LABELS_PATH,
            low_memory=False,
            encoding='utf-8-sig'
        )
        thought_labels_df.columns = thought_labels_df.columns.str.strip()
        print(f"Cleaned columns for {THOUGHT_LABELS_PATH}: {thought_labels_df.columns}")
        missing_cols = [col for col in thought_labels_required_cols if col not in thought_labels_df.columns]
        if missing_cols:
            print(f"Warning: Missing required columns in {THOUGHT_LABELS_PATH}: {missing_cols}")
        else:
             print(f"Found required columns in {THOUGHT_LABELS_PATH}")
    except Exception as e:
        print(f"Error loading or processing {THOUGHT_LABELS_PATH}: {e}")
        thought_labels_df = pd.DataFrame(columns=thought_labels_required_cols)

    # --- Load Verbatim Map (DEBUGGING) ---
    verbatim_map_required_cols = ['Question ID', 'Question Text', 'Participant ID', 'Thought ID', 'Thought Text']
    verbatim_map_df = pd.DataFrame(columns=verbatim_map_required_cols) # Default empty
    try:
        print(f"\nAttempting to load {VERBATIM_MAP_PATH} with header=0, engine='python', encoding='utf-8'...")
        verbatim_map_df = pd.read_csv(
            VERBATIM_MAP_PATH,
            delimiter=',',
            quotechar='"',
            on_bad_lines='warn', # Keep warning to see issues
            header=0,
            encoding='utf-8',   # Explicitly set encoding
            engine='python'     # Try the python engine for robustness
        )
        print(f"Successfully loaded {VERBATIM_MAP_PATH}.")
        print(f"Columns: {verbatim_map_df.columns.tolist()}")
        print(f"Shape: {verbatim_map_df.shape}")

        # Basic validation
        missing_cols = [col for col in verbatim_map_required_cols if col not in verbatim_map_df.columns]
        if missing_cols:
            print(f"Warning: Missing required columns after loading {VERBATIM_MAP_PATH}: {missing_cols}")
        else:
            print(f"Found all required columns in {VERBATIM_MAP_PATH}")

    except Exception as e:
         print(f"Error loading {VERBATIM_MAP_PATH}: {e}")
         verbatim_map_df = pd.DataFrame(columns=verbatim_map_required_cols)

    # --- Load Aggregate Standardized ---
    aggregate_std_required_cols = ['All', 'Participant ID', 'Question ID']
    try:
        warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
        aggregate_std_df = pd.read_csv(
            AGGREGATE_STD_PATH,
            low_memory=False
        )
        warnings.simplefilter(action='default', category=pd.errors.DtypeWarning)
        print(f"Columns for {AGGREGATE_STD_PATH}: {aggregate_std_df.columns[:20]}...")
        actual_participant_col = 'Participant ID'
        actual_all_col = 'All'
        actual_qid_col = 'Question ID'
        aggregate_std_required_cols = [actual_all_col, actual_participant_col, actual_qid_col]
        missing_cols = [col for col in aggregate_std_required_cols if col not in aggregate_std_df.columns]
        if missing_cols:
             print(f"Warning: Missing required columns in {AGGREGATE_STD_PATH}: {missing_cols}")
             aggregate_std_df = pd.DataFrame(columns=['All_Agreement', 'Author Participant ID', 'Question ID', 'Thought ID', 'Thought Text'])
        else:
             print(f"Found required columns ('{actual_all_col}', '{actual_participant_col}', '{actual_qid_col}') in {AGGREGATE_STD_PATH}")
             aggregate_std_df.rename(columns={actual_participant_col: 'Author Participant ID'}, inplace=True)
             print(f"Renamed '{actual_participant_col}' to 'Author Participant ID'")
             aggregate_std_df['Thought ID'] = np.nan
             aggregate_std_df['Thought Text'] = np.nan
             print("Added placeholder 'Thought ID' and 'Thought Text' columns")
             segment_cols = [col for col in aggregate_std_df.columns if col.startswith(('O1:', 'O2:', 'O3:', 'O4:', 'O5:', 'O6:', 'O7:'))]
             print(f"Found {len(segment_cols)} segment columns to parse.")
             for col in [actual_all_col] + segment_cols:
                 numeric_col_name = f"{col}_numeric"
                 try:
                     aggregate_std_df[numeric_col_name] = aggregate_std_df[col].astype(str).str.rstrip('%').astype(float) / 100.0
                 except Exception as parse_err:
                     aggregate_std_df[numeric_col_name] = np.nan
             print("Finished parsing segment columns.")
             if f"{actual_all_col}_numeric" in aggregate_std_df.columns:
                  aggregate_std_df.rename(columns={f"{actual_all_col}_numeric": 'All_Agreement'}, inplace=True)
                  print(f"Renamed parsed '{actual_all_col}_numeric' to 'All_Agreement'")
    except Exception as e:
        print(f"Error loading or processing {AGGREGATE_STD_PATH}: {e}")
        aggregate_std_df = pd.DataFrame(columns=['All_Agreement', 'Author Participant ID', 'Question ID', 'Thought ID', 'Thought Text'])

    # --- Join Aggregate Standardized with Verbatim Map ---
    print("\nAttempting to join Aggregate Standardized with Verbatim Map...")
    if not verbatim_map_df.empty and not aggregate_std_df.empty:
        print(f"Aggregate shape before merge: {aggregate_std_df.shape}")
        print(f"Aggregate Thought ID null count before merge: {aggregate_std_df['Thought ID'].isnull().sum()}")
        print(f"Aggregate Thought ID unique values before merge: {aggregate_std_df['Thought ID'].nunique()}") # Should be 1 (placeholder)

        # Select only necessary columns from verbatim_map for the merge
        verbatim_subset = verbatim_map_df[['Question ID', 'Participant ID', 'Thought ID', 'Thought Text']].copy()
        print(f"Verbatim Map subset shape: {verbatim_subset.shape}")

        # Perform the merge
        aggregate_std_df = pd.merge(
            aggregate_std_df,
            verbatim_subset,
            how='left',
            left_on=['Question ID', 'Author Participant ID'],
            right_on=['Question ID', 'Participant ID'],
            suffixes=('_agg', '_vmap') # Use specific suffixes
        )
        print(f"Aggregate shape after merge: {aggregate_std_df.shape}")
        print(f"Columns after merge: {aggregate_std_df.columns.tolist()}")

        # Check if the merge created the expected columns from verbatim_map
        if 'Thought ID_vmap' in aggregate_std_df.columns:
            # Update original 'Thought ID' where merge was successful
            aggregate_std_df['Thought ID'] = aggregate_std_df['Thought ID_vmap'].fillna(aggregate_std_df['Thought ID_agg'])
            print(f"Aggregate Thought ID null count after update: {aggregate_std_df['Thought ID'].isnull().sum()}")
            print(f"Aggregate Thought ID unique values after update: {aggregate_std_df['Thought ID'].nunique()}")
            matched_rows = aggregate_std_df['Thought ID_vmap'].notna().sum()
            print(f"Found {matched_rows} rows matched from Verbatim Map based on Thought ID_vmap.")

            # Update 'Thought Text' similarly
            if 'Thought Text_vmap' in aggregate_std_df.columns:
                 aggregate_std_df['Thought Text'] = aggregate_std_df['Thought Text_vmap'].fillna(aggregate_std_df['Thought Text_agg'])

            # Drop temporary and redundant columns
            cols_to_drop = ['Participant ID_vmap', 'Thought ID_agg', 'Thought ID_vmap', 'Thought Text_agg', 'Thought Text_vmap']
            aggregate_std_df.drop(columns=[col for col in cols_to_drop if col in aggregate_std_df.columns], inplace=True)
            print(f"Columns after cleanup: {aggregate_std_df.columns.tolist()}")
            print(f"Final Aggregate shape: {aggregate_std_df.shape}")
        else:
            print("Merge did not produce 'Thought ID_vmap' column. Check merge keys and verbatim_map data.")

    else:
        print("Skipping join due to empty or incomplete dataframes.")

    print("\nData loading and join attempt complete.")
    # Check if the crucial All_Agreement column exists before proceeding
    if 'All_Agreement' not in aggregate_std_df.columns:
         print("Error: 'All_Agreement' column not found in aggregate_std_df after processing.")
         return None # Or raise an error

    # Final check for fragmentation and copy if needed (REMOVED - .blocks is internal)
    # num_fragments = len(aggregate_std_df.blocks)
    # if num_fragments > 100: # Arbitrary threshold, adjust as needed
    #     print(f"Warning: DataFrame is highly fragmented ({num_fragments} blocks). Creating a copy to improve performance.")
    #     aggregate_std_df = aggregate_std_df.copy()

    # Get unique participant IDs from a comprehensive source (e.g., binary votes)
    all_participant_ids = binary_df['Participant ID'].unique()
    print(f"Calculating signals for {len(all_participant_ids)} participants...")

    return binary_df, preference_df, thought_labels_df, verbatim_map_df, aggregate_std_df, all_participant_ids

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
    that received 'universal disagreement' across major segments.
    NOTE: Relies on aggregate_std_df having 'Thought ID' populated via join
          and numeric segment columns (e.g., O1: ..._numeric).
    """
    # Debug Print
    # print(f"[UDebug {participant_id}] Checking authored thoughts...")
    authored_thought_ids = verbatim_map_df.get('Thought ID', pd.Series(dtype=str))[verbatim_map_df['Participant ID'] == participant_id].unique()

    # Debug Print
    # print(f"[UDebug {participant_id}] Found {len(authored_thought_ids)} authored thoughts.")
    if len(authored_thought_ids) == 0:
        return 0.0

    # Filter aggregate data for these thoughts and ensure Thought ID exists
    authored_aggr_df = aggregate_std_df[
        aggregate_std_df['Thought ID'].isin(authored_thought_ids) &
        aggregate_std_df['Thought ID'].notna()
    ].copy()

    # Debug Print
    # print(f"[UDebug {participant_id}] Found {len(authored_aggr_df)} matching rows in aggregate_std_df.")
    if authored_aggr_df.empty:
         return 0.0

    # --- Check for Universal Disagreement across major segments ---
    # Identify numeric segment columns (O1-O7)
    segment_cols_numeric = [col for col in authored_aggr_df.columns if col.startswith(('O1:', 'O2:', 'O3:', 'O4:', 'O5:', 'O6:', 'O7:')) and col.endswith('_numeric')]

    # Debug Print
    # print(f"[UDebug {participant_id}] Found {len(segment_cols_numeric)} numeric segment columns.")
    if not segment_cols_numeric:
        print(f"Warning [P:{participant_id}]: No numeric segment columns found for universal disagreement calculation. Defaulting to 0.")
        return 0.0

    # For each thought, check if agreement is below threshold in almost all segments
    # This is a simplified check: count how many segments are *below* threshold
    # A more precise check might need segment counts to weigh them
    below_threshold_count = (authored_aggr_df[segment_cols_numeric] < UNIVERSAL_DISAGREEMENT_THRESHOLD).sum(axis=1)

    # Consider it universally disagreed if it's below threshold in a high proportion of segments
    proportion_below_threshold = below_threshold_count / len(segment_cols_numeric)
    is_universally_disagreed = proportion_below_threshold >= UNIVERSAL_DISAGREEMENT_COVERAGE

    num_universally_disagreed = is_universally_disagreed.sum()

    actual_evaluated_count = len(authored_aggr_df['Thought ID'].unique())

    # Debug Print
    # print(f"[UDebug {participant_id}] Num universally disagreed: {num_universally_disagreed}, Total evaluated: {actual_evaluated_count}")
    if actual_evaluated_count == 0:
        return 0.0

    return num_universally_disagreed / actual_evaluated_count


def calculate_asc_score(participant_id, binary_df, aggregate_std_df):
    """
    Calculates the participant's rate of voting AGAINST strong consensus.
    (Lower is better, so this raw score needs inversion later).
    NOTE: Relies on aggregate_std_df having 'Thought ID' populated via join.
    """
    print(f"[ASCDebug {participant_id}] Calculating ASC score...")

    # --- Debug: Inspect high/low consensus rows before filtering ---
    print(f"[ASCDebug {participant_id}] High consensus rows sample (All_Agreement >= {ASC_HIGH_THRESHOLD}):")
    print(aggregate_std_df.loc[aggregate_std_df['All_Agreement'] >= ASC_HIGH_THRESHOLD, ['All_Agreement', 'Thought ID']].head())
    print(f"[ASCDebug {participant_id}] Low consensus rows sample (All_Agreement <= {ASC_LOW_THRESHOLD}):")
    print(aggregate_std_df.loc[aggregate_std_df['All_Agreement'] <= ASC_LOW_THRESHOLD, ['All_Agreement', 'Thought ID']].head())
    # --- End Debug ---

    # Identify strong consensus thoughts
    strong_agree_ids = aggregate_std_df.loc[aggregate_std_df['All_Agreement'] >= ASC_HIGH_THRESHOLD, 'Thought ID'].unique()
    strong_disagree_ids = aggregate_std_df.loc[aggregate_std_df['All_Agreement'] <= ASC_LOW_THRESHOLD, 'Thought ID'].unique()
    strong_consensus_ids = np.concatenate([
        strong_agree_ids[pd.notna(strong_agree_ids)],
        strong_disagree_ids[pd.notna(strong_disagree_ids)]
    ])

    # Debug Print
    print(f"[ASCDebug {participant_id}] Found {len(strong_agree_ids)} strong agree IDs, {len(strong_disagree_ids)} strong disagree IDs. Total: {len(strong_consensus_ids)}")

    if len(strong_consensus_ids) == 0:
        # print(f"Warning [P:{participant_id}]: No strong consensus thoughts found.")
        return np.nan

    # Get participant's votes on these specific thoughts
    participant_votes = binary_df[
        (binary_df['Participant ID'] == participant_id) &
        (binary_df['Thought ID'].isin(strong_consensus_ids))
    ].dropna(subset=['VoteNumeric'])

    # Debug Print
    print(f"[ASCDebug {participant_id}] Found {len(participant_votes)} votes by participant on consensus thoughts.")

    if participant_votes.empty:
        # print(f"[ASCDebug {participant_id}] Participant did not vote on any consensus thoughts.")
        return np.nan

    # Join votes with consensus info
    votes_with_consensus = participant_votes.merge(
        aggregate_std_df[['Thought ID', 'All_Agreement']].drop_duplicates(subset=['Thought ID']).dropna(subset=['Thought ID', 'All_Agreement']), # Ensure unique, non-null Thought ID and Agreement for merge
        on='Thought ID',
        how='left'
    )

    # Debug Print
    print(f"[ASCDebug {participant_id}] Shape after merging votes with consensus: {votes_with_consensus.shape}")
    print(f"[ASCDebug {participant_id}] Null All_Agreement count after merge: {votes_with_consensus['All_Agreement'].isnull().sum()}")

    # Check if merge failed (no matching Thought IDs with valid agreement)
    if 'All_Agreement' not in votes_with_consensus.columns or votes_with_consensus['All_Agreement'].isnull().all():
         # print(f"Warning [P:{participant_id}]: Could not merge votes with consensus (All_Agreement missing or all null). ASC score is NaN.")
         return np.nan

    # Filter out rows where merge failed to find an agreement score
    votes_with_consensus.dropna(subset=['All_Agreement'], inplace=True)
    if votes_with_consensus.empty:
        # print(f"[ASCDebug {participant_id}] No valid consensus votes remain after dropping merge NaNs.")
        return np.nan

    # Identify votes AGAINST consensus
    voted_agree_on_low = (votes_with_consensus['VoteNumeric'] == 1) & (votes_with_consensus['All_Agreement'] <= ASC_LOW_THRESHOLD)
    voted_disagree_on_high = (votes_with_consensus['VoteNumeric'] == 0) & (votes_with_consensus['All_Agreement'] >= ASC_HIGH_THRESHOLD)

    num_against_consensus = voted_agree_on_low.sum() + voted_disagree_on_high.sum()
    total_consensus_votes = len(votes_with_consensus) # Use length AFTER dropping NaNs

    # Debug Print
    print(f"[ASCDebug {participant_id}] Num against consensus: {num_against_consensus}, Total valid consensus votes: {total_consensus_votes}")

    if total_consensus_votes == 0:
        # Should be caught earlier, but as a safeguard
        return np.nan

    return num_against_consensus / total_consensus_votes


# --- Main Processing ---

def calculate_all_pri_signals(binary_df, preference_df, thought_labels_df, verbatim_map_df, aggregate_std_df):
    """
    Calculates all PRI signals for all participants.
    
    The Participant Reliability Index (PRI) combines multiple signals to assess participant
    response quality and reliability. For detailed documentation of the PRI components and
    methodology, see Data/Documentation/PRI_GUIDE.md.
    
    Args:
        binary_df: DataFrame containing binary vote data
        preference_df: DataFrame containing preference judgment data
        thought_labels_df: DataFrame containing response quality tags
        verbatim_map_df: DataFrame mapping responses to participants
        aggregate_std_df: DataFrame containing standardized aggregate data
        
    Returns:
        DataFrame containing calculated PRI signals for each participant
    """

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
    binary_df, preference_df, thought_labels_df, verbatim_map_df, aggregate_std_df, all_participant_ids = load_and_clean_data()

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
