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
    # Configure pandas to display more information
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 120)

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
        # First, try to read the first few rows to see what's in the file
        try:
            sample_df = pd.read_csv(
                AGGREGATE_STD_PATH,
                low_memory=False,
                nrows=5
            )
            print(f"Sample of aggregate standardized data (first 5 rows, first 5 columns):")
            print(sample_df.iloc[:5, :5])
        except Exception as e:
            print(f"Error reading sample from {AGGREGATE_STD_PATH}: {e}")
            
        # Now read the full file, being explicit about types to avoid conversion errors
        aggregate_std_df = pd.read_csv(
            AGGREGATE_STD_PATH,
            low_memory=False,
            dtype={
                'Question ID': str,
                'Question Type': str,
                'Question': str,
                'Response': str,
                'All': str,  # Keep percentage columns as strings initially
                'Participant ID': str
            }
        )
        
        # Create a copy to avoid fragmentation
        aggregate_std_df = aggregate_std_df.copy()
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
             # Safer percentage conversion with better error handling
             print(f"Converting percentage columns to numeric values...")
             
             # Define a function to safely convert percentage strings to floats
             def safe_pct_to_float(x):
                 if pd.isna(x):
                     return np.nan
                 try:
                     # First check if it's already a numeric value
                     if isinstance(x, (int, float)):
                         return float(x) / 100.0 if x > 1 else float(x)
                     # Then try to convert from string
                     x_str = str(x).strip()
                     if x_str.endswith('%'):
                         return float(x_str.rstrip('%')) / 100.0
                     else:
                         # If no % sign, try direct conversion but check range
                         val = float(x_str)
                         return val / 100.0 if val > 1 else val
                 except (ValueError, TypeError) as e:
                     print(f"Warning: Could not convert value '{x}' to float: {e}")
                     return np.nan
             
             # Process columns in batches to avoid fragmentation
             numeric_columns = {}
             print(f"Converting {actual_all_col} column to numeric...")
             
             # Always convert the 'All' column
             numeric_columns[f"{actual_all_col}_numeric"] = aggregate_std_df[actual_all_col].apply(safe_pct_to_float)
             
             # Sample some values to check
             sample_values = aggregate_std_df[actual_all_col].head(5).tolist()
             sample_converted = [safe_pct_to_float(x) for x in sample_values]
             print(f"Sample conversion of '{actual_all_col}' column:")
             for orig, conv in zip(sample_values, sample_converted):
                 print(f"  '{orig}' -> {conv}")
             
             # Convert segment columns in smaller batches
             batch_size = 20
             for i in range(0, len(segment_cols), batch_size):
                 batch = segment_cols[i:i+batch_size]
                 print(f"Converting batch of {len(batch)} segment columns...")
                 for col in batch:
                     numeric_col_name = f"{col}_numeric"
                     numeric_columns[numeric_col_name] = aggregate_std_df[col].apply(safe_pct_to_float)
             
             # Add all numeric columns at once
             numeric_df = pd.DataFrame(numeric_columns)
             aggregate_std_df = pd.concat([aggregate_std_df, numeric_df], axis=1)
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
        print(f"Aggregate Thought ID unique values before merge: {aggregate_std_df['Thought ID'].nunique()}")
        
        # Get a list of unique Question IDs to check overlap
        agg_questions = aggregate_std_df['Question ID'].unique()
        verbatim_questions = verbatim_map_df['Question ID'].unique()
        common_questions = set(agg_questions) & set(verbatim_questions)
        print(f"Aggregate has {len(agg_questions)} unique questions, Verbatim has {len(verbatim_questions)}")
        print(f"Common questions between datasets: {len(common_questions)}")

        # CRITICAL FIX: In the aggregate data, the unique identifier for a response is a combination of 
        # Question ID and Response, not Question ID and Participant ID
        # Let's create a direct mapping of Thought IDs from verbatim_map
        thought_id_map = verbatim_map_df[['Thought ID', 'Thought Text']].set_index('Thought ID')
        
        # Let's also check if we can safely use verbatim_map_df directly as a source of thought data
        print(f"Verbatim Map contains {len(verbatim_map_df)} rows and {verbatim_map_df['Thought ID'].nunique()} unique thoughts")
        
        # Instead of trying to merge the dataframes, we'll set the verbatim map as our source of truth
        # for ASC calculations. We'll use the aggregate data for agreement scores, but the verbatim
        # data for thought IDs and participant mapping.
        
        # Add this as a separate property to avoid modifying aggregate_std_df
        
        print("Using verbatim_map directly as source of truth for Thought IDs.")
        
        # Show sample of verbatim map to ensure it has proper structure
        print("\nSample of verbatim map data:")
        print(verbatim_map_df[['Question ID', 'Participant ID', 'Thought ID']].head())
        
        # Check for agreement scores with valid IDs
        test_join = pd.merge(
            verbatim_map_df[['Question ID', 'Thought ID']],
            aggregate_std_df[['Question ID', 'All_Agreement']],
            on='Question ID',
            how='inner'
        )
        print(f"\nJoining verbatim and aggregate on Question ID produces {len(test_join)} matched rows")
        
        # If we have valid data, we can proceed with the ASC calculation
        if len(test_join) > 0:
            print("Proceeding with ASC calculation using verbatim_map as source of truth")
        else:
            print("WARNING: No matching Question IDs between verbatim and aggregate. ASC calculation may fail!")
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


def calculate_asc_score(participant_id, binary_df, aggregate_std_df, verbatim_map_df=None):
    """
    Calculates the participant's rate of voting AGAINST strong consensus.
    (Lower is better, so this raw score needs inversion later).
    NOTE: Uses verbatim_map_df as a source of thought IDs when available
    """
    print(f"[ASCDebug {participant_id}] Calculating ASC score...")

    # Check for verbatim_map - if provided, use it to get thought IDs
    if verbatim_map_df is not None and not verbatim_map_df.empty:
        # Create a mapping of question ID to agreement score from aggregate data
        question_agreement_map = {}
        
        # First, check if aggregate_std_df has the required columns
        if 'All_Agreement' not in aggregate_std_df.columns:
            print(f"[ASCDebug {participant_id}] Warning: 'All_Agreement' column not found in aggregate_std_df")
            print(f"[ASCDebug {participant_id}] Available columns: {aggregate_std_df.columns[:10]}...")
            # Create some placeholder agreement scores for testing (all will be neutral)
            # This allows us to test the rest of the processing pipeline
            print(f"[ASCDebug {participant_id}] Creating placeholder agreement scores of 0.5 for all questions")
            question_ids = verbatim_map_df['Question ID'].unique()
            for qid in question_ids:
                question_agreement_map[qid] = 0.5
        else:
            # Normal processing with real agreement scores
            for _, row in aggregate_std_df.iterrows():
                if 'Question ID' in row and 'All_Agreement' in row and pd.notna(row['All_Agreement']):
                    question_id = row['Question ID']
                    agreement = row['All_Agreement']
                    # For each question, store the highest agreement score we find
                    if question_id in question_agreement_map:
                        question_agreement_map[question_id] = max(question_agreement_map[question_id], agreement)
                    else:
                        question_agreement_map[question_id] = agreement
        
        print(f"[ASCDebug {participant_id}] Created agreement map for {len(question_agreement_map)} questions")
        
        # For each thought in verbatim map, get its question's agreement score
        thought_agreement_pairs = []
        for _, row in verbatim_map_df.iterrows():
            thought_id = row['Thought ID']
            question_id = row['Question ID']
            if question_id in question_agreement_map:
                thought_agreement_pairs.append((thought_id, question_agreement_map[question_id]))
        
        # Create a DataFrame with these pairs for easier filtering
        thought_agreement_df = pd.DataFrame(thought_agreement_pairs, columns=['Thought ID', 'All_Agreement'])
        
        # Filter for strong consensus thoughts
        strong_agree_df = thought_agreement_df[thought_agreement_df['All_Agreement'] >= ASC_HIGH_THRESHOLD]
        strong_disagree_df = thought_agreement_df[thought_agreement_df['All_Agreement'] <= ASC_LOW_THRESHOLD]
        
        strong_agree_ids = strong_agree_df['Thought ID'].unique()
        strong_disagree_ids = strong_disagree_df['Thought ID'].unique()
        
        print(f"[ASCDebug {participant_id}] Using verbatim map as source of Thought IDs")
        print(f"[ASCDebug {participant_id}] Mapped {len(thought_agreement_df)} thoughts to agreement scores")
    else:
        # Fallback to using aggregate_std_df directly
        print(f"[ASCDebug {participant_id}] No verbatim_map provided, using aggregate_std_df")
        valid_consensus_df = aggregate_std_df[aggregate_std_df['Thought ID'].notna()].copy()
        
        if valid_consensus_df.empty:
            print(f"[ASCDebug {participant_id}] No valid Thought IDs found. Unable to calculate ASC score.")
            return np.nan
        
        strong_agree_df = valid_consensus_df[valid_consensus_df['All_Agreement'] >= ASC_HIGH_THRESHOLD]
        strong_disagree_df = valid_consensus_df[valid_consensus_df['All_Agreement'] <= ASC_LOW_THRESHOLD]
        
        strong_agree_ids = strong_agree_df['Thought ID'].unique()
        strong_disagree_ids = strong_disagree_df['Thought ID'].unique()
    
    # Debug Print
    print(f"[ASCDebug {participant_id}] High consensus rows (All_Agreement >= {ASC_HIGH_THRESHOLD}): {len(strong_agree_df)} rows")
    print(f"[ASCDebug {participant_id}] Low consensus rows (All_Agreement <= {ASC_LOW_THRESHOLD}): {len(strong_disagree_df)} rows")
    
    # Combine strong consensus IDs
    strong_consensus_ids = np.concatenate([
        strong_agree_ids if len(strong_agree_ids) > 0 else np.array([]),
        strong_disagree_ids if len(strong_disagree_ids) > 0 else np.array([])
    ]) if (len(strong_agree_ids) > 0 or len(strong_disagree_ids) > 0) else np.array([])
    
    # Debug Print
    print(f"[ASCDebug {participant_id}] Found {len(strong_agree_ids)} strong agree IDs, {len(strong_disagree_ids)} strong disagree IDs. Total: {len(strong_consensus_ids)}")
    
    # Return NaN if no consensus thoughts were found
    if len(strong_consensus_ids) == 0:
        print(f"[ASCDebug {participant_id}] No strong consensus thoughts found.")
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

    # Join votes with consensus info - using the appropriate source
    if verbatim_map_df is not None and not verbatim_map_df.empty:
        # Create an agreement map for direct lookup
        thought_agreement_map = dict(thought_agreement_pairs)
        
        # Add agreement scores directly to votes
        votes_with_consensus = participant_votes.copy()
        votes_with_consensus['All_Agreement'] = votes_with_consensus['Thought ID'].map(thought_agreement_map)
    else:
        # Fallback to using aggregate_std_df for merge
        votes_with_consensus = participant_votes.merge(
            valid_consensus_df[['Thought ID', 'All_Agreement']].drop_duplicates(subset=['Thought ID']),
            on='Thought ID',
            how='left'
        )
    
    # Debug Print
    print(f"[ASCDebug {participant_id}] Shape after adding agreement scores: {votes_with_consensus.shape}")
    print(f"[ASCDebug {participant_id}] Null All_Agreement count: {votes_with_consensus['All_Agreement'].isna().sum()}")
    
    # Check if we have valid agreement scores
    if 'All_Agreement' not in votes_with_consensus.columns or votes_with_consensus['All_Agreement'].isna().all():
        print(f"[ASCDebug {participant_id}] No agreement scores available for participant's votes.")
        return np.nan
    
    # Filter out rows with missing agreement scores
    votes_with_consensus = votes_with_consensus.dropna(subset=['All_Agreement'])
    if votes_with_consensus.empty:
        print(f"[ASCDebug {participant_id}] No valid consensus votes remain after dropping null agreements.")
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

    # Make a copy of the DataFrame to avoid fragmentation warnings
    aggregate_std_df = aggregate_std_df.copy()
    

    for participant_id in all_participant_ids:
        # 1. Duration
        duration = calculate_duration(participant_id, binary_times_df, preference_times_df)

        # 2. Low Quality Tags %
        low_quality_perc = calculate_low_quality_tag_perc(participant_id, thought_labels_df)

        # 3. Universal Disagreement %
        universal_disagreement_perc = calculate_universal_disagreement_perc(participant_id, verbatim_map_df, aggregate_std_df)

        # 4. ASC Score (raw - lower is better)
        asc_raw = calculate_asc_score(participant_id, binary_df, aggregate_std_df, verbatim_map_df)

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
    essential_dfs = [binary_df, preference_df, verbatim_map_df]
    if any(df.empty for df in essential_dfs):
        print("Error during data loading of essential datasets. Exiting.")
        exit(1)
    
    # Verify we have the necessary data in verbatim_map
    if 'Participant ID' not in verbatim_map_df.columns or 'Thought ID' not in verbatim_map_df.columns:
        print("Error: verbatim_map_df is missing required columns. Exiting.")
        exit(1)
    
    # If aggregate has issues but verbatim_map is good, we can still calculate some metrics
    if aggregate_std_df.empty:
        print("Warning: aggregate_std_df is empty, but will attempt to continue using verbatim_map.")
    elif 'All_Agreement' not in aggregate_std_df.columns:
        print("Warning: 'All_Agreement' column not found in aggregate_std_df.")
        print("This will affect ASC score calculation, but other metrics can still be computed.")


    # 2. Calculate Raw Signals
    pri_signals_df = calculate_all_pri_signals(
        binary_df, preference_df, thought_labels_df, verbatim_map_df, aggregate_std_df
    )

    print("Raw PRI Signals Head:")
    print(pri_signals_df.head())

    # --- Phase 3: Normalization and Final PRI ---
    print("\nApplying normalization and calculating final PRI scores...")
    
    # Check how many NaN values we have in each column
    print("\nNaN counts in raw signals:")
    print(pri_signals_df[['Duration_seconds', 'LowQualityTag_Perc', 'UniversalDisagreement_Perc', 'ASC_Score_Raw']].isna().sum())
    
    # Simple min-max normalization function
    def min_max_normalize(series, invert=False):
        """Min-max normalization with optional inversion"""
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
            normalized = (filled_series - min_val) / (max_val - min_val)
            
        # Invert if needed (for metrics where lower raw value is better)
        if invert:
            normalized = 1 - normalized
            
        # Restore NaN values
        normalized[series.isna()] = np.nan
        
        return normalized
    
    # Normalize duration (longer duration is better)
    pri_signals_df['Duration_Norm'] = min_max_normalize(pri_signals_df['Duration_seconds'])
    
    # Normalize low quality tags (lower percentage is better, so invert)
    pri_signals_df['LowQualityTag_Norm'] = min_max_normalize(pri_signals_df['LowQualityTag_Perc'], invert=True)
    
    # Normalize universal disagreement (lower percentage is better, so invert)
    pri_signals_df['UniversalDisagreement_Norm'] = min_max_normalize(pri_signals_df['UniversalDisagreement_Perc'], invert=True)
    
    # Check if we have valid ASC scores
    asc_available = not pri_signals_df['ASC_Score_Raw'].isna().all()
    
    if asc_available:
        # Normal calculation with ASC
        pri_signals_df['ASC_Norm'] = min_max_normalize(pri_signals_df['ASC_Score_Raw'], invert=True)
        
        # Define weights for each component
        weights = {
            'Duration_Norm': 0.2,
            'LowQualityTag_Norm': 0.3,
            'UniversalDisagreement_Norm': 0.3,
            'ASC_Norm': 0.2
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
        adjusted_weights = {
            'Duration_Norm': 0.25,
            'LowQualityTag_Norm': 0.375,
            'UniversalDisagreement_Norm': 0.375
        }
        
        # Calculate final weighted PRI score without ASC
        pri_signals_df['PRI_Score'] = (
            pri_signals_df['Duration_Norm'] * adjusted_weights['Duration_Norm'] +
            pri_signals_df['LowQualityTag_Norm'] * adjusted_weights['LowQualityTag_Norm'] +
            pri_signals_df['UniversalDisagreement_Norm'] * adjusted_weights['UniversalDisagreement_Norm']
        )
    
    # Create a 1-5 scale version for easier interpretation
    pri_signals_df['PRI_Scale_1_5'] = pri_signals_df['PRI_Score'] * 4 + 1
    
    print("PRI Score Statistics:")
    print(pri_signals_df[['PRI_Score', 'PRI_Scale_1_5']].describe())
    print("\nTop 5 Most Reliable Participants:")
    print(pri_signals_df.sort_values('PRI_Score', ascending=False).head(5)[['Participant ID', 'PRI_Score', 'PRI_Scale_1_5']])
    print("\nBottom 5 Least Reliable Participants:")
    print(pri_signals_df.sort_values('PRI_Score', ascending=True).head(5)[['Participant ID', 'PRI_Score', 'PRI_Scale_1_5']])
    
    # Save results to CSV
    output_path = f"{DATA_DIR}/GD{GD_NUMBER}_pri_scores.csv"
    pri_signals_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
