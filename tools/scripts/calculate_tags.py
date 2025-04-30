import pandas as pd
import numpy as np
import os
import argparse
import logging
import re
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
TAG_COL_PATTERN = re.compile(r'^Tag \d+$')

# --- Helper Functions ---

def safe_read_csv(path, **kwargs):
    """Reads a CSV, attempting to detect and skip initial metadata rows using heuristics."""
    try:
        # Try reading normally first
        return pd.read_csv(path, **kwargs)
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        logging.warning(f"{type(e).__name__} reading {path}: {e}. Attempting to find header...")

        header_row = -1
        min_commas_for_header = 5 # Heuristic: header row should have at least this many fields
        # Define essential columns that MUST be in the header
        essential_markers = {
            'participants': ['Participant Id'], # Focus on the most critical ID
            'labels': ['Question ID', 'Participant ID', 'ResponseText'],
            'categories': ['Question ID', 'Category'],
            'guide': ['Item type (dropdown)', 'Content'],
            'aggregate': ['Question ID', 'Question Type', 'Question'], # Standardized aggregate
        }
        markers = []
        filename = Path(path).name
        for key, value in essential_markers.items():
            # Use startswith for better matching (e.g., GD3_participants.csv)
            if filename.startswith(key) or key in filename:
                 markers = value
                 break
        if not markers:
            markers = ['Question ID'] # Generic fallback

        encodings_to_try = [kwargs.get('encoding', 'utf-8-sig'), 'utf-8', 'latin1', 'iso-8859-1']
        detected_encoding = None

        for enc in encodings_to_try:
            try:
                with open(path, 'r', encoding=enc) as f:
                    for i, line in enumerate(f):
                        # Check 1: Does it have enough commas?
                        if line.count(',') >= min_commas_for_header:
                            # Check 2: Does it contain all essential markers?
                            if all(marker in line for marker in markers):
                                header_row = i
                                detected_encoding = enc
                                logging.info(f"Detected potential header on line {header_row+1} in {path} using encoding '{detected_encoding}' and markers: {markers}")
                                break # Found potential header
                if header_row != -1:
                    break # Stop trying encodings if header found
            except UnicodeDecodeError:
                logging.warning(f"Encoding {enc} failed for {path}, trying next...")
                continue
            except Exception as file_read_error:
                 logging.error(f"Unexpected error reading {path} with encoding {enc}: {file_read_error}")
                 break

        if header_row != -1 and detected_encoding:
            try:
                kwargs['encoding'] = detected_encoding
                # --- Read with detected header and VALIDATE columns ---
                df_temp = pd.read_csv(path, header=header_row, **kwargs)
                # Validate: Check if essential markers are actual columns
                if all(marker in df_temp.columns for marker in markers):
                    logging.info(f"Header validated. Columns look reasonable: {df_temp.columns[:5].tolist()}...")
                    return df_temp
                else:
                    logging.error(f"Header validation failed! Detected header on line {header_row+1} but loaded columns ({df_temp.columns.tolist()}) don't contain all markers {markers}. Check file structure or markers.")
                    # Optional: Could try header_row-1 here, but it gets complex.
                    # For now, raise the original error as detection failed.
                    raise e
            except Exception as read_error:
                 logging.error(f"Error reading {path} even after detecting header at line {header_row+1}: {read_error}")
                 raise read_error
        else:
            logging.error(f"Could not reliably detect header row in {path} using markers {markers} and comma count heuristic.")
            raise e # Re-raise the original error

    except Exception as general_error:
         logging.error(f"Unexpected error reading {path}: {general_error}")
         raise general_error


# --- Core Logic Functions ---

def load_and_prep_data(gd_number, data_dir):
    """Loads and preprocesses all necessary input files."""
    logging.info("Loading and preparing data...")
    paths = {
        'labels': data_dir / f"GD{gd_number}" / "tags" / f"all_thought_labels.csv",
        'categories': data_dir / f"GD{gd_number}" / "tags" / f"all_tag_categories.csv",
        'participants': data_dir / f"GD{gd_number}" / f"GD{gd_number}_participants.csv",
        'aggregate': data_dir / f"GD{gd_number}" / f"GD{gd_number}_aggregate_standardized.csv",
        'discussion_guide': data_dir / f"GD{gd_number}" / f"GD{gd_number}_discussion_guide.csv", # Needed to identify segment columns
    }

    # Check if files exist
    for name, path in paths.items():
        if not path.exists():
            logging.error(f"Required input file not found: {path}")
            return None # Indicate failure

    try:
        # 1. Load Thought Labels and Melt
        logging.info(f"Loading {paths['labels']}...")
        labels_df = safe_read_csv(paths['labels'], encoding='utf-8-sig')
        logging.info(f"  Loaded {len(labels_df)} rows.")
        # Clean column names (remove BOM/extra spaces if any)
        labels_df.columns = labels_df.columns.str.replace('^\ufeff', '', regex=True).str.strip()
        # Identify tag columns
        tag_cols = [col for col in labels_df.columns if TAG_COL_PATTERN.match(col)]
        id_vars = [col for col in labels_df.columns if col not in tag_cols]
        if not tag_cols:
            logging.error("No columns matching 'Tag \\d+' found in all_thought_labels.csv")
            return None

        logging.info(f"Melting tag columns: {tag_cols}")
        melted_labels = pd.melt(labels_df,
                                id_vars=id_vars,
                                value_vars=tag_cols,
                                var_name='Tag Source',
                                value_name='Tag')
        # Drop rows where Tag is NaN/None/empty and clean Tag text
        melted_labels.dropna(subset=['Tag'], inplace=True)
        melted_labels['Tag'] = melted_labels['Tag'].astype(str).str.strip()
        melted_labels = melted_labels[melted_labels['Tag'] != '']
        logging.info(f"  Melted to {len(melted_labels)} tag instances.")
        # Ensure correct types
        melted_labels['Participant ID'] = melted_labels['Participant ID'].astype(str)
        melted_labels['Question ID'] = melted_labels['Question ID'].astype(str)


        # 2. Load Tag Categories
        logging.info(f"Loading {paths['categories']}...")
        categories_df = safe_read_csv(paths['categories'], encoding='utf-8-sig')
        logging.info(f"  Loaded {len(categories_df)} rows.")
        categories_df.columns = categories_df.columns.str.replace('^\ufeff', '', regex=True).str.strip()
        # Identify tag columns in categories file to melt *its* tags for joining
        cat_tag_cols = [col for col in categories_df.columns if TAG_COL_PATTERN.match(col)]
        cat_id_vars = ['Question ID', 'Category'] # Assume these are the key identifiers
        if not all(c in categories_df.columns for c in cat_id_vars):
             logging.error(f"Missing required columns ('Question ID', 'Category') in {paths['categories']}")
             return None
        if not cat_tag_cols:
            logging.error(f"No columns matching 'Tag \\d+' found in {paths['categories']}")
            return None

        logging.info("Melting category tag columns...")
        melted_categories = pd.melt(categories_df,
                                    id_vars=cat_id_vars,
                                    value_vars=cat_tag_cols,
                                    value_name='Tag')
        melted_categories.dropna(subset=['Tag'], inplace=True)
        melted_categories['Tag'] = melted_categories['Tag'].astype(str).str.strip()
        melted_categories = melted_categories[melted_categories['Tag'] != '']
        # Keep only necessary columns and drop duplicates just in case
        category_map = melted_categories[['Question ID', 'Tag', 'Category']].drop_duplicates()
        logging.info(f"  Created category map with {len(category_map)} unique QID-Tag-Category entries.")
        category_map['Question ID'] = category_map['Question ID'].astype(str)

        # Merge Category onto Melted Labels
        logging.info("Merging categories onto labels...")
        analysis_df = pd.merge(melted_labels, category_map, on=['Question ID', 'Tag'], how='left')
        missing_cats = analysis_df['Category'].isnull().sum()
        if missing_cats > 0:
            logging.warning(f"{missing_cats} tag instances could not be mapped to a category. Check consistency between labels and categories files.")
            # Optionally fill missing categories
            # analysis_df['Category'].fillna('Uncategorized', inplace=True)
        logging.info(f"  Merge complete. Current rows: {len(analysis_df)}")

        # 3. Load Participants Data for Segments
        logging.info(f"Loading {paths['participants']}...")
        participants_df = safe_read_csv(paths['participants'], encoding='utf-8-sig', low_memory=False)
        logging.info(f"  Loaded {len(participants_df)} rows.")
        # --- DEBUG: Print loaded columns ---
        logging.info(f"  Columns loaded from participants file: {participants_df.columns.tolist()}")
        # --- END DEBUG ---
        participants_df.columns = participants_df.columns.str.replace('^\ufeff', '', regex=True).str.strip()
        # Ensure Participant Id is string for merging
        if 'Participant Id' not in participants_df.columns:
             logging.error("'Participant Id' column not found in participants file.")
             return None
        participants_df['Participant Id'] = participants_df['Participant Id'].astype(str)

        # Identify Segment Columns - Use Discussion Guide
        logging.info(f"Loading {paths['discussion_guide']} to identify segment questions...")
        guide_df = safe_read_csv(paths['discussion_guide'], encoding='utf-8-sig')
        guide_df.columns = guide_df.columns.str.replace('^\ufeff', '', regex=True).str.strip()
        # Segment questions are typically 'onboarding single select' or potentially early 'poll single select'
        # We need their 'Content' (which matches the column header in participants_df)
        segment_q_types = ['onboarding single select', 'poll single select'] # Add more if needed
        segment_questions = guide_df[guide_df['Item type (dropdown)'].isin(segment_q_types)]['Content'].tolist()

        # Filter participants_df to keep only Participant Id and actual segment columns present
        segment_cols_to_keep = [col for col in participants_df.columns if col in segment_questions]
        if not segment_cols_to_keep:
            logging.warning("Could not identify any segment columns in participants file based on discussion guide. Segment frequency analysis will be limited.")
            participants_segments = participants_df[['Participant Id']].copy() # Keep only ID
        else:
            logging.info(f"Identified segment columns from participants file: {segment_cols_to_keep}")
            participants_segments = participants_df[['Participant Id'] + segment_cols_to_keep].copy()

        # Merge Segments onto Analysis DF
        logging.info("Merging participant segments onto analysis data...")
        # Rename participant ID column for merge
        analysis_df.rename(columns={'Participant ID': 'Participant Id'}, inplace=True)
        analysis_df = pd.merge(analysis_df, participants_segments, on='Participant Id', how='left')
        logging.info(f"  Merge complete. Current rows: {len(analysis_df)}")


        # 4. Load Standardized Aggregate for Agreement Scores & Question Text
        logging.info(f"Loading {paths['aggregate']}...")
        agg_df = pd.read_csv(paths['aggregate'], encoding='utf-8', low_memory=False) # Assume preprocess_aggregate handles BOM
        logging.info(f"  Loaded {len(agg_df)} rows.")
        # Keep only relevant columns: QID, PID (author), Question Text, Agreement (All)
        # Find the 'All' agreement column (might have varying N)
        all_col_pattern = re.compile(r'^All\s*\((\d+)\)\s*$')
        all_agreement_col = None
        for col in agg_df.columns:
            if all_col_pattern.match(col):
                all_agreement_col = col
                break
        if not all_agreement_col:
            logging.error("Could not find 'All (N)' agreement column in aggregate_standardized.csv")
            return None
        logging.info(f"Found agreement column: {all_agreement_col}")

        cols_to_keep_agg = ['Question ID', 'Participant ID', 'Question', all_agreement_col]
        if not all(c in agg_df.columns for c in ['Question ID', 'Participant ID', 'Question']):
             logging.error("Missing required columns ('Question ID', 'Participant ID', 'Question') in aggregate_standardized.csv")
             return None

        agg_subset = agg_df[cols_to_keep_agg].copy()
        # Ensure types for merging
        agg_subset['Question ID'] = agg_subset['Question ID'].astype(str)
        agg_subset['Participant ID'] = agg_subset['Participant ID'].astype(str)
        agg_subset.rename(columns={'Question': 'Question Text',
                                   'Participant ID': 'Participant Id', # Match name for merge
                                   all_agreement_col: 'Agreement Score'}, inplace=True)

        # Parse agreement score
        # Use the helper function defined in analyze_dialogues.py or replicate it here
        def parse_percentage(value):
            if isinstance(value, (int, float)):
                return float(value) if not np.isnan(value) else np.nan
            if isinstance(value, str):
                value = value.strip()
                if value == '-' or value == '': return np.nan
                if value.endswith('%'):
                    try: return float(value[:-1]) / 100.0
                    except ValueError: return np.nan
            return np.nan
        agg_subset['Agreement Score'] = agg_subset['Agreement Score'].apply(parse_percentage)

        # Merge Agreement Score onto Analysis DF
        logging.info("Merging agreement scores onto analysis data...")
        # We need to merge based on the response author (Participant Id) and Question ID
        analysis_df = pd.merge(analysis_df, agg_subset, on=['Question ID', 'Participant Id'], how='left')
        missing_agg = analysis_df['Agreement Score'].isnull().sum()
        if missing_agg > 0:
             # This could happen if agg_standardized doesn't have rows for *every* response author
             # (e.g., maybe only Ask Opinion?) - check assumptions if this occurs.
             logging.warning(f"{missing_agg} tag instances could not be mapped to an agreement score.")
        logging.info(f"  Merge complete. Final rows for analysis: {len(analysis_df)}")

        # Keep only necessary final columns before analysis
        final_cols = ['Question ID', 'Question Text', 'Category', 'Tag', 'Participant Id', 'Sentiment'] + segment_cols_to_keep + ['Agreement Score']
        # Handle case where Question Text might be missing after merge if agg data was incomplete
        if 'Question Text_x' in analysis_df.columns: # Pandas adds suffixes on merge conflicts
             analysis_df['Question Text'] = analysis_df['Question Text_x'].fillna(analysis_df['Question Text_y'])
             final_cols = [c.replace('_x','').replace('_y','') for c in final_cols] # Adjust list if needed
             analysis_df = analysis_df[[c for c in final_cols if c in analysis_df.columns]]
        else:
             analysis_df = analysis_df[[c for c in final_cols if c in analysis_df.columns]]


        logging.info(f"Data preparation complete. Final columns: {analysis_df.columns.tolist()}")
        return analysis_df, segment_cols_to_keep # Return segment cols for frequency calc

    except FileNotFoundError as e:
        logging.error(f"File not found during loading: {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred during data loading/preparation: {e}", exc_info=True)
        return None


def calculate_unified_report(df, segment_columns, output_path):
    """Calculates aggregated metrics and segment frequencies into a single wide report."""
    logging.info("Calculating unified tag analysis report...")
    if df is None or df.empty:
        logging.warning("Input DataFrame is empty, cannot generate report.")
        return

    # Define aggregation functions for metrics
    agg_funcs = {
        'Participant Id': lambda x: x.nunique(), # Count unique participants for N Responses
        'Agreement Score': ['mean', 'median'],
        'Sentiment': [
            lambda x: (x == 'Positive').sum() / x.count() if x.count() > 0 else 0,
            lambda x: (x == 'Negative').sum() / x.count() if x.count() > 0 else 0
        ]
    }

    # Group by QID, Text, Category, Tag to get base metrics
    logging.info("Calculating base metrics (agreement, sentiment)...")
    grouped = df.groupby(['Question ID', 'Question Text', 'Category', 'Tag'], observed=True, dropna=False)
    metrics_df = grouped.agg(agg_funcs)

    # Flatten MultiIndex columns and rename
    metrics_df.columns = ['_'.join(col).strip() for col in metrics_df.columns.values]
    metrics_df.rename(columns={
        'Participant Id_<lambda>': 'N Responses',
        'Agreement Score_mean': 'Avg Agreement (All)',
        'Agreement Score_median': 'Median Agreement (All)',
        'Sentiment_<lambda_0>': 'Proportion Positive Sentiment',
        'Sentiment_<lambda_1>': 'Proportion Negative Sentiment'
    }, inplace=True)
    logging.info(f"Calculated base metrics for {len(metrics_df)} QID-Tag combinations.")

    # --- Calculate Segment Frequencies ---
    logging.info("Calculating segment frequencies...")
    all_frequency_dfs = [metrics_df] # Start with metrics

    # Calculate total frequency ('All (Frequency)') - same as 'N Responses'
    metrics_df['All (Frequency)'] = metrics_df['N Responses']

    # Calculate frequency for each segment column
    if segment_columns:
        for seg_col in segment_columns:
            logging.info(f"  Calculating frequencies for segment: {seg_col}")
            # Group by QID, Tag AND the segment value column
            # Use observed=True to handle categorical data efficiently if present
            freq_grouped = df.groupby(['Question ID', 'Tag', seg_col], observed=True, dropna=False)
            # Count unique Participant Ids within each group
            segment_freq = freq_grouped['Participant Id'].nunique().reset_index()
            segment_freq.rename(columns={'Participant Id': 'Frequency'}, inplace=True)

            # Pivot to wide format: QID, Tag as index, Segment Values as columns
            try:
                # Construct new column names like "Segment Name: Value (Frequency)"
                # Need to handle potential non-string segment values if any exist
                segment_freq['Segment Header'] = segment_freq[seg_col].apply(
                    lambda x: f"{seg_col}: {str(x)} (Frequency)"
                )
                pivot_df = segment_freq.pivot_table(index=['Question ID', 'Tag'],
                                                   columns='Segment Header',
                                                   values='Frequency',
                                                   fill_value=0) # Fill missing segment values with 0 count
                all_frequency_dfs.append(pivot_df)
                logging.info(f"    Pivoted frequencies for {seg_col}.")
            except Exception as e:
                 logging.error(f"    Error pivoting frequencies for segment '{seg_col}': {e}. Skipping this segment.", exc_info=True)
    else:
        logging.warning("No segment columns identified; skipping segment frequency calculation.")


    # --- Merge Metrics and Frequencies ---
    logging.info("Merging all metrics and frequencies...")
    try:
        # Start with metrics_df (which includes 'All (Frequency)')
        final_report_df = metrics_df
        # Sequentially merge pivoted frequency tables
        for freq_df in all_frequency_dfs[1:]: # Skip the first element (metrics_df itself)
             # Ensure indices match before merging
            final_report_df = final_report_df.merge(freq_df, left_index=True, right_index=True, how='left')

        # Fill NaN values in frequency columns with 0 (if any slipped through)
        freq_cols = [col for col in final_report_df.columns if '(Frequency)' in col]
        final_report_df[freq_cols] = final_report_df[freq_cols].fillna(0).astype(int)

        # Reset index to bring QID, Tag etc. back as columns
        final_report_df.reset_index(inplace=True)

        # Reorder columns for clarity
        id_cols = ['Question ID', 'Question Text', 'Category', 'Tag']
        metric_cols = ['N Responses', 'Avg Agreement (All)', 'Median Agreement (All)',
                       'Proportion Positive Sentiment', 'Proportion Negative Sentiment']
        # Dynamically get all frequency columns, sort them alphabetically
        freq_cols_sorted = sorted([col for col in final_report_df.columns if '(Frequency)' in col])

        final_columns_order = id_cols + metric_cols + freq_cols_sorted
        # Ensure all expected columns are present before reordering
        final_columns_order = [col for col in final_columns_order if col in final_report_df.columns]
        final_report_df = final_report_df[final_columns_order]

        logging.info(f"Final report calculated with {len(final_report_df)} rows and {len(final_report_df.columns)} columns.")

        # Save the report
        logging.info(f"Saving unified report to: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_report_df.to_csv(output_path, index=False, float_format='%.4f')
        logging.info("Unified report saved successfully.")

    except Exception as e:
        logging.error(f"Error merging/saving the final report: {e}", exc_info=True)


def generate_cooccurrence_heatmap(df, output_path):
    """Calculates tag co-occurrence and saves a heatmap."""
    logging.info("Calculating tag co-occurrence matrix...")
    if df is None or df.empty or 'Tag' not in df.columns or 'Participant Id' not in df.columns or 'Question ID' not in df.columns:
        logging.warning("Input DataFrame is missing required columns for co-occurrence analysis. Skipping.")
        return

    # Group tags by response (QID + PID)
    # Using observed=True for potential performance gain with categoricals
    response_tags = df.groupby(['Question ID', 'Participant Id'], observed=True)['Tag'].apply(list)

    # Generate pairs of tags for each response (ignore order, count A-B same as B-A)
    tag_pairs = []
    for tags in response_tags:
        # Create combinations of 2 tags for responses with multiple tags
        if len(tags) >= 2:
            # Sort tags within the list to ensure ('A','B') is treated the same as ('B','A')
            sorted_tags = sorted(list(set(tags))) # Use set to handle potential duplicates within a response just in case
            tag_pairs.extend(itertools.combinations(sorted_tags, 2))

    if not tag_pairs:
        logging.warning("No co-occurring tag pairs found. Skipping heatmap generation.")
        return

    # Count frequency of each pair
    pair_counts = pd.Series(tag_pairs).value_counts()
    logging.info(f"Found {len(pair_counts)} unique co-occurring tag pairs.")

    # Convert pair counts to DataFrame for pivoting
    cooccurrence_df = pair_counts.reset_index()
    cooccurrence_df.columns = ['Tag Pair', 'Frequency']
    cooccurrence_df[['Tag A', 'Tag B']] = pd.DataFrame(cooccurrence_df['Tag Pair'].tolist(), index=cooccurrence_df.index)

    # Create the matrix (pivot table)
    # Include both A->B and B->A for a symmetric matrix
    matrix_a_b = cooccurrence_df.pivot_table(index='Tag A', columns='Tag B', values='Frequency', fill_value=0)
    matrix_b_a = cooccurrence_df.pivot_table(index='Tag B', columns='Tag A', values='Frequency', fill_value=0)

    # Combine and ensure symmetry, filling NaNs
    cooccurrence_matrix = matrix_a_b.add(matrix_b_a, fill_value=0)

    # Ensure all tags are in both index and columns
    all_tags_in_pairs = sorted(list(set(cooccurrence_df['Tag A']) | set(cooccurrence_df['Tag B'])))
    cooccurrence_matrix = cooccurrence_matrix.reindex(index=all_tags_in_pairs, columns=all_tags_in_pairs, fill_value=0)

    # Fill diagonal with NaN or 0 (conventionally diagonal is ignored in co-occurrence)
    np.fill_diagonal(cooccurrence_matrix.values, 0) # Or np.nan if preferred

    # Optional: Filter matrix for stronger relationships if it's too large
    # e.g., keep only pairs with frequency > threshold
    # Or only plot Top N tags by total co-occurrence frequency

    if cooccurrence_matrix.empty:
        logging.warning("Co-occurrence matrix is empty after processing. Skipping heatmap.")
        return

    # --- Generate Heatmap ---
    logging.info("Generating co-occurrence heatmap...")
    num_tags = len(cooccurrence_matrix)
    # Dynamic sizing - adjust multipliers as needed
    fig_size_factor = 0.5
    min_fig_size = 8
    fig_width = max(min_fig_size, num_tags * fig_size_factor)
    fig_height = max(min_fig_size, num_tags * fig_size_factor)
    # Adjust font size based on number of tags
    annot_size = max(5, 10 - num_tags * 0.1)
    tick_font_size = max(6, 11 - num_tags * 0.1)


    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(cooccurrence_matrix, cmap="viridis", annot=False, fmt="d") # Annotations might be too dense
    plt.title(f'Tag Co-occurrence Matrix (GD{args.gd_number})', fontsize=14)
    plt.xlabel("Tag", fontsize=tick_font_size)
    plt.ylabel("Tag", fontsize=tick_font_size)
    plt.xticks(rotation=90, fontsize=tick_font_size)
    plt.yticks(rotation=0, fontsize=tick_font_size)
    plt.tight_layout()

    # Save the heatmap
    logging.info(f"Saving co-occurrence heatmap to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info("Heatmap saved successfully.")
    except Exception as e:
        logging.error(f"Error saving heatmap: {e}")
    plt.close() # Close the plot to free memory


def main(args):
    """Main execution flow."""
    data_dir = Path("./Data") # Assumes script is run from workspace root
    output_base_dir = Path("./analysis_output")
    output_dir = output_base_dir / f"GD{args.gd_number}" / "tags"
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    logging.info(f"Starting tag analysis for GD{args.gd_number}")
    logging.info(f"Data Directory: {data_dir.resolve()}")
    logging.info(f"Output Directory: {output_dir.resolve()}")

    # 1. Load and Prepare Data
    prepared_data, segment_columns = load_and_prep_data(args.gd_number, data_dir)

    if prepared_data is None:
        logging.error("Data loading failed. Exiting.")
        return

    # 2. Calculate Unified Report
    report_path = output_dir / "tag_analysis_report.csv"
    calculate_unified_report(prepared_data, segment_columns, report_path)

    # 3. Generate Co-occurrence Heatmap
    heatmap_path = output_dir / "tag_cooccurrence_heatmap.png"
    generate_cooccurrence_heatmap(prepared_data, heatmap_path)

    logging.info(f"Tag analysis for GD{args.gd_number} completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Remesh tag data for Global Dialogues.")
    parser.add_argument("gd_number", type=int, help="Global Dialogue cadence number (e.g., 3).")
    # Add other arguments as needed (e.g., input/output directory overrides)

    args = parser.parse_args()
    main(args) 