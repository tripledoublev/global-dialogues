# Placeholder for indicator analysis script
import argparse
import logging
import os
import re
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lib.analysis_utils import load_standardized_data, parse_percentage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Semantic Ordering for Responses ---
# Defines the desired order for common response sets on the heatmap x-axis.
RESPONSE_ORDERS = {
    # Trust Scale (Distrust -> Trust)
    tuple(sorted(("Strongly Distrust", "Somewhat Distrust", "Neither Trust Nor Distrust", "Somewhat Trust", "Strongly Trust"))):
        ["Strongly Distrust", "Somewhat Distrust", "Neither Trust Nor Distrust", "Somewhat Trust", "Strongly Trust"],
    # Frequency Scale (Least -> Most Frequent)
    tuple(sorted(("daily", "weekly", "monthly", "annually", "never"))):
        ["never", "annually", "monthly", "weekly", "daily"],
    # Societal Impact Scale (Risk -> Benefit)
    tuple(sorted(("Risks far outweigh benefits", "Risks slightly outweigh benefits", "Risks and benefits are equal", "Benefits slightly outweigh risks", "Benefits far outweigh risks"))):
        ["Risks far outweigh benefits", "Risks slightly outweigh benefits", "Risks and benefits are equal", "Benefits slightly outweigh risks", "Benefits far outweigh risks"],
    # Future/Current Impact Scale (Worse -> Better)
    tuple(sorted(("Profoundly Worse", "Noticeably Worse", "No Major Change", "Noticeably Better", "Profoundly Better"))):
        ["Profoundly Worse", "Noticeably Worse", "No Major Change", "Noticeably Better", "Profoundly Better"],
    # AI Perception Scale (Concerned -> Excited)
    tuple(sorted(("More excited than concerned", "Equally concerned and excited", "More concerned than excited"))):
        ["More concerned than excited", "Equally concerned and excited", "More excited than concerned"],
    # Agree/Disagree/Unsure Scale (Disagree -> Agree)
    tuple(sorted(("Agree", "Disagree", "Unsure"))):
        ["Disagree", "Unsure", "Agree"],
    # Yes/No/Don't Know Scale (No -> Yes)
    tuple(sorted(("Yes", "No", "Don't Know"))):
         ["No", "Don't Know", "Yes"],
    # Automation Impact Scale (Least -> Most Affected)
    tuple(sorted(("Not at all", "I know someone who has lost their job", "I know several people who have lost their job", "I know many people who have lost their job"))):
        ["Not at all", "I know someone who has lost their job", "I know several people who have lost their job", "I know many people who have lost their job"]
}

def get_ordered_columns(columns):
    """Finds the correct predefined semantic order for a given set of column/response names."""
    # Sort the input column names to ensure consistent dictionary key lookup
    sorted_columns_tuple = tuple(sorted(columns))
    return RESPONSE_ORDERS.get(sorted_columns_tuple)

# --- End Semantic Ordering ---

def longest_common_suffix(strings):
    """Calculates the longest common suffix of a list of strings."""
    if not strings:
        return ""
    reversed_strings = [s[::-1] for s in strings]
    # Find common prefix of reversed strings
    lcp_reversed = os.path.commonprefix(reversed_strings)
    # Reverse back to get common suffix
    return lcp_reversed[::-1]

def generate_indicator_heatmaps(standardized_df, indicator_codesheet_path, output_dir):
    """
    Generates heatmaps for Indicator poll questions using standardized data,
    grouped by category defined in the codesheet.

    Args:
        standardized_df (pd.DataFrame): DataFrame from _aggregate_standardized.csv.
        indicator_codesheet_path (str): Path to the indicator codesheet CSV file.
        output_dir (str): Directory to save the indicator heatmap PNG files.
    """
    print("\n--- Generating Indicator Heatmaps (using standardized data) --- ")
    os.makedirs(output_dir, exist_ok=True)

    # --- Load Indicator Codesheet ---
    try:
        indicator_df = pd.read_csv(indicator_codesheet_path)
        # Filter for poll questions and map question text to category/code
        indicator_polls = indicator_df[indicator_df['question_type'] == 'Poll Single Select'].copy()
        indicator_polls.dropna(subset=['question_text'], inplace=True)
        qtext_to_category = indicator_polls.set_index('question_text')['question_category'].to_dict()
        qtext_to_code = indicator_polls.set_index('question_text')['question_code'].to_dict()
        print(f"  Loaded {len(indicator_polls)} indicator poll questions from: {indicator_codesheet_path}")
    except FileNotFoundError: print(f"  Error: Indicator codesheet not found at {indicator_codesheet_path}"); return
    except Exception as e: print(f"  Error loading indicator codesheet: {e}"); return

    # --- Filter Standardized Data for Indicator Polls ---
    indicator_q_texts = list(qtext_to_category.keys())
    indicator_data = standardized_df[
        (standardized_df['Question Type'] == 'Poll Single Select') &
        (standardized_df['Question'].isin(indicator_q_texts))
    ].copy()

    if indicator_data.empty:
        print("  Warning: No matching indicator poll question data found in the standardized file."); return

    # Add category information based on the codesheet mapping
    indicator_data['Category'] = indicator_data['Question'].map(qtext_to_category)

    # --- Generate Heatmap per Category ---
    for category, group in indicator_data.groupby('Category'):
        print(f"  Generating heatmap for category: {category} ({group['Question'].nunique()} questions)")
        category_data_for_pivot = []
        ordered_labels_info = []
        full_texts_in_category = group['Question'].unique().tolist()

        # --- Derive Labels & Title (same logic as before) ---
        text_to_label = {}; 
        title_line1 = category
        title_line2 = ""
        y_label_type = 'code' # Default
        y_label_wrap_width = 35

        if len(full_texts_in_category) > 1:
            lcp = os.path.commonprefix(full_texts_in_category)
            lcsuf = longest_common_suffix(full_texts_in_category)
            min_len = min(len(s) for s in full_texts_in_category)
            if len(lcp) + len(lcsuf) < min_len and (len(lcp) > 0 or len(lcsuf) > 0): 
                y_label_type = 'varying_part'
                for text in full_texts_in_category:
                    label = text[len(lcp):len(text)-len(lcsuf)].strip()
                    text_to_label[text] = label if len(label) > 1 else qtext_to_code.get(text, text[:30])
                if len(lcp) > 5 and len(lcsuf) > 5: title_line2 = f"{lcp} ___ {lcsuf}"
                elif len(lcp) > 5: title_line2 = f"{lcp}..."
                elif len(lcsuf) > 5: title_line2 = f"... {lcsuf}"
                else: title_line2 = "(Multiple Questions)"
            else:
                for text in full_texts_in_category: text_to_label[text] = qtext_to_code.get(text, text[:30])
                title_line2 = "(Multiple Questions)"
        elif len(full_texts_in_category) == 1:
             text = full_texts_in_category[0]
             text_to_label[text] = qtext_to_code.get(text, text[:30])
             title_line2 = text
            
        # --- Prepare Data & Pivot ---
        # Use standardized columns: 'Question', 'Response', 'All'
        if 'All' not in group.columns:
             print(f"    Warning: Skipping category '{category}' - 'All' segment column not found."); continue
        if 'Response' not in group.columns:
             print(f"    Warning: Skipping category '{category}' - 'Response' column not found."); continue
            
        # Apply parse_percentage to the 'All' column
        try:
             group['All_Parsed'] = group['All'].apply(parse_percentage)
             group['All_Parsed'] = pd.to_numeric(group['All_Parsed'], errors='coerce')
        except Exception as e:
             print(f"    Warning: Error parsing 'All' column for category '{category}': {e}"); continue
            
        # Store labels in order
        max_lines_in_ylabel = 1
        for q_text in full_texts_in_category:
             current_label = text_to_label.get(q_text)
             wrapped_label = textwrap.fill(current_label, width=y_label_wrap_width)
             if q_text not in [t for t, l in ordered_labels_info]: # Avoid duplicates if q_text appears multiple times
                 ordered_labels_info.append((q_text, wrapped_label))
                 max_lines_in_ylabel = max(max_lines_in_ylabel, wrapped_label.count('\n') + 1)
        ordered_texts = [text for text, label in ordered_labels_info]

        try:
            # Pivot using standardized columns and parsed percentage
            heatmap_pivot = group.pivot_table(index='Question', columns='Response', values='All_Parsed', aggfunc='first')
            # Reindex rows based on ordered labels and sort columns
            heatmap_pivot = heatmap_pivot.reindex(ordered_texts)
            heatmap_pivot = heatmap_pivot.sort_index(axis=1)
        except Exception as e:
            print(f"    Error pivoting data for category '{category}': {e}"); continue
        if heatmap_pivot.empty:
            print(f"    Warning: Pivoted data empty for '{category}'."); continue
            
        # --- >>> Apply Semantic Column Ordering <<< ---
        try:
            current_columns = heatmap_pivot.columns.tolist()
            ordered_columns = get_ordered_columns(current_columns)
            if ordered_columns:
                # Filter order to only columns present in the data
                final_order = [col for col in ordered_columns if col in current_columns]
                # Apply order only if it perfectly matches the columns present
                if set(final_order) == set(current_columns):
                     heatmap_pivot = heatmap_pivot[final_order]
                     logging.debug(f"    Applied custom column order for '{category}': {final_order}")
                else:
                     logging.warning(f"    Custom order definition mismatch for category '{category}'. Columns: {current_columns}. Defined order: {ordered_columns}. Using alphabetical sort.")
                     heatmap_pivot = heatmap_pivot.sort_index(axis=1) # Fallback to alphabetical
            else:
                 logging.debug(f"    No custom column order defined for category '{category}'. Using alphabetical sort. Columns: {current_columns}")
                 heatmap_pivot = heatmap_pivot.sort_index(axis=1) # Explicit alphabetical sort if no order defined
        except Exception as sort_e:
             logging.warning(f"    Error applying column sorting for category '{category}': {sort_e}. Using alphabetical sort.")
             heatmap_pivot = heatmap_pivot.sort_index(axis=1) # Fallback on any sorting error
        # --- <<< END Semantic Column Ordering >>> ---

        # --- Plotting (same logic as before) ---
        n_rows, n_cols = heatmap_pivot.shape
        fig_width = max(8, n_cols * 0.9 + max(0, max_lines_in_ylabel -1) * 1.5 )
        fig_height = max(5, n_rows * 0.7 + 2.5)
        plt.figure(figsize=(fig_width, fig_height))

        wrapped_title_line2 = textwrap.fill(title_line2, width=80)
        final_title = f"{title_line1}\n{wrapped_title_line2}".strip()

        ax = sns.heatmap(heatmap_pivot, annot=True, fmt=".0%", cmap="Blues", linewidths=.5, cbar=False, annot_kws={"size": 9})
        
        ordered_wrapped_labels = [label for text, label in ordered_labels_info]
        ax.set_yticklabels(ordered_wrapped_labels, rotation=0, fontsize=9, va='center') 
        
        plt.xticks(rotation=30, ha='right', fontsize=9)
        plt.title(final_title, fontsize=11, pad=25)
        plt.xlabel("Response Options", fontsize=10)
        plt.ylabel("Question Detail" if y_label_type == 'varying_part' else "Question Code", fontsize=10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
        # --- Save heatmap ---
        safe_category_name = re.sub(r'[^\w\-\. ]', '', category).strip().replace(' ', '_')
        heatmap_filename = f"indicator_heatmap_{safe_category_name}.png"
        heatmap_path = os.path.join(output_dir, heatmap_filename)
        try:
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            print(f"    Saved heatmap to: {heatmap_path}")
        except Exception as e:
            print(f"    Error saving heatmap for category '{category}': {e}")
        plt.close()

    print("--- Indicator Heatmap Generation Complete ---")

def main():
    parser = argparse.ArgumentParser(description='Generate indicator analysis heatmaps from standardized data.')
    
    # Input specification
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--gd_number", type=int, help="Global Dialogue cadence number (e.g., 1, 2, 3). Constructs default paths.")
    input_group.add_argument("--standardized_csv", help="Explicit path to the standardized aggregate CSV file.")

    # Required codesheet path
    parser.add_argument('--indicator_codesheet', default="Data/Documentation/INDICATOR_CODESHEET.csv", 
                       help='Path to the Indicator Codesheet CSV file.')

    # Output directory
    parser.add_argument('-o', '--output_dir', help='Directory to save indicator heatmap output files (required if --standardized_csv is used).')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging.')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # --- Determine File Paths ---
    std_csv_path = None
    output_path = None
    codesheet_path = args.indicator_codesheet # Use provided or default

    if args.gd_number:
        gd_num = args.gd_number
        gd_identifier = f"GD{gd_num}"
        data_dir = os.path.join("Data", gd_identifier)
        output_base_dir = os.path.join("analysis_output", gd_identifier)

        std_csv_path = os.path.join(data_dir, f"{gd_identifier}_aggregate_standardized.csv")
        output_path = args.output_dir if args.output_dir else os.path.join(output_base_dir, "indicators") # Default output subfolder

        logging.info(f"Using GD number {gd_num} to determine paths:")
        if not os.path.exists(std_csv_path):
            parser.error(f"Standardized input file not found for GD{gd_num}. Expected at: {std_csv_path}")

    else: # Explicit paths
        if not args.standardized_csv or not args.output_dir:
             parser.error("--standardized_csv and --output_dir are required when --gd_number is not used.")
        std_csv_path = args.standardized_csv
        output_path = args.output_dir
        logging.info(f"Using explicitly provided paths:")

    logging.info(f"  Standardized Data: {std_csv_path}")
    logging.info(f"  Indicator Codesheet: {codesheet_path}")
    logging.info(f"  Output Directory: {output_path}")
    
    if not os.path.exists(codesheet_path):
         logging.warning(f"Indicator codesheet not found at: {codesheet_path}")
         # Continue for now, but heatmap generation will fail later

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

    # --- Generate Heatmaps ---
    generate_indicator_heatmaps(standardized_data, codesheet_path, output_path)
    
    logging.info("Indicator analysis script finished.")

if __name__ == "__main__":
    main() 