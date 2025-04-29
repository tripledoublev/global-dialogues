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

def longest_common_suffix(strings):
    """Calculates the longest common suffix of a list of strings."""
    if not strings:
        return ""
    reversed_strings = [s[::-1] for s in strings]
    # Find common prefix of reversed strings
    lcp_reversed = os.path.commonprefix(reversed_strings)
    # Reverse back to get common suffix
    return lcp_reversed[::-1]

def generate_indicator_heatmaps(indicator_codesheet_path, questions_data, indicators_output_dir):
    """
    Generates heatmaps for Indicator poll questions, grouped by category.
    (With improved formatting for labels and layout, and structured titles)
    """
    print("\n--- Generating Indicator Heatmaps --- ")
    # Ensure output dir exists
    os.makedirs(indicators_output_dir, exist_ok=True)

    # --- Load Indicator Codesheet ---
    try:
        indicator_df = pd.read_csv(indicator_codesheet_path)
        indicator_polls = indicator_df[indicator_df['question_type'] == 'Poll Single Select'].copy()
        indicator_polls.dropna(subset=['question_text'], inplace=True)
        qtext_to_category = indicator_polls.set_index('question_text')['question_category'].to_dict()
        qtext_to_code = indicator_polls.set_index('question_text')['question_code'].to_dict()
        print(f"  Loaded {len(indicator_polls)} indicator poll questions from: {indicator_codesheet_path}")
    except FileNotFoundError: print(f"  Error: Indicator codesheet not found at {indicator_codesheet_path}"); return
    except Exception as e: print(f"  Error loading indicator codesheet: {e}"); return

    # --- Map Aggregate Data to Indicators ---
    indicator_question_data = {}
    for metadata, df in questions_data:
        q_text = metadata.get('text'); q_type = metadata.get('type')
        if q_type == 'Poll Single Select' and q_text in qtext_to_category:
            category = qtext_to_category[q_text]
            if category not in indicator_question_data: indicator_question_data[category] = []
            indicator_question_data[category].append((metadata, df, q_text))
    if not indicator_question_data: print("  Warning: No matching indicator poll questions found."); return

    # --- Generate Heatmap per Category ---
    for category, questions_in_category in indicator_question_data.items():
        print(f"  Generating heatmap for category: {category} ({len(questions_in_category)} questions)")
        category_data_for_pivot = []; ordered_labels_info = []
        full_texts_in_category = [q_text for meta, df, q_text in questions_in_category]

        # --- Calculate LCP/LCSuf and derive/wrap labels & title ---
        text_to_label = {}; 
        title_line1 = category # Always start with category name
        title_line2 = "" # Second line for common text or full question
        y_label_type = 'code' # Default to using question codes for y-labels
        y_label_wrap_width = 35 # Characters

        if len(full_texts_in_category) > 1:
            lcp = os.path.commonprefix(full_texts_in_category); lcsuf = longest_common_suffix(full_texts_in_category)
            min_len = min(len(s) for s in full_texts_in_category)
            # Check if LCP/LCSuf are meaningful
            if len(lcp) + len(lcsuf) < min_len and (len(lcp) > 0 or len(lcsuf) > 0): 
                y_label_type = 'varying_part' # Use derived labels
                for text in full_texts_in_category:
                    label = text[len(lcp):len(text)-len(lcsuf)].strip()
                    text_to_label[text] = label if len(label) > 1 else qtext_to_code.get(text, text[:30])
                
                # Construct second title line with common parts
                if len(lcp) > 5 and len(lcsuf) > 5: title_line2 = f"{lcp} ___ {lcsuf}"
                elif len(lcp) > 5: title_line2 = f"{lcp}..."
                elif len(lcsuf) > 5: title_line2 = f"... {lcsuf}"
                else: title_line2 = "(Multiple Questions)" # Fallback if LCP/LCSuf are short
            else:
                # LCP/LCSuf overlap or too short, use codes as labels
                for text in full_texts_in_category: text_to_label[text] = qtext_to_code.get(text, text[:30])
                title_line2 = "(Multiple Questions)" # Indicate multiple questions without clear pattern
        elif len(full_texts_in_category) == 1:
             # Single question: use code for label, full text for title line 2
             text = full_texts_in_category[0]
             text_to_label[text] = qtext_to_code.get(text, text[:30])
             title_line2 = text # Use full text as second title line

        # --- Prepare data for pivoting ---
        max_lines_in_ylabel = 1
        for metadata, df, q_text in questions_in_category:
            q_id = metadata.get('id'); all_n_col = next((col for col in df.columns if col.startswith("All(") and col.endswith(")")), None)
            if 'Responses' not in df.columns or not all_n_col: print(f"    Warning: Skipping QID {q_id} - Missing columns."); continue
            df[all_n_col] = pd.to_numeric(df[all_n_col], errors='coerce')
            temp_df = df[['Responses', all_n_col]].rename(columns={all_n_col: 'Percentage'}); temp_df['QuestionText'] = q_text
            category_data_for_pivot.append(temp_df)
            current_label = text_to_label.get(q_text)
            if q_text not in [t for t, l in ordered_labels_info]:
                wrapped_label = textwrap.fill(current_label, width=y_label_wrap_width)
                ordered_labels_info.append((q_text, wrapped_label))
                max_lines_in_ylabel = max(max_lines_in_ylabel, wrapped_label.count('\n') + 1)
        if not category_data_for_pivot: print(f"    Warning: No valid data for category '{category}'."); continue
        combined_df = pd.concat(category_data_for_pivot, ignore_index=True)
        
        # --- Pivot and Reindex ---
        try:
            heatmap_pivot = combined_df.pivot_table(index='QuestionText', columns='Responses', values='Percentage', aggfunc='first')
            ordered_texts = [text for text, label in ordered_labels_info]
            heatmap_pivot = heatmap_pivot.reindex(ordered_texts)
            heatmap_pivot = heatmap_pivot.sort_index(axis=1)
        except Exception as e: print(f"    Error pivoting data for category '{category}': {e}"); continue
        if heatmap_pivot.empty: print(f"    Warning: Pivoted data empty for '{category}'."); continue
             
        # --- Plotting (with structured title) ---
        n_rows, n_cols = heatmap_pivot.shape
        fig_width = max(8, n_cols * 0.9 + max(0, max_lines_in_ylabel -1) * 1.5 )
        fig_height = max(5, n_rows * 0.7 + 2.5) # Increase base height slightly more for title
        plt.figure(figsize=(fig_width, fig_height))

        # Wrap the second title line if it's long (e.g., full question text)
        wrapped_title_line2 = textwrap.fill(title_line2, width=80) # Adjust width as needed
        final_title = f"{title_line1}\n{wrapped_title_line2}".strip() # Combine lines

        ax = sns.heatmap(heatmap_pivot, annot=True, fmt=".0%", cmap="Blues", linewidths=.5, cbar=False, annot_kws={"size": 9})
        
        ordered_wrapped_labels = [label for text, label in ordered_labels_info]
        ax.set_yticklabels(ordered_wrapped_labels, rotation=0, fontsize=9, va='center') 
        
        plt.xticks(rotation=30, ha='right', fontsize=9)
        # Use final combined title, adjust padding
        plt.title(final_title, fontsize=11, pad=25) # Slightly smaller title font, more padding
        plt.xlabel("Response Options", fontsize=10)
        plt.ylabel("Question Detail" if y_label_type == 'varying_part' else "Question Code", fontsize=10) # Adjust y-axis title based on label type
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust top boundary slightly for title
        
        # --- Save heatmap ---
        safe_category_name = re.sub(r'[^\w\-\. ]', '', category).strip().replace(' ', '_')
        heatmap_filename = f"indicator_heatmap_{safe_category_name}.png"
        heatmap_path = os.path.join(indicators_output_dir, heatmap_filename)
        try:
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            print(f"    Saved heatmap to: {heatmap_path}")
        except Exception as e:
            print(f"    Error saving heatmap for category '{category}': {e}")
        plt.close()

    print("--- Indicator Heatmap Generation Complete ---")

def main():
    parser = argparse.ArgumentParser(description='Generate indicator analysis heatmaps from standardized data.')
    parser.add_argument('standardized_csv', help='Path to the standardized aggregate CSV file.')
    parser.add_argument('output_dir', help='Directory to save indicator heatmap output files.')
    parser.add_argument('--indicator_codesheet', default="Data/Documentation/INDICATOR_CODESHEET.csv", 
                       help='Path to the Indicator Codesheet CSV file.')
    parser.add_argument('--min_segment_size', type=int, default=15,
                       help='Minimum participant size for a segment to be included in analysis.')

    args = parser.parse_args()

    logging.info(f"Starting indicator analysis using {args.standardized_csv}")
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
                'text': group['Question'].iloc[0]
            }
            questions_data.append((metadata, group))
        
        # Generate heatmaps
        generate_indicator_heatmaps(args.indicator_codesheet, questions_data, args.output_dir)
        logging.info("Indicator analysis complete.")
    else:
        logging.error("Failed to load data, aborting indicator analysis.")

if __name__ == "__main__":
    main() 