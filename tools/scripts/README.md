# Analysis Scripts

This directory contains scripts for processing raw Remesh.ai data for Global Dialogues analysis.

## `preprocess_tag_files.py`

**Purpose:** Cleans and organizes raw tag export files (`*_Tag_Categories.csv`, `*_Thought_Labels.csv`) downloaded from Remesh for each Ask Opinion question.

**Workflow:**

1.  **Download Raw Data:** Place the raw Remesh export CSVs into a temporary input directory (e.g., `Data/GD3/tag_codes_raw/`). Get these from Remesh via *Conversation > Analysis > Auto Code*. Select each Question from the Auto Code dropdown and download via *Export Codes*.
2.  **Run Script:** Execute the script, specifying the raw input directory and the desired output directory for cleaned, version-controlled files.
    ```bash
    # Example for GD3:
    python tools/scripts/preprocess_tag_files.py --raw_dir Data/GD3/tag_codes_raw/ --output_dir Data/GD3/tags/
    ```
3.  **Output:** The script generates:
    *   Cleaned individual tag files named `<QuestionID>_*.csv` in the output directory.
    *   Combined files `all_tag_categories.csv` and `all_thought_labels.csv` in the output directory.
4.  **.gitignore:** Add your raw input directory (e.g., `**/tag_codes_raw/`) to your project's `.gitignore` file.
5.  **Commit:** Commit the contents of the processed output directory (e.g., `Data/GD3/tags/`) to Git.
6.  **Clean Up (Optional):** Delete the files from the raw input directory.

*Note: Rerunning the script with new/updated files in the raw directory will update the corresponding individual files and rebuild the combined files in the output directory.*

## `analyze_dialogues.py`

**Purpose:** Analyzes a processed `aggregate.csv` file for a specific Global Dialogue cadence to generate divergence reports, consensus profiles, and indicator question heatmaps.

**Workflow:**

1.  **Prerequisites:** Ensure the relevant `aggregate.csv` file exists (e.g., `Data/GD3/GD3_aggregate.csv`) and the indicator codesheet (`Data/Documentation/INDICATOR_CODESHEET.csv`) is present.
2.  **Run Script:** Execute the script, specifying the Global Dialogue cadence number OR the direct path to the `aggregate.csv` file. Other options control output location, filtering, etc.
    ```bash
    # Option 1: Specify by GD number (looks for Data/GD<N>/GD<N>_aggregate.csv)
    python tools/scripts/analyze_dialogues.py --gd_number 3 

    # Option 2: Specify explicit file path
    python tools/scripts/analyze_dialogues.py --csv_filepath /path/to/your/aggregate.csv

    # Example with custom output and filtering:
    python tools/scripts/analyze_dialogues.py --gd_number 3 -o custom_output --min_segment_size 50
    ```
3.  **Output:** The script saves results into a structured directory based on the GD cadence or input file (e.g., `analysis_output/GD3/`). Outputs include:
    *   `processed_data.pkl`: Cached preprocessed data (consider adding `*processed_data.pkl` or `**/analysis_output/**/processed_data.pkl` to `.gitignore`).
    *   `divergence/`: CSV reports on divergent responses.
    *   `consensus/`: CSV reports on consensus profiles.
    *   `indicators/`: PNG heatmaps for indicator question categories.

*Note: The script uses caching (`processed_data.pkl`). Use the `--force_reparse` flag if you need to re-process the input CSV file instead of using the cache.* 