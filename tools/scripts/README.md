# Analysis Scripts

This directory contains scripts for processing raw Remesh.ai data for Global Dialogues analysis and generating standard reports.

**IMPORTANT:** Before running most analysis scripts, you should run the preprocessing scripts first.

## Workflow Overview

1.  **Download Raw Data:** Place raw Remesh CSVs into the correct `Data/GD<N>/` directory.
2.  **Cleanup Metadata:** Run `preprocess_cleanup_metadata.py` to remove metadata headers from most raw CSVs (modifies files in place).
    ```bash
    python tools/scripts/preprocess_cleanup_metadata.py <N>
    ```
3.  **Preprocess Aggregate:** Run `preprocess_aggregate.py` to standardize the complex `aggregate.csv` file.
    ```bash
    python tools/scripts/preprocess_aggregate.py --gd_number <N>
    ```
4.  **Preprocess Tags (Optional):** Run `preprocess_tag_files.py` if analyzing tags (requires separate raw tag exports).
    ```bash
    python tools/scripts/preprocess_tag_files.py --raw_dir Data/GD<N>/tag_codes_raw/ --output_dir Data/GD<N>/tags/
    ```
5.  **Run Analyses:** Now you can run individual `calculate_*.py` scripts or the master `analyze_dialogues.py` script, as they expect the preprocessed data.

## Script Details

### `preprocess_tag_files.py`

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

### `preprocess_aggregate.py`

**Purpose:** Processes the raw `_aggregate.csv` file (which contains metadata rows and repeated/varying header rows for different question types) into standardized formats suitable for analysis.

**Workflow:**

1.  **Input:** Takes either a Global Dialogue number (`--gd_number`) to find the standard input file (`Data/GD<N>/GD<N>_aggregate.csv`) or explicit `--input_file`, `--output_file`, and (optional) `--segment_counts_output` paths.
2.  **Run Script:**
    ```bash
    # Simplest example using GD number:
    python tools/scripts/preprocess_aggregate.py --gd_number 3
    ```
3.  **Output:** By default (when using `--gd_number`), generates two files in the corresponding `Data/GD<N>/` directory:
    *   `GD<N>_aggregate_standardized.csv`: A CSV with a single header row, consistent columns (including merged `Response` and `OriginalResponse` columns), and data mapped correctly from all question blocks. Metadata and repeated headers are removed.
    *   `GD<N>_segment_counts_by_question.csv`: A CSV detailing the participant count (`N`) for each segment *for each specific question*.

*Note: This script is crucial for preparing the aggregate data before running subsequent analysis scripts.*

### `calculate_consensus.py`

**Purpose:** Calculates consensus profiles (percentile minimums) and highest minimum agreement across major segments for *Ask Opinion* questions.

**Input:** Uses `_aggregate_standardized.csv` and `_segment_counts_by_question.csv`.

**Run Script:**
```bash
# Simplest example using GD number:
python tools/scripts/calculate_consensus.py --gd_number 3
```

**Output:** Saves CSV reports (e.g., `consensus_profiles.csv`, `major_segment_min_agreement_top10.csv`) to the `analysis_output/GD<N>/consensus/` directory.

### `calculate_divergence.py`

**Purpose:** Calculates divergence scores for *Ask Opinion* questions, identifying responses with high disagreement between segments.

**Input:** Uses `_aggregate_standardized.csv` and `_segment_counts_by_question.csv`.

**Run Script:**
```bash
# Simplest example using GD number:
python tools/scripts/calculate_divergence.py --gd_number 3
```

**Output:** Saves CSV reports (e.g., `divergence_by_question.csv`, `divergence_overall.csv`) to the `analysis_output/GD<N>/divergence/` directory.

### `calculate_indicators.py`

**Purpose:** Generates heatmaps visualizing responses to predefined *Indicator Poll* questions, grouped by category.

**Input:** Uses `_aggregate_standardized.csv` and the `INDICATOR_CODESHEET.csv`.

**Run Script:**
```bash
# Simplest example using GD number:
python tools/scripts/calculate_indicators.py --gd_number 3
```

**Output:** Saves PNG heatmap images (e.g., `indicator_heatmap_<category>.png`) to the `analysis_output/GD<N>/indicators/` directory.

### `analyze_dialogues.py` (Master Script)

**Purpose:** Acts as a master controller to run the entire standard analysis pipeline for a given Global Dialogue cadence number.

**Workflow:** Executes the following scripts in order, using the provided `--gd_number`:
1.  `preprocess_aggregate.py`
2.  `calculate_consensus.py`
3.  `calculate_divergence.py`
4.  `calculate_indicators.py`

**Run Script:**
```bash
# Run full pipeline for GD3:
python tools/scripts/analyze_dialogues.py 3
```

**Output:** Creates/populates the default output directories (e.g., `Data/GD3/` for standardized files, `analysis_output/GD3/{consensus,divergence,indicators}/` for reports and plots) as generated by the individual scripts.

## Advanced Analysis Scripts

### `tools/analysis/thematic_ranking.py`

**Purpose:** Performs thematic ranking analysis using semantic embeddings to identify responses most relevant to predefined themes. This is an advanced analysis tool that requires OpenAI API access and pre-computed embeddings.

**Prerequisites:**
1. **Embeddings File:** Requires a `GD<N>_embeddings.json` file in the corresponding `Data/GD<N>/` directory. This file contains response text data with pre-computed vector embeddings.
2. **OpenAI API Key:** Create a `.env` file in the project root with `OPENAI_API_KEY=your_key_here` to enable theme embedding generation.
3. **Dependencies:** Requires additional packages: `openai`, `scikit-learn`, `python-dotenv`

**Workflow:**
1. **Setup Environment:** Ensure you have an OpenAI API key configured and the embeddings file downloaded.
2. **Run Analysis:**
   ```bash
   # Analyze thematic rankings for GD1:
   python tools/analysis/thematic_ranking.py --gd 1
   
   # Or for other Global Dialogues:
   python tools/analysis/thematic_ranking.py --gd 2
   python tools/analysis/thematic_ranking.py --gd 3
   ```

**How It Works:**
- Uses cosine similarity to compare response embeddings against predefined thematic queries
- Themes are defined in `tools/analysis/thematic_queries.txt` (one per line) and can be customized
- Ranks the top 100 most relevant responses for each theme using semantic similarity

**Output:** Saves a comprehensive CSV file (`thematic_rankings.csv`) to `analysis_output/GD<N>/thematic_rankings/` containing:
- Theme name and cosine similarity scores
- Response text, question details, and participant information  
- Run metadata (timestamp, unique run ID) for tracking different analysis runs

**Note:** This script is located in `tools/analysis/` rather than `tools/scripts/` because it requires additional dependencies and API access beyond the standard analysis pipeline.