# global-dialogues
Global Dialogues data and analysis tools

This repository contains data and analysis tools from the "Global Dialogues" survey project, conducted in collaboration with Remesh.ai, Prolific, and The Collective Intelligence Project.

## About the Project
Global Dialogues are recurring surveys where participants answer demographic questions, attitude polls, and open-ended opinion questions. Participants also vote (Agree/Disagree) on a sample of other participants' responses and complete pairwise comparisons of responses.

## Installation

Install required packages:
```bash
pip install -r requirements.txt
```

## Workflow Overview

1.  **Download Raw Data:** Obtain the raw CSV export files from Remesh.ai for a given Global Dialogue cadence (e.g., GD3) and place them in the corresponding `Data/GD<N>/` directory.
2.  **Cleanup Metadata:** Run the `preprocess_cleanup_metadata.py` script to remove initial metadata rows from the raw CSV files, ensuring the header is the first line. This modifies the files in place.
    ```bash
    python tools/scripts/preprocess_cleanup_metadata.py <N>
    ```
3.  **Preprocess Aggregate Data:** Run `preprocess_aggregate.py` to generate the essential `_aggregate_standardized.csv` and `_segment_counts_by_question.csv` files needed by ALL analysis scripts. This is a critical step as the standardized format is the primary reference for analysis.
    ```bash
    python tools/scripts/preprocess_aggregate.py --gd_number <N>
    ```
4.  **Preprocess Tag Data (if using tags):** Run `preprocess_tag_files.py` if you need to analyze Remesh tag data (requires separate raw tag exports).
    ```bash
    python tools/scripts/preprocess_tag_files.py --raw_dir Data/GD<N>/tag_codes_raw/ --output_dir Data/GD<N>/tags/
    ```
5.  **Run Analyses:** Execute the desired analysis scripts (e.g., `calculate_tags.py`, `calculate_consensus.py`, etc.) or the master `analyze_dialogues.py` script.
    ```bash
    python tools/scripts/calculate_tags.py <N>
    # OR
    python tools/scripts/analyze_dialogues.py <N>
    ```

## Data Structure

The project involves recurring survey cadences:

| Dialogue | Time          | Status   |
|----------|---------------|----------|
| GD1      | September 2024 | Complete |
| GD2      | January 2025    | Complete  |
| GD3      | March 2025    | Complete  |
| GD4      | May 2025    | Planned  |

Data for each completed cadence is stored in its respective folder within `/Data` (e.g., `/Data/GD1`). Detailed descriptions of the data files and columns can be found in `Data/Documentation/DATA_GUIDE.md`.

### Large Embedding Data

Some advanced analyses (such as thematic ranking) require pre-computed text embeddings, which result in large data files (~800MB each) unsuitable for direct storage in this Git repository. These files are hosted separately and can be downloaded as needed.

#### Automated Download

The repository includes a script to easily download embedding files:

```bash
# Show available embedding files and their status
python tools/scripts/download_embeddings.py --list

# Download embeddings for a specific Global Dialogue
python tools/scripts/download_embeddings.py 3  # For GD3

# Download all available embedding files
python tools/scripts/download_embeddings.py --all

# Force re-download even if file already exists
python tools/scripts/download_embeddings.py 3 --force
```

You can also use the Makefile commands:

```bash
make download-embeddings        # Show available files
make download-embeddings-gd3    # Download for GD3
make download-all-embeddings    # Download all
```

The files will be automatically placed in the correct locations:

```
global-dialogues/
└── Data/
    ├── GD1/
    │   └── GD1_embeddings.json
    ├── GD2/
    │   └── GD2_embeddings.json
    └── GD3/
        └── GD3_embeddings.json
```

*Note: These embedding files are included in `.gitignore` to prevent accidental commits.*

### Data Files per Cadence

Each cadence folder contains the following data files:

#### Primary Analysis Files (Processed)

*   **`GD<N>_aggregate_standardized.csv`**: The primary file for all analyses - a cleaned and standardized version of the raw aggregate data with consistent columns and formatting. This file is created by running the preprocessing script and should be used for all analysis work.
*   **`GD<N>_segment_counts_by_question.csv`**: Contains participant counts for each segment per question, needed for certain analyses.

#### Raw Data Files (Original Exports from Remesh.ai)

*   **`GD<N>_aggregate.csv`**: Raw survey data aggregated by Question, including segment agreement rates for *Ask Opinion* questions. Has inconsistent formatting and metadata rows.
*   **`GD<N>_binary.csv`**: Individual participant votes (`agree`/`disagree`/`neutral`) on *Ask Opinion* responses.
*   **`GD<N>_participants.csv`**: Survey data organized by Participant, showing individual responses to each question. Includes overall agreement rates for *Ask Opinion* responses submitted by the participant.
*   **`GD<N>_preference.csv`**: Pairwise preference judgments between *Ask Opinion* responses.
*   **`GD<N>_verbatim_map.csv`**: Mapping of *Ask Opinion* response text (`Thought Text`) to the participant who authored it and the question it belongs to.
*   **`GD<N>_summary.csv`**: LLM-generated summaries for the overall dialogue and individual questions.

Detailed descriptions of file formats and columns can be found in `Data/Documentation/DATA_GUIDE.md`.

## Analysis Tools

Analysis scripts and notebooks can be found in the `/tools` directory.

### Participant Reliability Index (PRI)

The PRI is a composite score that assesses participant response quality and reliability. It combines multiple signals including duration, response quality tags, universal disagreement rates, and consensus voting patterns. For detailed documentation, see [PRI_GUIDE.md](Data/Documentation/PRI_GUIDE.md).

 


