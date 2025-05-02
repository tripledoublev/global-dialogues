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
3.  **Preprocess Aggregate Data:** Run `preprocess_aggregate.py` to generate the essential `_aggregate_standardized.csv` and `_segment_counts_by_question.csv` files needed by many analysis scripts.
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

Some analyses require pre-computed text embeddings, which result in large data files unsuitable for direct storage in this Git repository. These files are hosted separately.

*   **GD3 Embeddings:** The combined aggregate data and text embeddings for GD3 (`GD3_embeddings.json`, ~800MB) can be downloaded from:
    *   [**LINK_TO_YOUR_DATA_HOSTING_HERE**] (e.g., Zenodo, Figshare, Google Drive)

    Please download this file and place it in the `Data/GD3/` directory:
    ```
    global-dialogues/
    └── Data/
        └── GD3/
            └── GD3_embeddings.json  # <-- Place downloaded file here
            └── aggregate.csv
            └── ... (other GD3 files)
    ```
    *Note: This file path is included in `.gitignore` to prevent accidental commits.*

### Data Files per Cadence

Each cadence folder contains the following core data files (raw outputs from Remesh.ai):

*   **`aggregate.csv`**: Survey data aggregated by Question, including segment agreement rates for *Ask Opinion* questions.
*   **`binary.csv`**: Individual participant votes (`agree`/`disagree`/`neutral`) on *Ask Opinion* responses.
*   **`participants.csv`**: Survey data organized by Participant, showing individual responses to each question. Includes overall agreement rates for *Ask Opinion* responses submitted by the participant.
*   **`preference.csv`**: Pairwise preference judgments between *Ask Opinion* responses.
*   **`verbatim_map.csv`**: Mapping of *Ask Opinion* response text (`Thought Text`) to the participant who authored it and the question it belongs to.
*   **`summary.csv`**: LLM-generated summaries for the overall dialogue and individual questions.

## Analysis Tools

Analysis scripts and notebooks can be found in the `/tools` directory.

 


