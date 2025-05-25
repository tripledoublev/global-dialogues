# global-dialogues
Global Dialogues data and analysis tools

This repository contains data and analysis tools from the "Global Dialogues" survey project, conducted in collaboration with Remesh.ai, Prolific, and The Collective Intelligence Project.

## About the Project
Global Dialogues are recurring surveys where participants answer demographic questions, attitude polls, and open-ended opinion questions. Participants also vote (Agree/Disagree) on a sample of other participants' responses and complete pairwise comparisons of responses.

## Quick Start

Install required packages:
```bash
pip install -r requirements.txt
```

This repository includes a comprehensive Makefile with commands for all common tasks. To see all available commands:
```bash
make help
```

For new users, the typical workflow is:
1. **Preprocess data**: `make preprocess-gd3` (or other GD number)
2. **Run analysis**: `make analyze-gd3`
3. **Download embeddings** (for advanced analysis): `make download-embeddings-gd3`

## Detailed Workflow

For users who prefer direct Python commands or need to understand the underlying processes:

1.  **Download Raw Data:** Obtain the raw CSV export files from Remesh.ai for a given Global Dialogue cadence (e.g., GD3) and place them in the corresponding `Data/GD<N>/` directory.

2.  **Preprocess Data:** Use the Makefile command (recommended) or run scripts directly:
    ```bash
    # Recommended: Use Makefile
    make preprocess-gd3
    
    # Alternative: Run scripts directly
    python tools/scripts/preprocess_cleanup_metadata.py 3
    python tools/scripts/preprocess_aggregate.py --gd_number 3
    ```

3.  **Run Analysis:** Use the Makefile command (recommended) or run scripts directly:
    ```bash
    # Recommended: Use Makefile for full analysis pipeline
    make analyze-gd3
    
    # Alternative: Run individual analyses
    python tools/scripts/calculate_consensus.py --gd_number 3
    python tools/scripts/calculate_divergence.py --gd_number 3
    # etc.
    ```

4.  **Advanced Analysis (Optional):** Download embeddings and run thematic analysis:
    ```bash
    make download-embeddings-gd3
    make run-thematic-ranking-gd3
    ```

**Note:** All Make commands are documented in the Makefile. Run `make help` to see the full list of available commands for preprocessing, analysis, and utilities.

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

You can also use the Makefile commands (recommended):

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

 


