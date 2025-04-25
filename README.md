# global-dialogues
Global Dialogues data and analysis tools

This repository contains data and analysis tools from the "Global Dialogues" survey project, conducted in collaboration with Remesh.ai, Prolific, and The Collective Intelligence Project.

## About the Project
Global Dialogues are recurring surveys where participants answer demographic questions, attitude polls, and open-ended opinion questions. Participants also vote (Agree/Disagree) on a sample of other participants' responses and complete pairwise comparisons of responses.

## Data Structure

The project involves recurring survey cadences:

| Dialogue | Time          | Status   |
|----------|---------------|----------|
| GD1      | September 2024 | Complete |
| GD2      | January 2025    | Complete  |
| GD3      | March 2025    | Complete  |
| GD4      | May 2025    | Planned  |

Data for each completed cadence is stored in its respective folder within `/Data` (e.g., `/Data/GD1`). Detailed descriptions of the data files and columns can be found in `Data/Documentation/DATA_GUIDE.md`.

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

 


