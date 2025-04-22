# global-dialogues
Global Dialogues data and analysis tools

This repository contains data and analysis tools from the "Global Dialogues" survey project, conducted in collaboration with Remesh.ai, Prolific, and The Collective Intelligence Project.

## About the Project
Global Dialogues are recurring surveys where participants answer demographic questions, attitude polls, and open-ended opinion questions. Participants also vote (Agree/Disagree) on a sample of other participants' responses and complete pairwise comparisons of responses.

## Data Structure
The first dialogue took place in September 2024. Data for this cadence is stored in `Data/GD1` with the following files:

- `aggregate.csv`: Complete survey data by Question, including agreement rates (Agree/Disagree) on open-ended questions by participant poll group
- `verbatim_map.csv`: Complete set of every participant's response to every open-ended question
- `binary.csv`: Complete list of every participant's vote on any response
- `preference.csv`: Complete list of pairwise preferences of responses for each question
- `participants.csv`: List of participants with completion rates and additional metadata
- `global_inputs.json`: Complete data from aggregate.csv in JSON format, including pre-computed text embeddings for text similarity analysis
 
 


