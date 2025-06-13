# Data Guide

## Overview

Global Dialogues is a recurring, cross-national survey program designed to collect public perspectives on AI. Each cadence involves an online, AI-moderated dialogue session (30-60 minutes) hosted on Remesh.ai. Participants, recruited through Prolific, answer standard poll questions and open-ended opinion questions, and vote on peer statements, generating quantitative evidence of agreement.

Some standard initial *Poll Single Select* questions are used to define participant *Segments*. These standard Segments primarily include:

-   Age groups (~10-year buckets)
-   Gender (Male/Female/Non-binary)
-   Religion
-   Country (also grouped into various regions and subregions)
-   Environment (Urban/Rural/Suburban)

Other *Poll Single Select* questions simply collect additional survey information. All Polls use pre-defined responses.

*Ask Opinion* questions pose an open-ended question. Participants respond with freeform text and are then prompted to vote 'agree' or 'disagree' on a random set of 5 other participants' responses. Segments are used to express the "agreement rate" – the estimated percentage of participants in that Segment who agree with a particular *Ask Opinion* response based on the vote data.

*Ask Experience* questions are also open-ended but do not prompt participants to vote on others' responses.

Participants are guided through an experience designed to feel like a moderated "dialogue", led by a named moderator. Context is provided to prepare them for each set of questions. This context might not always be clear from the question text itself and may need to be referenced from the separate Dialogue Guide, which includes the full moderation prompts.

Global Dialogues were conducted in multiple languages (Arabic, English, Russian, Chinese, Hindi, French, Portuguese, Spanish), with questions translated accordingly. When participants responded to *Ask Opinion* questions, their answers were machine-translated into the voter's selected language for the subsequent voting phase. For transparency, the *Original Responses* are preserved as written, alongside a machine-translated *English Response* for easier analysis.

## Global Dialogues Indicators

Starting with `GD3`, *Global Dialogues Indicators* are recurring Poll questions that will be asked in every cadence (with minimal modification). These intend to capture a "global pulse" over time to capture changes in attitudes and establish baselines for subsequent questions in the rest of the dialogue. Nothing in the data files distinguishes Indicator questions, so these are outlined and categorized for easy reference in `INDICATOR_CODESHEET.csv`.

## Data Files

**Note:** All data files follow the naming convention `GD<N>_filename.csv` where `<N>` is the Global Dialogue number (e.g., GD3, GD4).

### **`GD<N>_discussion_guide.csv`**

This file outlines the structure and content of the dialogue as experienced by participants, in chronological order.

| Column                                           | Description                                                                                                                                                                                                                                                                             |
| :----------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Item type (dropdown)`                           | Type of interaction: <br> - `onboarding single select`: Poll used for demographic segmentation. Appears as `Poll Single Select` elsewhere. <br> - `speak`: Moderator context; no participant input. <br> - `poll single select`: Standard poll question. <br> - `ask opinion`: Open-ended question with peer voting. <br> - `ask experience multi select`: Open-ended question without peer voting. |
| `Content`                                        | Text of the dialogue item (question or moderator statement). Corresponds to `Question Text` in other files.                                                                                                                                                                             |
| `Duration in minutes (dropdown)`                 | Estimated time for the participant to complete the item.                                                                                                                                                                                                                                |
| `Randomize options or categories for each participant` | Indicates if poll options were presented in random order (`yes`/`no`).                                                                                                                                                                                                                      |
| `Add 'Other' as an option`                       | Indicates if an 'Other' option was available for polls (`yes`/`no`).                                                                                                                                                                                                                      |
| `Add 'None of the above' as an option`           | Indicates if a 'None of the above' option was available for polls (`yes`/`no`).                                                                                                                                                                                                         |
| `Poll or Category Option 1`, `...Option N`       | Predefined response options for `poll single select` questions.                                                                                                                                                                                                                       |

### **`GD<N>_aggregate_standardized.csv`**

This is the primary data file for analysis, created by preprocessing the raw `aggregate.csv` file. It provides a standardized format with consistent headers across all question types and segments expressed as percentages.

| Column                | Description                                                                                                     |
| :-------------------- | :-------------------------------------------------------------------------------------------------------------- |
| `Question ID`         | Unique ID for the question.                                                                                      |
| `Question Type`       | Identifies the type of question: *Poll Single Select* or *Ask Opinion*.                                           |
| `Question`            | Text of the question presented to participants.                                                                   |
| `Response`            | For Poll questions: The response option. For Ask Opinion questions: The English translation of the response text. |
| `OriginalResponse`    | For Ask Opinion questions: The verbatim response in its original language.                                        |
| `Star`                | For Ask Opinion questions: Rating applied to the response.                                                        |
| `Categories`          | For Ask Experience questions: Categories assigned to the response.                                                |
| `Sentiment`           | For Ask Opinion questions: Sentiment analysis (Positive/Neutral/Negative) of the response.                        |
| `Submitted By`        | For Ask Opinion questions: Segment description of the participant who submitted the response.                     |
| `Language`            | For Ask Opinion questions: Original language of the response.                                                     |
| `Sample ID`           | Platform-specific ID related to the participant's sample source.                                                  |
| `Participant ID`      | Unique ID for the participant who authored the response.                                                          |
| `All`                 | Agreement rate among ALL participants as a percentage (numeric value without % symbol, e.g., 75.5).              |
| *Segment Columns*     | Various segment columns (regions, demographics, etc.) showing agreement rates as percentages.                    |

**Note:** This standardized format is the primary source for all analysis scripts. Segment names appear as column headers directly (without the `(N)` participant count suffix found in the raw file), and agreement rates are expressed as percentages (e.g., `75.5`).

### **`GD<N>_aggregate.csv`** (Raw Data)

This is the raw export file from Remesh.ai that compiles Global Dialogue data aggregated by question, showing the breakdown of each Segment's agreement rate with each question response.

| Column                | Description                                                                                                                                                                                                                                                                                                                      |
| :-------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Question ID`         | Unique ID for the question.                                                                                                                                                                                                                                                                                                      |
| `Question Type`       | Identifies the type of question: *Poll* or *Ask Opinion*. <br> - *Polls* provide participants with a list of options to select from. Some initial Poll questions define population segments (indicated in columns like `O1`, `O2`, etc.). <br> - *Ask Opinion* questions allow open-ended responses, followed by peer voting. |
| `Question`            | Text of the question presented to participants.                                                                                                                                                                                                                                                                                  |
| `Responses`           | (Poll Questions only) The multiple-choice response selected by the participant.                                                                                                                                                                                                                                                  |
| `English Response`    | (Opinion Questions only) Participant's response, machine-translated to English if necessary. Character limit: 500.                                                                                                                                                                                                               |
| `Original Response`   | (Opinion Questions only) Participant's verbatim response in the original language. Character limit: 500.                                                                                                                                                                                                                         |
| `Response Language`   | (Opinion Questions only) Original language of the submitted response.                                                                                                                                                                                                                                                            |
| `Submitted by`        | (Opinion Questions only) Segment description of the participant who submitted the response.                                                                                                                                                                                                                                      |
| `Participant ID`      | (Opinion Questions only) Unique ID for the participant who submitted the response.                                                                                                                                                                                                                                               |
| `Sentiment`           | (Opinion Questions only) Sentiment analysis (Positive/Neutral/Negative) of the response text, determined by the dialogue platform.                                                                                                                                                                                               |
| `All(N)`              | Shows the calculated agreement rate\* among ALL participants for the given response. `(N)` indicates the total number of participants in this segment.                                                                                                                                                                             |
| `O1: <language> (N)`  | Agreement rate among participants segmented by language.                                                                                                                                                                                                                                                                         |
| `O2: <age> (N)`       | Agreement rate among participants segmented by age.                                                                                                                                                                                                                                                                              |
| `O3: <gender> (N)`    | Agreement rate among participants segmented by gender.                                                                                                                                                                                                                                                                           |
| `O4: <urban/rural> (N)`| Agreement rate among participants segmented by urban/rural environment.                                                                                                                                                                                                                                                          |
| `O5: <concern AI> (N)`| Agreement rate among participants segmented by 'concern about AI'.                                                                                                                                                                                                                                                               |
| `O6: <religion> (N)`  | Agreement rate among participants segmented by religion.                                                                                                                                                                                                                                                                         |
| `O7: <country> (N)`   | Agreement rate among participants segmented by country.                                                                                                                                                                                                                                                                          |
| `<Region> (N)`        | Agreement rate among participants grouped by region.                                                                                                                                                                                                                                                                             |

**Note:** This raw file format is processed by the `preprocess_aggregate.py` script to create the standardized `GD<N>_aggregate_standardized.csv` file, which should be used for all analyses.

\* **Agreement Rate Definition:**
-   For *Poll* Questions: The percentage of participants who selected a specific response.
-   For *Ask Opinion* Questions: An estimated measure of the percentage of participants who indicated "Agree" with the given response (vs. "Disagree"). Due to the impracticality of all participants voting on every response, a prediction algorithm (~85% accuracy) imputes votes based on a limited sample (5 votes per participant per question).

### **`GD<N>_binary.csv`**

This file documents all individual votes cast on *Ask Opinion* question responses.

| Column           | Description                                                                   |
| :--------------- | :---------------------------------------------------------------------------- |
| `Question ID`    | Unique ID for the *Ask Opinion* question.                                     |
| `Participant ID` | Unique ID for the participant who voted.                                      |
| `Thought ID`     | Unique ID for the specific response (thought) being voted on.                 |
| `Vote`           | The vote cast: `agree`, `disagree`, or `neutral`.                             |
| `Timestamp`      | Timestamp of when the vote was cast.                                          |

### **`GD<N>_participants.csv`**

This file contains responses to every question, organized by participant.

| Column               | Description                                                                                                                                                                                                                                                                                             |
| :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Participant Id`     | Unique ID for the participant in the survey.                                                                                                                                                                                                                                                            |
| `Sample Provider Id` | Unique ID linking the participant to the originating survey platform (Prolific).                                                                                                                                                                                                                        |
| *Subsequent Columns* | Each subsequent column represents a question in chronological order. Column headers show the question text. <br> - *Poll* questions: Shows the selected response option. <br> - *Ask Opinion* questions: Shows the participant's text response in the original language. <br> - Additional columns with `(English)` suffix contain English translations of non-English responses. <br> - Columns with `All (%agree)` suffix show the estimated agreement percentage for that participant's response. |

### **`GD<N>_preference.csv`**

This file documents participant preferences between pairs of *Ask Opinion* responses, collected via pairwise comparison tasks. This data helps inform the machine learning algorithm used for calculating agreement rates.

| Column           | Description                                                                        |
| :--------------- | :--------------------------------------------------------------------------------- |
| `Question ID`    | Unique ID for the *Ask Opinion* question.                                          |
| `Participant ID` | Unique ID for the participant who voted.                                           |
| `Thought A ID`   | Unique ID for one response in the pair.                                            |
| `Thought B ID`   | Unique ID for the other response in the pair.                                      |
| `Vote`           | The preference indicated: `Thought A`, `Thought B`, `I agree with both`, `I disagree with both`. |
| `Timestamp`      | Timestamp of when the preference vote was cast.                                    |

### **`GD<N>_verbatim_map.csv`**

This file maps *Ask Opinion* question responses (thoughts) to their authors and the corresponding question.

| Column           | Description                                                     |
| :--------------- | :-------------------------------------------------------------- |
| `Question ID`    | Unique ID for the *Ask Opinion* question.                       |
| `Question Text`  | Text of the *Ask Opinion* question.                             |
| `Participant ID` | Unique ID for the participant who authored the response.        |
| `Thought ID`     | Unique ID for the response (thought).                           |
| `Thought Text`   | Verbatim text of the participant's response to the question.    |

### **`GD<N>_summary.csv`**

This file provides LLM-generated summaries for the dialogue as a whole and for each individual question included in the dataset.

| Column                | Description                                                                                      |
| :-------------------- | :----------------------------------------------------------------------------------------------- |
| `Conversation ID`     | Unique ID assigned to the dialogue conversation by the platform.                                 |
| `Conversation Title`  | Title of the conversation, often matching the overall `Title`.                                   |
| `Questions Selected`  | The number of questions from the dialogue included in this summary file.                         |
| `Summary Format`      | The format used for the overall dialogue summary (e.g., "Paragraph Summary").                    |
| `Conversation Summary`| An LLM-generated summary paragraph describing the key findings and themes of the entire dialogue. |
| `Question ID`         | Unique ID for the specific question being summarized.                                            |
| `Question Type`       | The type of question (e.g., `Poll`, `Ask Opinion`).                                               |
| `Question Text`       | The full text of the question presented to participants.                                         |
| `Question Summary`    | An LLM-generated summary paragraph describing the responses and findings for that specific question. |

### **`GD<N>_sanity_upload.csv`**

This file contains a subset of *Ask Opinion* questions and responses used for data quality verification.

| Column             | Description                                                                                      |
| :----------------- | :----------------------------------------------------------------------------------------------- |
| `Question`         | Text of the *Ask Opinion* question.                                                              |
| `English Response` | Participant's response, machine-translated to English if necessary.                             |
| `Agreement`        | Percentage agreement rate for this response among all participants.                              |
| *Segment Columns*  | Various demographic and geographic segments (e.g., `Male`, `Female`, `China`, etc.) with agreement rates. |

### **`GD<N>_segment_counts_by_question.csv`**

This file provides participant counts for each segment broken down by question, useful for understanding sample sizes and segment representation.

| Column         | Description                                                     |
| :------------- | :-------------------------------------------------------------- |
| `Question ID`  | Unique ID for the question.                                     |
| `Segment Name` | Name of the segment (matches column headers in aggregate files). |
| `Count`        | Number of participants in this segment who answered the question. |

### **`GD<N>_embeddings.json`**

This large JSON file contains semantic embeddings for questions and responses, enabling advanced analysis like thematic clustering and semantic similarity calculations. Due to its size (typically >500MB), this file is not stored in the Git repository and must be downloaded separately.

Structure:
- Contains question texts, response texts, and their corresponding embedding vectors
- Used by advanced analysis scripts like `thematic_ranking.py`
- Requires API key for regeneration if not available

### **Tag Data Files**

Starting with GD3, tagged categorizations of open-ended responses are collected. These appear in two directories:

#### **`tag_codes_raw/`**

Contains raw export files from the Remesh platform with timestamps. Multiple versions may exist for each question:
- `*_Tag_Categories.csv`: High-level category assignments for responses
- `*_Thought_Labels.csv`: Detailed labels for individual responses

#### **`tags/`** 

Contains processed tag files organized by Question ID after running `preprocess_tag_files.py`:
- `<question_id>_tag_categories.csv`: Category groupings for the question
- `<question_id>_thought_labels.csv`: Individual response labels
- `all_tag_categories.csv`: Consolidated categories across all questions
- `all_thought_labels.csv`: Consolidated labels across all responses

Tag file structure:

| Column          | Description                                                  |
| :-------------- | :----------------------------------------------------------- |
| `question_id`   | Unique ID linking to the *Ask Opinion* question.             |
| `category`      | High-level category name for grouping responses.             |
| `label`         | Specific label assigned to a response within the category.   |
| `thought_id`    | Unique ID for the response being labeled.                    |
| `thought_text`  | Text of the response being categorized.                      |



