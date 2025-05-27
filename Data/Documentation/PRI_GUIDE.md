# Participant Reliability Index (PRI)

## Overview

The Participant Reliability Index (PRI) is a composite score designed to assess the reliability and quality of participant responses in the Global Dialogues survey. It combines multiple signals to create a comprehensive measure of participant engagement and response quality.

## Components

### 1. Duration
- **Metric**: `Duration_seconds`
- **Description**: Total time spent by participants in the survey, measured through binary votes and preference judgments
- **Purpose**: Identifies rushed or inattentive responses
- **Calculation**: Time difference between first and last recorded activity
- **Thresholds**:
  - Reasonable Max Duration: 90 minutes

### 2. Low Quality Tag Percentage
- **Metric**: `LowQualityTag_Perc`
- **Description**: Proportion of a participant's responses tagged as "Uninformative Answer"
- **Purpose**: Identifies participants who consistently provide low-quality responses
- **Calculation**: Number of low-quality tagged responses / Total number of responses
- **Data Source**: Thought labels from manual or automated tagging

### 3. Universal Disagreement Percentage
- **Metric**: `UniversalDisagreement_Perc`
- **Description**: Frequency of responses that receive widespread disagreement across major demographic segments
- **Purpose**: Identifies participants whose responses fail to resonate with any major demographic group
- **Calculation**: 
  - Identifies major segments (those with ≥50 average participants across questions)
  - Counts responses where EITHER overall agreement < 30% OR no major segment has agreement ≥ 40%
  - Divides by total number of evaluated responses
- **Thresholds**:
  - Agreement Rate (All): < 30%
  - Agreement Rate (Major Segments): < 40% (no single major segment above this threshold)
  - Major Segment Definition: ≥ 50 average participants
- **Implementation**: Uses dynamic major segment detection from segment counts file

### 4. Anti-Social Consensus Score
- **Metric**: `ASC_Score_Raw`
- **Description**: Rate of voting against strong consensus
- **Purpose**: Identifies participants who consistently disagree with widely agreed-upon responses
- **Calculation**:
  - Identifies thoughts with strong consensus:
    - High consensus: ≥ 70% agreement
    - Low consensus: ≤ 30% agreement
  - Calculates proportion of votes against consensus
  - Lower scores indicate better reliability (requires inversion in final calculation)
- **Thresholds**:
  - High Consensus: ≥ 70% agreement
  - Low Consensus: ≤ 30% agreement

### 5. Length of Response (Not implemented)
- **Description**: Analysis of response length as a quality signal
- **Status**: Not currently implemented but could be added as additional signal

### 6. Response Reading Time (Not implemented - dataset not available)
- **Description**: Time from seeing Response options and selecting a vote (Agree/Disagree or Preference). Can also compute on Poll Options especially for longer questions. Requires dataset with timestamps on loading Response options and voting (not yet available).
- **Status**: Requires additional timestamp data not currently available

### 7. LLM Judgment of Question Responses (Implemented)
- **Metric**: `LLM_Judge_Score`
- **Description**: Multi-model LLM assessment of participant earnestness based on responses to "Ask Opinion" questions
- **Purpose**: Provides qualitative assessment of response authenticity and engagement quality
- **Calculation**: 
  - Extracts participant responses to open-ended "Ask Opinion" questions from discussion guide
  - Submits responses to multiple high-quality LLM models for assessment
  - Models evaluate earnestness, thoughtfulness, consistency, and engagement
  - Returns confidence score (0.0-1.0) with reasoning
  - Final score is average across all model assessments
- **Models Used**:
  - Anthropic Claude Sonnet 4
  - OpenAI GPT-4o-mini
  - Google Gemini 2.5 Pro Preview
- **API Integration**: OpenRouter.ai with async processing for efficiency
- **Toggle**: Enabled via `--llm-judge` flag (costs money and takes longer)
- **Correlation Analysis**: Automatically analyzes correlation with traditional PRI components


## Implementation

The PRI is implemented in `tools/scripts/calculate_pri.py`. The script:
1. Loads and processes necessary data files, including dynamic major segment detection
2. Calculates individual component scores using major segments for Universal Disagreement
3. Normalizes and weights the components with reasonable maximum capping for Duration
4. Produces a final composite score (0-1 scale) and 1-5 scale version

### Component Weights

**Traditional PRI (without LLM judge):**
- Duration: 30%
- Low Quality Tags: 30% 
- Universal Disagreement: 20%
- Anti-Social Consensus: 20%

**Enhanced PRI (with LLM judge):**
- Duration: 20%
- Low Quality Tags: 20%
- Universal Disagreement: 15%
- Anti-Social Consensus: 15%
- LLM Judge: 30%

### Normalization
- **Duration**: Min-max normalization with 90-minute reasonable maximum (longer is better)
- **Low Quality Tags**: Min-max normalization, inverted (lower percentage is better)
- **Universal Disagreement**: Min-max normalization, inverted (lower percentage is better)
- **Anti-Social Consensus**: Min-max normalization, inverted (lower score is better)
- **LLM Judge**: Min-max normalization (higher score is better, no inversion needed)

## Usage

The PRI can be used to:
- Filter out unreliable participants from analysis
- Weight participant contributions based on reliability
- Identify patterns in participant engagement and response quality
- Monitor survey quality across different demographic segments

## Data Requirements

The PRI calculator requires the following files:
- `GD{N}_binary.csv`: Binary vote data with timestamps
- `GD{N}_preference.csv`: Preference judgment data with timestamps  
- `GD{N}_verbatim_map.csv`: Mapping of thoughts to participants and questions
- `GD{N}_aggregate_standardized.csv`: Agreement scores by segment
- `GD{N}_segment_counts_by_question.csv`: Participation counts by segment (for major segment detection)
- `GD{N}_discussion_guide.csv`: Survey structure with question types (required for LLM judge)
- `tags/all_thought_labels.csv`: Quality tags for responses (optional)

### Additional Requirements for LLM Judge
- OpenRouter.ai API key in `.env` file as `OPENROUTER_API_KEY`
- Internet connection for API calls
- Additional dependencies: `aiohttp`, `pydantic`

## Future Improvements

Potential enhancements to the PRI include:
- Dynamic threshold adjustment based on survey context
- Additional quality signals (e.g., response length, language complexity)
- Machine learning-based reliability assessment
- Integration with other quality metrics
- Statistical randomness detection for voting patterns 