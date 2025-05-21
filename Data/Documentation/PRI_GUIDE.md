# Participant Reliability Index (PRI)

## Overview

The Participant Reliability Index (PRI) is a composite score designed to assess the reliability and quality of participant responses in the Global Dialogues survey. It combines multiple signals to create a comprehensive measure of participant engagement and response quality.

## Components

### 1. Duration
- **Metric**: `Duration_seconds`
- **Description**: Total time spent by participants in the survey, measured through binary votes and preference judgments
- **Purpose**: Identifies rushed or inattentive responses
- **Calculation**: Time difference between first and last recorded activity

### 2. Low Quality Tag Percentage
- **Metric**: `LowQualityTag_Perc`
- **Description**: Proportion of a participant's responses tagged as "Uninformative Answer"
- **Purpose**: Identifies participants who consistently provide low-quality responses
- **Calculation**: Number of low-quality tagged responses / Total number of responses
- **Data Source**: Thought labels from manual or automated tagging

### 3. Universal Disagreement Percentage
- **Metric**: `UniversalDisagreement_Perc`
- **Description**: Frequency of responses that receive widespread disagreement across demographic segments
- **Purpose**: Identifies participants whose responses fail to resonate with the broader population
- **Calculation**: 
  - Counts responses with agreement rates below 20% across 90% of major segments
  - Divides by total number of evaluated responses
- **Thresholds**:
  - Agreement Rate: < 20%
  - Segment Coverage: > 90%

### 4. Anti-Social Consensus Score
- **Metric**: `ASC_Score_Raw`
- **Description**: Rate of voting against strong consensus
- **Purpose**: Identifies participants who consistently disagree with widely agreed-upon responses
- **Calculation**:
  - Identifies thoughts with strong consensus:
    - High consensus: > 80% agreement
    - Low consensus: < 20% agreement
  - Calculates proportion of votes against consensus
  - Lower scores indicate better reliability (requires inversion in final calculation)

## Implementation

The PRI is implemented in `tools/scripts/calculate_pri.py`. The script:
1. Loads and processes necessary data files
2. Calculates individual component scores
3. Normalizes and weights the components
4. Produces a final composite score

## Usage

The PRI can be used to:
- Filter out unreliable participants from analysis
- Weight participant contributions based on reliability
- Identify patterns in participant engagement and response quality
- Monitor survey quality across different demographic segments

## Future Improvements

Potential enhancements to the PRI include:
- Dynamic threshold adjustment based on survey context
- Additional quality signals (e.g., response length, language complexity)
- Machine learning-based reliability assessment
- Integration with other quality metrics 