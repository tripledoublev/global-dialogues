# Placeholder for shared analysis utility functions
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TODO: Define utility functions (e.g., load_data, parse_percentage, etc.)

def load_standardized_data(csv_path):
    """Loads the standardized aggregate CSV into a pandas DataFrame."""
    logging.info(f"Loading standardized data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        logging.info(f"Successfully loaded dataframe with shape: {df.shape}")
        # Basic validation - check for expected columns
        expected_cols = ["Question ID", "Question Type", "Question", "Responses", "Participant ID"]
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            logging.warning(f"Standardized CSV missing expected columns: {missing_cols}")
        return df
    except FileNotFoundError:
        logging.error(f"Standardized data file not found: {csv_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading standardized data from {csv_path}: {e}")
        return None

# Example utility function (to be moved/refined)
def parse_percentage(value):
    if isinstance(value, str) and '%' in value:
        try:
            return float(value.strip('%')) / 100.0
        except ValueError:
            return pd.NA # Use pandas NA for missing numeric values
    elif pd.isna(value) or value == '' or value == '-': # Handle various missing representations
        return pd.NA
    try:
        # Attempt to convert directly if it's already numeric-like
        return float(value) / 100.0 if isinstance(value, (int, float)) and value > 1 else float(value)
    except (ValueError, TypeError):
         return pd.NA # Return NA if conversion fails

print("analysis_utils.py created.") # Placeholder print 