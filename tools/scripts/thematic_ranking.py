import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import openai
from dotenv import load_dotenv
import datetime
import warnings
import uuid
import argparse

# --- Load Environment Variables --- Must be called early!
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Initialize OpenAI Client --- (v1.0.0+ style)
client = None
if OPENAI_API_KEY:
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
else:
    print("Warning: OPENAI_API_KEY not found in environment. OpenAI calls will fail.")

# --- Configuration ---
EXPECTED_EMBEDDING_DIM = 1024  # Defined based on known source model
TOP_N_RESULTS = 100  # Number of top results to save for each theme

# --- Column Names ---
EMBEDDING_COLUMN = 'embedding'
TEXT_COLUMN = 'English Responses'
QUESTION_ID_COLUMN = 'Question ID'
QUESTION_TEXT_COLUMN = 'Question'
PARTICIPANT_ID_COLUMN = 'Participant ID'

# Default paths - overridden by get_data_paths function
DEFAULT_GD_NUMBER = 3 
DEFAULT_DATA_FILE_PATH = os.path.join("Data", f"GD{DEFAULT_GD_NUMBER}", f"GD{DEFAULT_GD_NUMBER}_embeddings.json")
DEFAULT_OUTPUT_DIR = os.path.join("analysis_output", f"GD{DEFAULT_GD_NUMBER}", "thematic_rankings")

def load_thematic_queries(queries_file=None):
    """Load thematic queries from a text file, one per line."""
    if queries_file is None:
        queries_file = os.path.join(os.path.dirname(__file__), "thematic_queries.txt")
    
    try:
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        return queries
    except FileNotFoundError:
        print(f"Warning: Thematic queries file not found at {queries_file}")
        print("Using default themes.")
        # Fallback to default themes
        return [
            "faith and religion",
            "economic impacts and jobs",
            "human-AI relationships and collaboration",
            "cultural integrity and diversity",
            "safety and security concerns",
            "governance and regulation of AI"
        ]
    except Exception as e:
        print(f"Error loading thematic queries from {queries_file}: {e}")
        return []

def get_data_paths(gd_number):
    """Get the appropriate file paths based on GD number."""
    data_file_path = os.path.join("Data", f"GD{gd_number}", f"GD{gd_number}_embeddings.json")
    output_dir = os.path.join("analysis_output", f"GD{gd_number}", "thematic_rankings")
    return data_file_path, output_dir

def get_embedding(text, model="text-embedding-3-small", dimensions=EXPECTED_EMBEDDING_DIM):
    """Generates an embedding using the specified OpenAI model and dimensions."""
    if client is None:
        print("OpenAI client not initialized. Cannot get embedding.")
        return None
    if not text:
        print("Warning: Empty text passed to get_embedding.")
        return None

    try:
        text = text.replace("\n", " ")
        # Use specified model and dimensions
        response = client.embeddings.create(input=[text], model=model, dimensions=dimensions)
        return response.data[0].embedding
    except openai.AuthenticationError as e:
        print(f"OpenAI Authentication Error: {e}")
        return None
    except openai.RateLimitError as e:
        print(f"OpenAI Rate Limit Error: {e}")
        return None
    except openai.APIConnectionError as e:
        print(f"OpenAI API Connection Error: {e}")
        return None
    except openai.APIStatusError as e:
        print(f"OpenAI API Status Error: {e.status_code} - {e.response}")
        return None
    except Exception as e:
        # Catch potential dimension errors if the model doesn't support it
        if "dimensions" in str(e).lower():
            print(f"Error: Model '{model}' may not support the dimensions parameter ({dimensions}). Details: {e}")
        else:
            print(f"An unexpected error occurred during OpenAI embedding: {e}")
        return None

# --- Helper Functions ---

def load_data_with_embeddings(file_path):
    """Loads JSON containing a list of datasets, concatenates them, and returns a single pandas DataFrame."""
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        print("Please ensure you have downloaded the file and placed it correctly.")
        print("See README.md for instructions.")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            list_of_data = json.load(f)

        if not isinstance(list_of_data, list) or not list_of_data:
            print(f"Error: Expected a non-empty list in JSON file {file_path}")
            return None

        list_of_dfs = []
        for i, data_item in enumerate(list_of_data):
            try:
                df_part = pd.DataFrame(data_item)
                list_of_dfs.append(df_part)
            except ValueError as e:
                print(f"Warning: Could not convert item {i} from JSON list into DataFrame: {e}")
                continue

        if not list_of_dfs:
            print("Error: No valid DataFrames could be created from the JSON list.")
            return None

        combined_df = pd.concat(list_of_dfs, ignore_index=True)

        if combined_df.empty:
            print(f"Error: Concatenated DataFrame is empty.")
            return None

        if EMBEDDING_COLUMN not in combined_df.columns:
            print(f"Error: Combined DataFrame must contain '{EMBEDDING_COLUMN}' column.")
            return None

        print(f"Successfully loaded and combined {len(combined_df)} items into DataFrame from {file_path}")
        return combined_df
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return None

def validate_embeddings(embeddings_list):
    """
    Validates embeddings to prevent numpy warnings in similarity calculations.
    Returns clean embeddings and indices of valid embeddings.
    """
    valid_embeddings = []
    valid_indices = []
    
    for idx, emb in enumerate(embeddings_list):
        # Skip if not a list or not the right dimension
        if not isinstance(emb, list) or len(emb) != EXPECTED_EMBEDDING_DIM:
            continue
            
        # Check for NaN or Inf values
        has_invalid = False
        for val in emb:
            if not np.isfinite(val):
                has_invalid = True
                break
                
        # Convert to array and check if it's all zeros
        emb_array = np.array(emb)
        if np.all(emb_array == 0):
            has_invalid = True
            
        if not has_invalid:
            valid_embeddings.append(emb)
            valid_indices.append(idx)
    
    return valid_embeddings, valid_indices

def normalize_embeddings(embeddings_matrix):
    """
    Normalize embedding vectors to unit length to ensure valid cosine similarity.
    Prevents divide by zero and other numerical issues.
    """
    # Calculate the norm (length) of each vector
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    
    # Replace zero norms with 1 to avoid division by zero
    norms[norms == 0] = 1
    
    # Normalize by dividing each vector by its norm
    normalized_matrix = embeddings_matrix / norms
    
    return normalized_matrix

def rank_responses_by_similarity(response_df, query_text):
    """Ranks responses in a DataFrame based on cosine similarity to the query text's embedding."""
    # Get query embedding (1024 dimensions)
    query_embedding = get_embedding(query_text)

    if query_embedding is None:
        print(f"Could not get embedding for query: '{query_text}'. Skipping ranking.")
        return None
    if response_df is None or response_df.empty:
        print("No response DataFrame provided for ranking.")
        return None

    df_copy = response_df.copy()
    
    # Extract embeddings and validate them
    embeddings_list = df_copy[EMBEDDING_COLUMN].tolist()
    valid_embeddings, valid_indices = validate_embeddings(embeddings_list)

    if not valid_embeddings:
        print(f"No valid embeddings found matching dimension {EXPECTED_EMBEDDING_DIM} in the response DataFrame.")
        return None

    # Convert to numpy arrays and normalize
    embeddings_matrix = np.array(valid_embeddings)
    query_embedding_array = np.array([query_embedding])
    
    # Normalize both query and response embeddings
    normalized_query = normalize_embeddings(query_embedding_array)
    normalized_embeddings = normalize_embeddings(embeddings_matrix)
    
    # Suppress warnings during calculation
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        # Calculate cosine similarity
        similarities = cosine_similarity(normalized_query, normalized_embeddings)[0]

    # Map similarities back to original indices
    similarity_dict = {df_copy.index[idx]: sim for idx, sim in zip(valid_indices, similarities)}
    
    # Create a Series with similarities
    similarity_series = pd.Series(similarity_dict)
    df_copy['cosine_similarity'] = similarity_series
    
    # Add theme info
    df_copy['theme'] = query_text
    
    # Sort by similarity
    df_sorted = df_copy.sort_values(by='cosine_similarity', ascending=False, na_position='last')
    
    return df_sorted

def save_thematic_rankings(all_rankings, output_dir=DEFAULT_OUTPUT_DIR, top_n=TOP_N_RESULTS):
    """
    Save thematic rankings to a single CSV file containing all themes.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get timestamp for metadata
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # List to store all top results
    all_top_results = []
    
    # Generate unique run ID
    run_id = str(uuid.uuid4())[:8]
    
    # Process each theme's results
    for theme, df in all_rankings.items():
        if df is None or df.empty:
            print(f"Skipping empty results for theme: '{theme}'")
            continue
        
        # Get top N results
        top_results = df.head(top_n).reset_index(drop=True)
        
        # Add metadata
        top_results['run_id'] = run_id
        top_results['timestamp'] = timestamp
        
        # Add to combined results
        all_top_results.append(top_results)
    
    # Save single file with all themes
    if all_top_results:
        combined_results = pd.concat(all_top_results, ignore_index=True)
        output_file = os.path.join(output_dir, "thematic_rankings.csv")
        
        try:
            # Sort by theme then similarity
            combined_results = combined_results.sort_values(
                by=['theme', 'cosine_similarity'], 
                ascending=[True, False]
            )
            
            # Select columns to save (exclude embedding to save space)
            columns_to_save = [
                'theme', 'cosine_similarity', TEXT_COLUMN, 
                QUESTION_ID_COLUMN, QUESTION_TEXT_COLUMN, PARTICIPANT_ID_COLUMN,
                'run_id', 'timestamp'
            ]
            
            # Filter to columns that actually exist
            existing_columns = [col for col in columns_to_save if col in combined_results.columns]
            
            # Save to CSV
            combined_results[existing_columns].to_csv(output_file, index=False, encoding='utf-8')
            print(f"Saved all thematic rankings to {output_file}")
        except Exception as e:
            print(f"Error saving thematic rankings: {e}")
    else:
        print("No valid rankings to save")
    
    return run_id, timestamp

# --- Main Execution ---

if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Run thematic ranking analysis on Global Dialogues data')
    parser.add_argument('--gd', type=int, choices=[1, 2, 3], required=True,
                      help='Global Dialogue number to analyze (1, 2, or 3)')
    parser.add_argument('--themes', type=str, 
                      help='Path to text file containing thematic queries (one per line)')
    args = parser.parse_args()

    # Get appropriate file paths
    DATA_FILE_PATH, OUTPUT_DIR = get_data_paths(args.gd)

    print(f"Loading data for GD{args.gd}...")
    survey_df = load_data_with_embeddings(DATA_FILE_PATH)

    if survey_df is not None:
        # Load thematic queries
        thematic_queries = load_thematic_queries(args.themes)
        if not thematic_queries:
            print("No thematic queries found. Exiting.")
            return
        
        print(f"\n--- Starting Thematic Ranking with {len(thematic_queries)} themes ---")
        all_rankings = {}
        
        for theme in thematic_queries:
            print(f"\nRanking responses for theme: '{theme}'")
            ranked_df = rank_responses_by_similarity(survey_df, theme)

            if ranked_df is not None:
                all_rankings[theme] = ranked_df
                print(f"Top 5 most similar responses for '{theme}':")
                for i, row in ranked_df.head(5).iterrows():
                    response_text = row.get(TEXT_COLUMN, 'N/A (Check TEXT_COLUMN)')
                    similarity = row.get('cosine_similarity', np.nan)
                    # Ensure similarity is a number before formatting
                    sim_str = f"{similarity:.4f}" if isinstance(similarity, (int, float)) and not np.isnan(similarity) else "NaN"
                    print(f"  {i+1}. Similarity: {sim_str} - \"{response_text[:100]}...\"")
            else:
                print(f"Could not generate rankings for theme: '{theme}'")

        # Save results to CSV
        run_id, timestamp = save_thematic_rankings(all_rankings, OUTPUT_DIR)
        print(f"\nAll thematic rankings saved to {os.path.join(OUTPUT_DIR, 'thematic_rankings.csv')}")
        print(f"Run ID: {run_id}, Timestamp: {timestamp}")
        print("\n--- Thematic Ranking Complete ---")
    else:
        print("Could not load data. Exiting.")