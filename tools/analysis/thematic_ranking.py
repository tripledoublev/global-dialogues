import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import openai
from dotenv import load_dotenv

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
DATA_FILE_PATH = os.path.join("Data", "GD3", "GD3_embeddings.json")
EXPECTED_EMBEDDING_DIM = 1024 # Defined based on known source model

# --- Potential Column Names --- Update if necessary ---
EMBEDDING_COLUMN = 'embedding'
TEXT_COLUMN = 'English Responses'

# Define the standard thematic queries
THEMATIC_QUERIES = [
    "faith and religion",
    "economic impacts and jobs",
    "human-AI relationships and collaboration",
    "cultural integrity and diversity",
    "safety and security concerns",
    "governance and regulation of AI",
    # Add more themes as needed
]

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

def rank_responses_by_similarity(response_df, query_text):
    """Ranks responses in a DataFrame based on cosine similarity to the query text's embedding."""
    # Query embedding will now be 1024 dimensions
    query_embedding = get_embedding(query_text)

    if query_embedding is None:
        print(f"Could not get embedding for query: '{query_text}'. Skipping ranking.")
        return None
    if response_df is None or response_df.empty:
        print("No response DataFrame provided for ranking.")
        return None

    df_copy = response_df.copy()

    valid_embeddings = []
    valid_indices = []
    # Use the known expected dimension
    embedding_dim = EXPECTED_EMBEDDING_DIM # Should be 1024

    embedding_series = df_copy[EMBEDDING_COLUMN]

    for index, embedding in embedding_series.items():
        if isinstance(embedding, list) and len(embedding) == embedding_dim:
            valid_embeddings.append(embedding)
            valid_indices.append(index)
        elif isinstance(embedding, list) and len(embedding) != embedding_dim:
             print(f"Warning: Skipping row index {index}. Embedding dimension {len(embedding)} != expected {embedding_dim}.")

    if not valid_embeddings:
        # This error message should now be accurate if it appears
        print(f"No valid embeddings found matching dimension {embedding_dim} in the response DataFrame.")
        return None

    embeddings_matrix = np.array(valid_embeddings)
    similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]

    similarity_series = pd.Series(similarities, index=valid_indices)
    df_copy['cosine_similarity'] = similarity_series

    df_sorted = df_copy.sort_values(by='cosine_similarity', ascending=False, na_position='last')

    return df_sorted

# --- Main Execution ---

if __name__ == "__main__":
    print("Loading data...")
    survey_df = load_data_with_embeddings(DATA_FILE_PATH)

    if survey_df is not None:
        print("\n--- Starting Thematic Ranking ---")
        all_rankings = {}
        for theme in THEMATIC_QUERIES:
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

        # Example: Save top N results for each theme to CSV
        # ... (saving logic remains the same, uses ranked_df)
        # output_dir = "thematic_rankings_output"
        # os.makedirs(output_dir, exist_ok=True)
        # for theme, df in all_rankings.items():
        #     safe_theme_name = "".join(c if c.isalnum() else '_' for c in theme) # Sanitize filename
        #     output_file = os.path.join(output_dir, f"ranking_{safe_theme_name}.csv")
        #     try:
        #         # Make sure columns exist before saving
        #         cols_to_save = [TEXT_COLUMN, 'cosine_similarity'] # Add other relevant columns if needed
        #         cols_exist = [col for col in cols_to_save if col in df.columns]
        #         if EMBEDDING_COLUMN not in df.columns: # Don't save embedding if column name is uncertain
        #              cols_exist.append(EMBEDDING_COLUMN)
        #         df.head(100)[cols_exist].to_csv(output_file, index=False, encoding='utf-8')
        #         print(f"Saved top 100 rankings for '{theme}' to {output_file}")
        #     except Exception as e:
        #          print(f"Error saving rankings for '{theme}' to CSV: {e}")

        print("\n--- Thematic Ranking Complete ---")
    else:
        print("Could not load data. Exiting.") 