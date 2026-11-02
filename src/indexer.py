import os
import pickle

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer


class CustomIndex:
    """
    A utility class for creating, storing, and retrieving semantic embeddings for text documents using
    a SentenceTransformer model. Enables indexing, query embedding, and similarity-based retrieval.

    Core Features:
    - Generates embeddings for documents and queries.
    - Creates and persists an index as a pickle file.
    - Loads and validates existing index files.
    - Retrieves top-K similar documents based on cosine similarity.

    This is the "retrieval" half of the RAG pipeline. There is no vector database:
    the index is just a dict holding a numpy matrix of embeddings plus the
    original FAQ metadata, saved to disk with pickle.
    """

    def __init__(self):
        # Cache of loaded SentenceTransformer models, keyed by model name,
        # so the (slow) model load only happens once per model
        self._models = {}
        # The most recently created/loaded index dict (None until one exists)
        self.index = None

    def _get_model(self, model_name: str) -> SentenceTransformer:
        """Return a cached SentenceTransformer, loading/downloading it on first use."""
        if model_name not in self._models:
            self._models[model_name] = SentenceTransformer(model_name)
        return self._models[model_name]

    def generate_embeddings(self, texts: list, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
        """
        Generate sentence embeddings for a list of input texts using a specified SentenceTransformer model.

        Args:
            texts (list): A non-empty list of input strings to embed.
            model_name (str): Name of the SentenceTransformer model to use.

        Returns:
            np.ndarray: Numpy array of embeddings corresponding to the input texts.

        Raises:
            ValueError: If input is not a non-empty list or if embedding generation fails.
        """
        # --- Validate input ---
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValueError("Input 'texts' must be a non-empty list of strings.")

        if not all(isinstance(t, str) for t in texts):
            raise ValueError("All elements in 'texts' must be strings.")

        try:
            model = self._get_model(model_name)
            # Each text becomes a fixed-size vector (384 dims for all-MiniLM-L6-v2)
            embeddings = model.encode(texts, normalize_embeddings=True)
            embeddings = np.array(embeddings, dtype=np.float32)
            # Ensure L2-normalized embeddings (defensive: re-normalize in case the
            # model's own normalization is off). With unit-length vectors, cosine
            # similarity reduces to a simple dot product.
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Replace zero norms with a tiny value to avoid division by zero
            norms = np.where(norms == 0, 1e-12, norms)
            embeddings = embeddings / norms
            return embeddings
        except Exception as e:
            raise ValueError(f"Embedding generation failed: {e}") from e

    def create_index(self, df: pd.DataFrame, index_file_name: str, index_folder_name: str, model_name: str = 'all-MiniLM-L6-v2') -> dict:
        """
        Builds and saves an index from a DataFrame containing document titles and content.

        Args:
            df (pd.DataFrame): DataFrame with 'id', 'title', and 'content' columns.
            index_file_name (str): Name of the pickle file to save the index.
            index_folder_name (str): Directory to store the index file.
            model_name (str): SentenceTransformer model to use for embedding generation.

        Returns:
            dict: A dictionary containing embeddings, document IDs, titles, and content.

        Raises:
            ValueError: If input is not a DataFrame or if internal validation fails.
        """
        # --- Validate the DataFrame and its schema ---
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input 'df' must be a pandas DataFrame.")

        if df.empty or len(df) == 0:
            raise ValueError("DataFrame cannot be empty.")

        required_columns = {'id', 'title', 'content'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"DataFrame is missing required columns: {sorted(missing_columns)}")

        # Combine title and content for comprehensive semantic retrieval
        # (the title often holds the question-like phrasing users search with)
        texts = [f"{row['title']} {row['content']}".strip() for _, row in df.iterrows()]
        embeddings = self.generate_embeddings(texts, model_name=model_name)

        # Row i of 'embeddings' lines up with position i of every list below,
        # which is how a similarity match is mapped back to its FAQ entry
        index_data = {
            'embeddings': embeddings,
            'ids': df['id'].tolist(),
            'titles': df['title'].tolist(),
            'contents': df['content'].tolist()
        }

        # Persist to disk so embeddings don't need recomputing on every app start
        try:
            os.makedirs(index_folder_name, exist_ok=True)
            file_path = os.path.join(index_folder_name, index_file_name)
            with open(file_path, 'wb') as f:
                pickle.dump(index_data, f)
        except Exception as e:
            raise ValueError(f"Failed to persist index file: {e}") from e

        self.index = index_data
        return index_data

    def build_index(self, index_file_name: str, index_folder_name: str) -> dict:
        """
        Loads an existing index from disk and validates its structure.

        Args:
            index_file_name (str): Name of the index pickle file to load.
            index_folder_name (str): Directory where the index file is stored.

        Returns:
            dict: Dictionary containing the loaded index data with keys: 'embeddings', 'ids', 'titles', 'contents'.

        Raises:
            FileNotFoundError: If the specified index file does not exist.
            ValueError: If the loaded file lacks the required index structure.
        """
        file_path = os.path.join(index_folder_name, index_file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Index file does not exist: {file_path}")

        # Note: only unpickle files you created yourself - pickle can execute
        # arbitrary code when loading untrusted data
        try:
            with open(file_path, 'rb') as f:
                index_data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to read index file: {e}") from e

        # --- Sanity-check the loaded structure so a stale/corrupt file fails clearly ---
        if not isinstance(index_data, dict):
            raise ValueError("Corrupted index file: expected a dictionary.")

        required_keys = {'embeddings', 'ids', 'titles', 'contents'}
        missing_keys = required_keys - set(index_data.keys())
        if missing_keys:
            raise ValueError(f"Index structure is invalid, missing keys: {sorted(missing_keys)}")

        self.index = index_data
        return index_data

    def get_query_embedding(self, query: str, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
        """
        Encodes a query string into a single vector embedding using a specified SentenceTransformer model.

        Args:
            query (str): Input query text to embed.
            model_name (str): Name of the SentenceTransformer model to use.

        Returns:
            np.ndarray: Vector embedding representing the input query.

        Raises:
            ValueError: If query is not a valid non-empty string or embedding fails.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a valid non-empty string.")

        try:
            # IMPORTANT: must use the same model as the one used to build the
            # index, otherwise query and document vectors are not comparable
            model = self._get_model(model_name)
            embedding = model.encode(query, normalize_embeddings=True)
            embedding = np.array(embedding, dtype=np.float32)
            # Re-normalize to unit length (skip if zero vector to avoid divide-by-zero)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception as e:
            raise ValueError(f"Failed to generate query embedding: {e}") from e

    def find_top_k_matches(self, query_embedding: np.ndarray, index: dict, k: int = 3) -> list:
        """
        Finds the top-k most semantically similar documents in the index based on cosine similarity.

        Args:
            query_embedding (np.ndarray): Vector embedding of the input query.
            index (dict): Index containing pre-computed document embeddings and metadata.
            k (int): Number of top results to return (default is 3).

        Returns:
            list: A list of tuples (index, similarity_score) for the top-k most similar documents.

        Raises:
            ValueError: If `k` is not positive, or if input types/structure are invalid.
            RuntimeError: If similarity computation fails unexpectedly.
        """
        # --- Validate arguments ---
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Argument 'k' must be a positive integer.")

        if not isinstance(query_embedding, np.ndarray):
            raise ValueError("query_embedding must be a numpy ndarray.")

        if not isinstance(index, dict):
            raise ValueError("index must be a dictionary.")

        if 'embeddings' not in index or index['embeddings'] is None:
            raise ValueError("index must contain an 'embeddings' entry.")

        try:
            doc_embeddings = np.array(index['embeddings'])
            if len(doc_embeddings) == 0:
                raise ValueError("Index contains no document embeddings.")

            # Flatten to 1-D and normalize the query vector
            q_vec = query_embedding.flatten()
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm

            # Brute-force search: compare the query against every document vector.
            # O(n) per query - fine for a small FAQ set, but would need a real
            # vector index (e.g. FAISS) for large corpora.
            results = []
            for idx, doc_vec in enumerate(doc_embeddings):
                d_vec = np.array(doc_vec).flatten()
                d_norm = np.linalg.norm(d_vec)
                if d_norm > 0:
                    d_vec = d_vec / d_norm

                # Compute cosine distance using scipy.spatial.distance.cosine
                # similarity = 1 - distance (1.0 = identical direction, 0 = unrelated).
                # A NaN distance (e.g. zero vector) is treated as zero similarity.
                cos_dist = cosine(q_vec, d_vec)
                sim = float(1.0 - cos_dist) if not np.isnan(cos_dist) else 0.0
                results.append((idx, sim))

            # Most similar first, then keep only the top k (idx, score) pairs
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
        except ValueError:
            # Our own validation errors pass through unchanged
            raise
        except Exception as e:
            raise RuntimeError(f"Error computing top-k matches: {e}") from e
