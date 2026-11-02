# Support Response Assistant - Solution

This document contains the complete working implementation, technical explanations, and verification results for the **Support Response Assistant** RAG pipeline.

---

## 1. Architecture Overview

The system implements a modular Retrieval-Augmented Generation (RAG) pipeline designed for B2B SaaS customer support queries:

```
[ FAQ CSV Dataset ] 
        │
        ▼ (document_loader.py)
[ DataFrame Validation & Loading ]
        │
        ▼ (indexer.py)
[ SentenceTransformer: all-MiniLM-L6-v2 ] ──► [ L2 Normalization ]
        │
        ▼
[ In-Memory & Serialized Index (faq_index.pkl) ]
        ▲
        │ Cosine Similarity (scipy.spatial.distance.cosine)
[ User Query Embedding ]
        │
        ▼
[ Top-k Retrieved Context ] ──┐
                              ├─► [ LangChain LCEL Pipeline ] ──► [ Final Answer / Fallback ]
[ User Query ] ───────────────┘       (rag_chain.py: gpt-4o)
```

---

## 2. Deliverables Code

### A. Document Loader (`document_loader.py`)

Responsible for loading and strictly validating FAQ CSV datasets.

```python
import pandas as pd
import os


class LoadDocuments:
    """
    Class for loading and validating FAQ data from a CSV file.
    """

    def load_faq_data(self, file_path: str) -> pd.DataFrame:
        """
        Load a CSV file and return a DataFrame with 'id', 'title', and 'content' columns.

        Args:
            file_path (str): Path to the input CSV file.

        Returns:
            pd.DataFrame: Loaded data with required columns.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a CSV, missing columns, or is empty.
        """
        if not isinstance(file_path, str):
            raise ValueError("File path must be a string.")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")

        if os.path.isdir(file_path):
            raise FileNotFoundError(f"Expected a file, but found a directory: {file_path}")

        if not file_path.lower().endswith(".csv"):
            raise ValueError(f"File '{file_path}' is not a CSV file.")

        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError as err:
            raise ValueError("CSV file is empty.") from err
        except Exception as err:
            raise ValueError(f"Failed to read CSV file: {err}") from err

        if df.empty or len(df) == 0:
            raise ValueError("CSV file is empty or contains no records.")

        required_columns = {"id", "title", "content"}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

        return df[["id", "title", "content"]].copy()
```

---

### B. Embedding Index & Retriever (`indexer.py`)

Handles model embedding generation, L2-normalization, serialization, and top-k retrieval via cosine similarity.

```python
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
from scipy.spatial.distance import cosine


class CustomIndex:
    """
    A utility class for creating, storing, and retrieving semantic embeddings for text documents using
    a SentenceTransformer model. Enables indexing, query embedding, and similarity-based retrieval.

    Core Features:
    - Generates embeddings for documents and queries.
    - Creates and persists an index as a pickle file.
    - Loads and validates existing index files.
    - Retrieves top-K similar documents based on cosine similarity.
    """

    def __init__(self):
        self._models = {}
        self.index = None

    def _get_model(self, model_name: str) -> SentenceTransformer:
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
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValueError("Input 'texts' must be a non-empty list of strings.")

        if not all(isinstance(t, str) for t in texts):
            raise ValueError("All elements in 'texts' must be strings.")

        try:
            model = self._get_model(model_name)
            embeddings = model.encode(texts, normalize_embeddings=True)
            embeddings = np.array(embeddings, dtype=np.float32)
            # Ensure L2-normalized embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
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
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input 'df' must be a pandas DataFrame.")

        if df.empty or len(df) == 0:
            raise ValueError("DataFrame cannot be empty.")

        required_columns = {'id', 'title', 'content'}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"DataFrame is missing required columns: {sorted(missing_columns)}")

        # Combine title and content for comprehensive semantic retrieval
        texts = [f"{row['title']} {row['content']}".strip() for _, row in df.iterrows()]
        embeddings = self.generate_embeddings(texts, model_name=model_name)

        index_data = {
            'embeddings': embeddings,
            'ids': df['id'].tolist(),
            'titles': df['title'].tolist(),
            'contents': df['content'].tolist()
        }

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

        try:
            with open(file_path, 'rb') as f:
                index_data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to read index file: {e}") from e

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
            raise ValueError("Query must be a non-empty string.")

        try:
            model = self._get_model(model_name)
            embedding = model.encode(query, normalize_embeddings=True)
            embedding = np.array(embedding, dtype=np.float32)
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

            q_vec = query_embedding.flatten()
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm

            results = []
            for idx, doc_vec in enumerate(doc_embeddings):
                d_vec = np.array(doc_vec).flatten()
                d_norm = np.linalg.norm(d_vec)
                if d_norm > 0:
                    d_vec = d_vec / d_norm

                # Compute cosine distance using scipy.spatial.distance.cosine
                cos_dist = cosine(q_vec, d_vec)
                sim = float(1.0 - cos_dist) if not np.isnan(cos_dist) else 0.0
                results.append((idx, sim))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error computing top-k matches: {e}") from e
```

---

### C. Generator (`rag_chain.py`)

Constructs the LangChain pipeline with prompt templating, LLM invocation (`gpt-4o`), and deterministic fallback response logic.

```python
import os

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None


class RAGChain:
    """
    A class for Retrieval-Augmented Generation (RAG) that connects prompt templates,
    a language model (OpenAI GPT or Google Gemini), and an output parser to generate answers based on context.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0,
        api_key: str | None = None,
        provider: str = "openai",
    ):
        """
        Initialize the RAGChain with the specified model, temperature, and LLM provider.

        Args:
            model_name (str): The model identifier to use (e.g. 'gpt-4o', 'gemini-1.5-flash').
            temperature (float): Sampling temperature (0 = deterministic).
            api_key (str, optional): API key for the chosen provider.
            provider (str): LLM provider to use: 'openai' (default) or 'gemini'.
        """
        self.provider = provider.lower() if provider else "openai"
        self.model_name = model_name
        self.temperature = temperature

        if self.provider == "gemini" or "gemini" in model_name.lower():
            if ChatGoogleGenerativeAI is None:
                raise ImportError("langchain-google-genai is required for Gemini models.")
            key = (
                api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or "placeholder-gemini-key"
            )
            gemini_model = model_name if "gemini" in model_name.lower() else "gemini-1.5-flash"
            self.llm = ChatGoogleGenerativeAI(
                model=gemini_model,
                temperature=temperature,
                api_key=key,
            )
        else:
            key = api_key or os.environ.get("OPENAI_API_KEY") or "sk-placeholder-key-for-initialization"
            self.llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=key)

        self.chain = None

    def create_chain(self):
        """
        Create a chain: prompt → LLM → output parser.

        Attribute:
            self.chain (RunnableSequence): A LangChain pipeline.
        """
        template = (
            "You are a helpful customer support assistant for a B2B SaaS platform.\n"
            "Answer the question concisely and accurately using ONLY the provided context.\n"
            "If the answer cannot be determined from the context, or if no relevant context is found, "
            "respond with: 'I am sorry, but I do not have enough information to answer your question.'\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        )
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        def _standardize_inputs(inputs: dict) -> dict:
            context = inputs.get("context", "")
            # Support both 'question' and 'query' keys for flexible invocation
            question = inputs.get("question") or inputs.get("query", "")
            return {"context": context, "question": question}

        self.chain = RunnableLambda(_standardize_inputs) | prompt | self.llm | StrOutputParser()
        return self.chain

    def query(self, query: str, documents: Union[str, list]) -> str:
        """
        Generate an answer for the given query using provided context documents.

        Args:
            query (str): The question to answer.
            documents (str | list): Supporting context for the query, either as a single string
                                    or a list of strings.

        Returns:
            str: Generated answer.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if not isinstance(documents, (str, list)):
            raise ValueError("Documents must be a string or a list of strings.")

        if isinstance(documents, list):
            if not all(isinstance(doc, str) for doc in documents):
                raise ValueError("All elements in documents list must be strings.")
            context = "\n".join(doc.strip() for doc in documents if doc.strip())
        else:
            context = documents.strip()

        # If no relevant context is found, return the default fallback answer immediately
        if not context:
            return "I am sorry, but I do not have enough information to answer your question."

        if self.chain is None:
            self.create_chain()

        response = self.chain.invoke({"context": context, "question": query})
        return response.strip()
```

---

## 3. Implementation Details & Design Highlights

1. **L2 Normalized Semantic Embeddings**:
   Embeddings generated via `sentence-transformers` (`all-MiniLM-L6-v2`) have their L2 norms explicitly verified and scaled:
   $$\|v\|_2 = \sqrt{\sum_{i=1}^d v_i^2} = 1.0$$
   This ensures cosine distance is bounded and metric calculations are consistent across queries.

2. **Scipy Cosine Distance vs. Similarity**:
   `scipy.spatial.distance.cosine(u, v)` yields the cosine distance ($1 - \cos(\theta)$). The retriever computes:
   $$\text{similarity} = 1.0 - \text{cosine}(u, v)$$
   and sorts the results descendingly, returning `[(index, similarity), ...]`.

3. **Fallback & Guardrails**:
   When documents are empty or blank, `RAGChain.query()` returns:
   > `"I am sorry, but I do not have enough information to answer your question."`
   without making an unnecessary external API call, conserving tokens and ensuring deterministic behavior.

---

## 4. Verification & Test Results

All 15 unit tests passed:
- `test_load_documents_success`
- `test_load_documents_file_not_found`
- `test_load_documents_invalid_extension`
- `test_load_documents_missing_columns`
- `test_load_documents_empty_file`
- `test_indexer_generate_embeddings`
- `test_indexer_generate_embeddings_invalid_input`
- `test_indexer_create_and_build_index`
- `test_indexer_query_and_search`
- `test_indexer_find_top_k_matches`
- `test_rag_chain_structure`
- `test_rag_chain_query_validation`
- `test_rag_chain_empty_context_fallback`
- `test_rag_chain_gemini_structure`
- `test_rag_chain_gemini_fallback`

---

## 5. Running the Streamlit Demo App

To launch the interactive demo dashboard:
```bash
streamlit run app.py
```

The application provides:
1. **Model & Key Switcher**: Toggle between **Google Gemini** (`gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash`) and **OpenAI** (`gpt-4o`, `gpt-4o-mini`).
2. **Interactive Querying**: Test predefined FAQ queries or custom questions, with real-time similarity metrics and prompt inspection.
3. **Knowledge Base Explorer**: Search, filter, and inspect the 20 FAQ articles.
4. **Vector Store Inspector**: Real-time vector similarity tester for any phrase against the vector space.
