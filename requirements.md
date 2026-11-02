# Support Response Assistant - Requirements

## Scenario
You are a backend developer at a product company that runs a B2B SaaS platform. Your team is building a RAG (Retrieval-Augmented Generation) system to automate customer support by answering user queries using product documentation and FAQs.

The pipeline is already wired up — chunking, indexing, retrieval, and response generation with an LLM — but it runs on placeholder embeddings that return junk results. To fix this, you decide to implement proper semantic embeddings using HuggingFace transformer models like MiniLM or DistilBERT. Your goal is to integrate a CPU-optimized embedding workflow that generates high-quality vector representations for documents and queries, boosting retrieval accuracy while keeping the system scalable for production workloads.

---

## Task
Implement a modular RAG pipeline to generate high-quality vector representations with the following components:

### 1. Document Loader (`document_loader.py`)
Create a class `LoadDocuments` to load FAQ data from a CSV file.
- Validate file structure and check required columns: `id`, `title`, and `content`.
- Raise appropriate exceptions for invalid or empty datasets:
  - `FileNotFoundError`: If the file does not exist.
  - `ValueError`: If the file is not a CSV, missing columns, or is empty.
- Return a clean pandas DataFrame for further processing.

### 2. Embedding Index and Retriever (`indexer.py`)
Create a class `CustomIndex` to handle semantic embedding, indexing, and retrieval:
- Use `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) to create L2-normalized embeddings.
- Store FAQ entries and their embeddings in memory.
- Support saving/loading index using Pickle.
- Embed user queries and compute cosine similarity to retrieve top-k FAQs.
- Methods to implement:
  - `generate_embeddings(texts: list, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray`:
    - Generates L2-normalized embeddings for a non-empty list of input strings.
    - Raises `ValueError` if input is invalid or embedding generation fails.
  - `create_index(df: pd.DataFrame, index_file_name: str, index_folder_name: str, model_name: str = 'all-MiniLM-L6-v2') -> dict`:
    - Builds and saves an index from a DataFrame containing document titles and content.
    - Persists index to a pickle file in the specified directory.
    - Returns a dictionary with keys: `'embeddings'`, `'ids'`, `'titles'`, `'contents'`.
    - Raises `ValueError` if input is not a DataFrame or internal validation fails.
  - `build_index(index_file_name: str, index_folder_name: str) -> dict`:
    - Loads an existing index from disk and validates its structure.
    - Returns dictionary with keys: `'embeddings'`, `'ids'`, `'titles'`, `'contents'`.
    - Raises `FileNotFoundError` if the file does not exist.
    - Raises `ValueError` if the file lacks the required index structure.
  - `get_query_embedding(query: str, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray`:
    - Encodes a query string into a single L2-normalized vector embedding.
    - Raises `ValueError` if query is not a valid non-empty string or embedding fails.
  - `find_top_k_matches(query_embedding: np.ndarray, index: dict, k: int = 3) -> list`:
    - Finds the top-k most semantically similar documents in the index based on cosine similarity.
    - Returns a list of tuples `(index, similarity_score)` sorted in descending order of similarity.
    - Raises `ValueError` if `k` is not positive or if input types/structures are invalid.
    - Raises `RuntimeError` if similarity computation fails unexpectedly.

### 3. Generator (`rag_chain.py`)
Create a `RAGChain` class using OpenAI's `gpt-4o` model to generate final answers:
- Connect prompt templates, a language model, and an output parser (`StrOutputParser`) to form a LangChain pipeline (`self.chain`).
- Accept the top-k relevant FAQ entries and a user query.
- Generate a concise and context-aware answer.
- If no relevant context is found (or empty context provided), return a default fallback answer.
- Methods to implement:
  - `__init__(model_name: str = "gpt-4o", temperature: float = 0)`:
    - Initializes the LLM (`ChatOpenAI`) with the specified model and temperature.
  - `create_chain()`:
    - Creates and configures the LangChain runnable sequence: `prompt | llm | output_parser`.
  - `query(query: str, documents: Union[str, list]) -> str`:
    - Validates inputs (raises `ValueError` on invalid inputs).
    - Formats context from documents and invokes the chain to generate an answer.
    - Returns fallback answer if no relevant context is present.

---

## Deliverables

| File | Description |
| --- | --- |
| `document_loader.py` | Loads and validates FAQ data from a CSV file, ensuring proper structure and data integrity. |
| `indexer.py` | Embeds documents, stores them in a vector store, and retrieves the most relevant documents based on similarity. |
| `rag_chain.py` | Generates the final answer by combining the user query with the retrieved documents and generating a response using OpenAI's GPT-4. |

---

## Sample Data & Interactions

### Sample Document (`faq_dataset.csv`)
```csv
id,title,content
1,Subscription Plans,The subscription plans are monthly, quarterly, and annual options.
2,Cancel Subscription,To cancel your subscription, visit the billing section in your account.
3,Refund Policy,Refunds are available within 14 days of purchase under specific conditions.
```

### Sample Input
```text
How do I reset my password?
```

### Sample Output
```text
To reset your password, go to the login page and click on 'Forgot Password'.
```
