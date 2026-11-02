# Support Response Assistant - RAG Pipeline

A production-grade, modular Retrieval-Augmented Generation (RAG) system built for customer support automation. 

Combines **CPU-optimized semantic embeddings** via `sentence-transformers` (`all-MiniLM-L6-v2`) with multi-provider LLM answer generation (**Google Gemini** and **OpenAI GPT**).

---

## Architecture Overview

```
[ FAQ CSV Dataset ] 
        │
        ▼ (src/document_loader.py)
[ DataFrame Validation & Loading ]
        │
        ▼ (src/indexer.py)
[ SentenceTransformer: all-MiniLM-L6-v2 ] ──► [ L2-Normalization ]
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
[ User Query ] ───────────────┘       (src/rag_chain.py: Gemini 2.0 Flash / GPT-4o)
```

---

## Features

- **Document Loader (`src/document_loader.py`)**: Robust CSV loader with validation for missing files, empty files, and required schema (`id`, `title`, `content`).
- **Semantic Vector Index (`src/indexer.py`)**: CPU-friendly `all-MiniLM-L6-v2` dense embeddings with unit L2-norm scaling, pickle serialization, and cosine similarity ranking.
- **Context-Aware Generator (`src/rag_chain.py`)**: Multi-provider support for **Google Gemini** (`gemini-2.0-flash`, `gemini-2.5-flash`) and **OpenAI** (`gpt-4o`, `gpt-4o-mini`). Includes deterministic fallback handling when no relevant context is found.
- **Interactive Streamlit Demo (`app.py`)**: Full-featured UI featuring:
  - Multi-LLM provider switcher (Gemini & OpenAI) with automatic model fallback
  - Dynamic dataset management: upload custom CSV files or edit data directly in the UI table (add/modify/delete rows)
  - One-click default dataset restoration
  - Interactive Q&A with predefined customer query presets
  - Knowledge Base Explorer with live search/filtering and CSV export
  - Real-time Cosine Similarity & Vector Inspector

---

## Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/rag-support-response-assistant.git
cd rag-support-response-assistant
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt # or uv pip install pandas numpy sentence-transformers scipy langchain-openai langchain-google-genai langchain-core streamlit python-dotenv pytest
```

### 3. Configure API Keys (Optional)
Copy `.env.example` to `.env` and configure your preferred provider:
```bash
cp .env.example .env
```
*(You can also input your API key directly in the Streamlit UI).*

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```

### 5. Run Test Suite
```bash
pytest test_rag.py -v
```

---

## License
MIT License
