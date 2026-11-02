<div align="center">

# 🤖 Support Response Assistant
### Production-Grade Modular RAG Pipeline with Multi-LLM Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C.svg)](https://www.langchain.com/)
[![Sentence Transformers](https://img.shields.io/badge/HuggingFace-MiniLM--L6--v2-yellow.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![CI Status](https://img.shields.io/badge/CI-Passing-success.svg)](https://github.com/shshwtkhr/rag-support-response-assistant/actions)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

An open-source, CPU-optimized **Retrieval-Augmented Generation (RAG)** platform designed for B2B SaaS customer support automation. Integrates local transformer embeddings (`all-MiniLM-L6-v2`) with flexible multi-LLM answer generation (**Google Gemini** & **OpenAI GPT**), dynamic knowledge base editing, and live cosine similarity inspection.

[Features](#-key-features) • [Architecture](#-architecture) • [Quickstart](#-quickstart) • [Interactive Demo UI](#-interactive-demo-walkthrough) • [Technical Deep-Dive](#-technical-deep-dive) • [Testing](#-test-suite)

</div>

---

## 🚀 Key Features

- **⚡ 100% Local CPU Retrieval**: Powered by HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`). Generates 384-dimensional dense vectors with strict unit $L_2$-norm scaling—fast, zero cloud cost, and no API keys required for semantic search.
- **🔄 Multi-LLM Provider Engine**: Seamlessly switch between **Google Gemini** (`gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-1.5-pro`) and **OpenAI** (`gpt-4o`, `gpt-4o-mini`). Includes intelligent automatic model fallback handling.
- **📂 Dynamic Knowledge Base Management**:
  - **CSV Upload**: Ingest and validate any custom FAQ CSV schema (`id`, `title`, `content`) in seconds.
  - **Live In-UI Data Editor**: Insert new FAQs, modify existing answers, or delete articles directly within the web table via `st.data_editor`.
  - **One-Click Rebuild & Restoration**: Instantly re-index custom edits or revert to the factory default 20 SaaS knowledge articles with a single click.
- **🎯 Deterministic Grounding & Guardrails**: LangChain LCEL pipeline engineered to prevent hallucination. Returns structured fallback messages when no relevant context is present without burning LLM tokens.
- **🔍 Vector Store & Similarity Inspector**: Real-time diagnostic tool allowing operators to test any phrase and view live cosine similarity rankings across the entire vector space.

---

## 🏛️ Architecture

```
                    ┌────────────────────────┐
                    │ FAQ CSV Knowledge Base │
                    └───────────┬────────────┘
                                │
                                ▼ (src/document_loader.py)
            ┌───────────────────────────────────────────────┐
            │ Schema Validation (id, title, content)        │
            └───────────────────┬───────────────────────────┘
                                │
                                ▼ (src/indexer.py)
            ┌───────────────────────────────────────────────┐
            │ all-MiniLM-L6-v2 Embedding + L2-Normalization │
            └───────────────────┬───────────────────────────┘
                                │
                                ▼
            ┌───────────────────────────────────────────────┐
            │ Serialized Vector Store (index/faq_index.pkl) │
            └───────────────────▲───────────────────────────┘
                                │ Cosine Similarity Metric
[ User Support Query ] ─────────┤ (1.0 - scipy.distance.cosine)
                                │
                                ▼
            ┌───────────────────────────────────────────────┐
            │ Top-K Retrieved FAQ Context Documents         │
            └───────────────────┬───────────────────────────┘
                                │
                                ▼ (src/rag_chain.py)
            ┌───────────────────────────────────────────────┐
            │ LangChain LCEL Grounded Prompt Pipeline       │
            │   ├─ Google Gemini (2.0 Flash / 2.5 Flash)    │
            │   └─ OpenAI GPT (gpt-4o / gpt-4o-mini)        │
            └───────────────────┬───────────────────────────┘
                                │
                                ▼
            ┌───────────────────────────────────────────────┐
            │ Verified, Context-Grounded Customer Response  │
            └───────────────────────────────────────────────┘
```

---

## 💻 Quickstart

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/shshwtkhr/rag-support-response-assistant.git
cd rag-support-response-assistant
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows (PowerShell):
.\.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Credentials (Optional)
Copy the template configuration file:
```bash
cp .env.example .env
```
Add your credentials to `.env`:
```env
# Google Gemini (Free tier available at https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=your_gemini_api_key_here

# OpenAI (https://platform.openai.com/api-keys)
OPENAI_API_KEY=your_openai_api_key_here
```
*(Note: You can also enter or paste your API keys dynamically in the Streamlit UI).*

### 4. Launch the Interactive Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Interactive Demo Walkthrough

The web dashboard is organized into three intuitive panels:

| Tab | Capability | Description |
| :--- | :--- | :--- |
| **💬 Assistant Demo** | **Q&A & Verification** | Ask customer support questions or click one of the quick presets (e.g. *Password Reset*, *Subscription Plans*, *Refunds*). View the generated answer, latency diagnostics, retrieved document cards with similarity percentages, and the exact filled prompt. |
| **📚 Knowledge Base & Data Editor** | **CRUD & CSV Ingestion** | Upload a new CSV dataset, edit FAQ questions/answers in real time using the interactive grid (`st.data_editor`), download custom tables as CSV, or restore the default dataset. |
| **🔍 Vector Index Inspector** | **Semantic Diagnostics** | Inspect vector dimensions ($d=384$) and test any arbitrary phrase in real time to see its cosine similarity score against all indexed articles. |

---

## 🔬 Technical Deep-Dive

### 1. Vector Normalization & Cosine Similarity
To guarantee consistent ranking across queries of varying lengths, all dense vectors produced by `CustomIndex` are strictly $L_2$-normalized:

$$\|v\|_2 = \sqrt{\sum_{i=1}^{d} v_i^2} = 1.0$$

Cosine similarity between query embedding $\mathbf{q}$ and document embedding $\mathbf{d}$ is computed using `scipy.spatial.distance.cosine`:

$$\text{Similarity}(\mathbf{q}, \mathbf{d}) = 1.0 - \text{CosineDistance}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$

### 2. Context-Grounded LCEL Pipeline
Answer generation uses LangChain Expression Language (LCEL) configured with deterministic zero-temperature sampling and structured fallback rules:
```python
chain = RunnableLambda(_standardize_inputs) | prompt | llm | StrOutputParser()
```
If retrieval yields empty context, `query()` bypasses LLM invocation and immediately returns:
> `"I am sorry, but I do not have enough information to answer your question."`

---

## 🧪 Test Suite

The repository includes a complete test suite with 15 automated unit tests covering dataset loading, vector normalization, pickle serialization, top-k ranking, and LLM provider structures:

```bash
pytest test_rag.py -v
```

**Test Output:**
```text
test_rag.py::test_load_documents_success PASSED
test_rag.py::test_load_documents_file_not_found PASSED
test_rag.py::test_load_documents_invalid_extension PASSED
test_rag.py::test_load_documents_missing_columns PASSED
test_rag.py::test_load_documents_empty_file PASSED
test_rag.py::test_indexer_generate_embeddings PASSED
test_rag.py::test_indexer_generate_embeddings_invalid_input PASSED
test_rag.py::test_indexer_create_and_build_index PASSED
test_rag.py::test_indexer_query_and_search PASSED
test_rag.py::test_indexer_find_top_k_matches PASSED
test_rag.py::test_rag_chain_structure PASSED
test_rag.py::test_rag_chain_query_validation PASSED
test_rag.py::test_rag_chain_empty_context_fallback PASSED
test_rag.py::test_rag_chain_gemini_structure PASSED
test_rag.py::test_rag_chain_gemini_fallback PASSED

============================= 15 passed in 13.41s =============================
```

---

## 📂 Project Structure

```text
rag-support-response-assistant/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/
│       └── ci.yml
├── data/
│   └── faq_dataset.csv
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── indexer.py
│   └── rag_chain.py
├── .env.example
├── .gitignore
├── app.py
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── faq_dataset.csv
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
├── solution.md
└── test_rag.py
```

---

## 🤝 Contributing

Contributions, bug reports, and feature proposals are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before submitting a pull request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
