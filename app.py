"""
Streamlit Demo Application for the Support Response Assistant (RAG Pipeline).

Demonstrates the complete end-to-end RAG workflow:
  1. Knowledge Base Loading & Validation (CSV via LoadDocuments)
  2. CPU-Optimized Semantic Vector Indexing (SentenceTransformers + L2 Normalization via CustomIndex)
  3. Real-Time Cosine Similarity Retrieval (Top-k document matching)
  4. Response Generation with Context Grounding (Google Gemini or OpenAI GPT via RAGChain)
"""

import logging
import os
import sys
import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure current directory and src directory are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from src.document_loader import LoadDocuments
    from src.indexer import CustomIndex
    from src.rag_chain import RAGChain
except ImportError:
    from document_loader import LoadDocuments
    from indexer import CustomIndex
    from rag_chain import RAGChain

# Load environment variables (OPENAI_API_KEY, GEMINI_API_KEY, GOOGLE_API_KEY)
load_dotenv()

# Set up logging with dedicated logger
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("rag_app")
if not logger.handlers:
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)


# ==============================================================================
# Helper Functions (API-compatible with starter code and headless tests)
# ==============================================================================

def load_dataset(custom_path: str | None = None) -> pd.DataFrame:
    """Locate and load the FAQ CSV, returning a validated DataFrame."""
    loader = LoadDocuments()
    if custom_path and os.path.exists(custom_path):
        return loader.load_faq_data(custom_path)

    candidates = [
        "data/faq_dataset.csv",
        "faq_dataset.csv",
        os.path.join(current_dir, "data", "faq_dataset.csv"),
        os.path.join(current_dir, "faq_dataset.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return loader.load_faq_data(path)

    return loader.load_faq_data("data/faq_dataset.csv")


def load_or_create_index(df: pd.DataFrame, force_recreate: bool = False):
    """
    Return (index, indexer). Reuses the saved index on disk when it exists,
    otherwise embeds the FAQ DataFrame and saves a new one.
    """
    index_folder = os.path.join(current_dir, "index")
    index_file = "faq_index.pkl"
    index_path = os.path.join(index_folder, index_file)
    indexer = CustomIndex()

    if os.path.exists(index_path) and not force_recreate:
        index = indexer.build_index(index_file, index_folder)
    else:
        index = indexer.create_index(df, index_file, index_folder)
    return index, indexer


def find_relevant_documents(query: str, index: dict, indexer: CustomIndex, k: int = 1):
    """Retrieval step: return top-k FAQ contents most similar to the query."""
    query_embedding = indexer.get_query_embedding(query)
    top_k = indexer.find_top_k_matches(query_embedding, index, k=k)
    documents = [index["contents"][idx] for idx, _ in top_k]
    return documents


def generate_answer(
    query: str,
    documents: list | str,
    model_name: str = "gpt-4o",
    temperature: float = 0.0,
    api_key: str | None = None,
    provider: str = "openai",
) -> str:
    """Generation step: have the LLM answer the query using retrieved documents."""
    rag_chain = RAGChain(
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        provider=provider,
    )
    rag_chain.create_chain()
    return rag_chain.query(query=query, documents=documents)


def fetch_available_gemini_models(api_key: str) -> list[str]:
    """Query Google API to list all Gemini models available for the given key."""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        discovered = []
        for m in client.models.list():
            clean_name = m.name.replace("models/", "")
            if "gemini" in clean_name.lower() and "embed" not in clean_name.lower():
                discovered.append(clean_name)
        return discovered
    except Exception as e:  # noqa: BLE001
        logger.info("Could not list models: %s", e)
        return []


# ==============================================================================
# Streamlit Interactive Dashboard
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Support Response Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state variables
    if "df" not in st.session_state:
        st.session_state.df = None
    if "index" not in st.session_state:
        st.session_state.index = None
    if "indexer" not in st.session_state:
        st.session_state.indexer = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "discovered_gemini_models" not in st.session_state:
        st.session_state.discovered_gemini_models = []

    # Auto-load dataset and index on startup if available
    if st.session_state.df is None:
        try:
            st.session_state.df = load_dataset()
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
            logger.info("Default dataset auto-load deferred: %s", e)

    if st.session_state.df is not None and st.session_state.index is None:
        try:
            st.session_state.index, st.session_state.indexer = load_or_create_index(st.session_state.df)
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as e:
            logger.info("Default index auto-load deferred: %s", e)

    # --------------------------------------------------------------------------
    # Sidebar: Provider & Model Configuration
    # --------------------------------------------------------------------------
    st.sidebar.title("⚙️ RAG Configuration")

    st.sidebar.subheader("1. LLM Provider")
    provider_option = st.sidebar.radio(
        "Select Provider",
        options=["Google Gemini", "OpenAI"],
        index=0,
        help="Choose which Large Language Model provider generates the final answers.",
    )
    provider_key = "gemini" if provider_option == "Google Gemini" else "openai"

    st.sidebar.subheader("2. Model & API Key")
    if provider_key == "gemini":
        detected_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        api_key_input = st.sidebar.text_input(
            "Gemini API Key",
            value=detected_key,
            type="password",
            help="Get a key from https://aistudio.google.com/app/apikey or set GEMINI_API_KEY in .env",
        )

        col_key1, col_key2 = st.sidebar.columns([1, 1])
        with col_key1:
            if not api_key_input:
                st.info("💡 [Get Free Key](https://aistudio.google.com/app/apikey)")
            else:
                st.success("✅ Key ready")

        with col_key2:
            if api_key_input and st.button("🔍 Discover", help="List models supported by your API key"):
                with st.spinner("Checking key..."):
                    found = fetch_available_gemini_models(api_key_input.strip())
                    if found:
                        st.session_state.discovered_gemini_models = found
                        st.toast(f"Found {len(found)} available models!")
                    else:
                        st.sidebar.error("Could not fetch models. Check key.")

        # Default model list prioritizing active production models
        base_models = [
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro",
            "gemini-2.0-pro",
            "Custom...",
        ]

        # If user discovered models via the API, merge them
        if st.session_state.discovered_gemini_models:
            model_options = list(dict.fromkeys(st.session_state.discovered_gemini_models + base_models))
        else:
            model_options = base_models

        selected_model_choice = st.sidebar.selectbox(
            "Gemini Model",
            options=model_options,
            index=0,
            help="gemini-2.0-flash and gemini-2.5-flash are the active generation models on Google AI Studio.",
        )

        if selected_model_choice == "Custom...":
            model_name = st.sidebar.text_input("Custom Model Name", value="gemini-2.0-flash")
        else:
            model_name = selected_model_choice

    else:
        model_name = st.sidebar.selectbox(
            "OpenAI Model",
            options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            index=0,
            help="gpt-4o is the default high-precision model required by the project specification.",
        )
        detected_key = os.environ.get("OPENAI_API_KEY") or ""
        api_key_input = st.sidebar.text_input(
            "OpenAI API Key",
            value=detected_key,
            type="password",
            help="Get a key from https://platform.openai.com/api-keys or set OPENAI_API_KEY in .env",
        )
        if not api_key_input:
            st.sidebar.warning("⚠️ Enter an OpenAI key or add OPENAI_API_KEY to your `.env` file.")
        else:
            st.sidebar.success("✅ OpenAI API Key detected.")

    st.sidebar.subheader("3. Retrieval Parameters")
    top_k = st.sidebar.slider(
        "Top-K Retrieved FAQs",
        min_value=1,
        max_value=5,
        value=2,
        help="Number of most semantically relevant documents retrieved as context for the LLM.",
    )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="0.0 provides strict, deterministic, grounded answers based directly on context.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("4. Vector Index Status")
    if st.session_state.index is not None:
        num_docs = len(st.session_state.index.get("ids", []))
        st.sidebar.success(f"🟢 Index Active: {num_docs} FAQs stored (384-d MiniLM vectors)")
    else:
        st.sidebar.warning("🟡 Index Not Loaded")

    if st.sidebar.button("🔄 Rebuild Vector Index", help="Re-embeds all records and overwrites pickle") and st.session_state.df is not None:
        with st.sidebar.status("Re-embedding documents with all-MiniLM-L6-v2..."):
            st.session_state.index, st.session_state.indexer = load_or_create_index(
                st.session_state.df, force_recreate=True
            )
        st.sidebar.success("Index rebuilt successfully!")
        st.rerun()

    # --------------------------------------------------------------------------
    # Main Window: Title & Navigation Tabs
    # --------------------------------------------------------------------------
    st.title("🤖 Support Response Assistant")
    st.markdown(
        "**AI Customer Support RAG Pipeline** | "
        "`sentence-transformers/all-MiniLM-L6-v2` (Local CPU) + **Google Gemini / OpenAI GPT**"
    )

    tab_chat, tab_dataset, tab_index = st.tabs(
        ["💬 Assistant Demo", "📚 Knowledge Base Explorer", "🔍 Vector Index Inspector"]
    )

    # ==========================================================================
    # TAB 1: Support Assistant Demo
    # ==========================================================================
    with tab_chat:
        col_main, col_presets = st.columns([2.2, 1])

        with col_presets:
            st.markdown("#### ⚡ Quick Preset Queries")
            preset_queries = [
                "How do I reset my password?",
                "What are the subscription plans?",
                "How to delete my account?",
                "Explain data privacy policies.",
                "How to enable two-factor authentication?",
                "What should I do if the app crashes?",
                "Can I get a refund on my purchase?",
                "What is your company phone number in Tokyo?",  # Out-of-domain test
            ]

            clicked_preset = None
            for p_query in preset_queries:
                if st.button(p_query, key=f"btn_p_{p_query}", use_container_width=True):
                    clicked_preset = p_query

        with col_main:
            st.markdown("#### ❓ Ask a Support Question")
            user_input = st.text_input(
                "Enter your customer query below:",
                value=clicked_preset if clicked_preset else st.session_state.last_query,
                placeholder="e.g. How do I reset my password?",
            )

            col_btn1, _ = st.columns([1, 4])
            with col_btn1:
                submit_button = st.button("🚀 Ask Assistant", type="primary", use_container_width=True)

            if submit_button or clicked_preset:
                query_to_run = user_input.strip() if user_input else clicked_preset
                if not query_to_run:
                    st.warning("Please enter or select a query.")
                elif st.session_state.index is None or st.session_state.indexer is None:
                    st.error("Please load the dataset and vector index first from Tab 2 or Tab 3.")
                else:
                    st.session_state.last_query = query_to_run

                    # --- Step A: Retrieval ---
                    t0 = time.perf_counter()
                    q_emb = st.session_state.indexer.get_query_embedding(query_to_run)
                    top_matches = st.session_state.indexer.find_top_k_matches(
                        q_emb, st.session_state.index, k=top_k
                    )
                    t_retrieval = (time.perf_counter() - t0) * 1000

                    retrieved_docs = [st.session_state.index["contents"][idx] for idx, _ in top_matches]
                    retrieved_titles = [st.session_state.index["titles"][idx] for idx, _ in top_matches]
                    similarity_scores = [score for _, score in top_matches]

                    # --- Step B: Generation ---
                    t1 = time.perf_counter()
                    with st.spinner(f"Generating answer using {provider_option} ({model_name})..."):
                        try:
                            active_key = api_key_input.strip() if api_key_input else None
                            rag = RAGChain(
                                model_name=model_name,
                                temperature=temperature,
                                api_key=active_key,
                                provider=provider_key,
                            )
                            rag.create_chain()
                            response_text = rag.query(query=query_to_run, documents=retrieved_docs)
                        except Exception as e:
                            err_str = str(e)
                            if "404" in err_str or "NOT_FOUND" in err_str:
                                response_text = (
                                    f"⚠️ **Model `{model_name}` was not found (404 NOT_FOUND).**\n\n"
                                    "Google AI Studio has migrated default flash endpoints to newer versions. "
                                    "**Solution:** In the sidebar under 'Gemini Model', select **`gemini-2.0-flash`** "
                                    "or click **🔍 Discover** to list models active for your key."
                                )
                            elif "INVALID_ARGUMENT" in err_str or "API key not valid" in err_str:
                                response_text = (
                                    "⚠️ **Invalid API Key.** Please verify your Google Gemini or OpenAI API key in the sidebar."
                                )
                            else:
                                response_text = (
                                    f"⚠️ **Error invoking LLM:** {e}\n\n"
                                    f"Please verify that your {provider_option} API key is valid."
                                )
                            logger.exception("LLM generation error")
                    t_gen = (time.perf_counter() - t1) * 1000

                    # Save last result
                    st.session_state.last_result = {
                        "query": query_to_run,
                        "response": response_text,
                        "matches": top_matches,
                        "docs": retrieved_docs,
                        "titles": retrieved_titles,
                        "scores": similarity_scores,
                        "t_retrieval": t_retrieval,
                        "t_gen": t_gen,
                        "provider": provider_option,
                        "model": model_name,
                    }

            # Display Result if available
            if st.session_state.last_result:
                res = st.session_state.last_result

                st.markdown("### 💬 Assistant Response")
                st.info(f"**Answer:**\n\n{res['response']}")

                st.caption(
                    f"⏱️ Retrieval Latency: **{res['t_retrieval']:.1f} ms** | "
                    f"LLM Generation: **{res['t_gen']:.1f} ms** | "
                    f"Engine: **{res['provider']} ({res['model']})**"
                )

                # Expandable Retrieved Context
                with st.expander("📄 View Retrieved Context Documents & Cosine Similarity", expanded=True):
                    for rank, (title, content, score) in enumerate(
                        zip(res["titles"], res["docs"], res["scores"]), start=1
                    ):
                        score_pct = score * 100
                        col_card1, col_card2 = st.columns([3, 1])
                        with col_card1:
                            st.markdown(f"**Rank #{rank}: {title}**")
                            st.write(content)
                        with col_card2:
                            st.metric(label="Cosine Similarity", value=f"{score_pct:.1f}%")
                        st.divider()

                # Expandable RAG Pipeline Inspector
                with st.expander("🛠️ RAG Pipeline Inspector (Full Prompt & Parameters)", expanded=False):
                    context_str = "\n".join(res["docs"])
                    sample_prompt = (
                        "You are a helpful customer support assistant for a B2B SaaS platform.\n"
                        "Answer the question concisely and accurately using ONLY the provided context.\n"
                        "If the answer cannot be determined from the context, or if no relevant context is found, "
                        "respond with: 'I am sorry, but I do not have enough information to answer your question.'\n\n"
                        f"Context:\n{context_str}\n\n"
                        f"Question:\n{res['query']}\n\n"
                        "Answer:"
                    )
                    st.code(sample_prompt, language="markdown")

    # ==========================================================================
    # TAB 2: Knowledge Base Explorer
    # ==========================================================================
    with tab_dataset:
        st.markdown("### 📚 Product Documentation & FAQ Dataset")
        st.markdown(
            "This table contains the SaaS knowledge base loaded and validated by `LoadDocuments`. "
            "Required columns: `id`, `title`, and `content`."
        )

        col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
        if st.session_state.df is not None:
            with col_d1:
                st.metric("Total FAQ Articles", len(st.session_state.df))
            with col_d2:
                avg_len = int(st.session_state.df["content"].str.len().mean())
                st.metric("Avg Article Length", f"{avg_len} chars")
            with col_d3:
                st.metric("Required Columns", "id, title, content ✅")

            search_filter = st.text_input("🔍 Filter FAQs by keyword:", placeholder="e.g. password, billing")
            filtered_df = st.session_state.df
            if search_filter:
                filtered_df = st.session_state.df[
                    st.session_state.df["title"].str.contains(search_filter, case=False, na=False)
                    | st.session_state.df["content"].str.contains(search_filter, case=False, na=False)
                ]

            st.dataframe(filtered_df, use_container_width=True, height=400)
        else:
            st.warning("Dataset not loaded.")
            if st.button("📂 Load Dataset Now"):
                st.session_state.df = load_dataset()
                st.rerun()

    # ==========================================================================
    # TAB 3: Vector Index Inspector
    # ==========================================================================
    with tab_index:
        st.markdown("### 🔍 Semantic Vector Store & Embeddings Inspector")
        st.markdown(
            "The vector index is powered by `sentence-transformers/all-MiniLM-L6-v2`, generating "
            "**384-dimensional dense vectors** with **L2-normalization**."
        )

        if st.session_state.index is not None:
            emb_matrix = st.session_state.index.get("embeddings")
            c_idx1, c_idx2, c_idx3 = st.columns(3)
            with c_idx1:
                st.metric("Indexed Documents", len(st.session_state.index["ids"]))
            with c_idx2:
                dim = emb_matrix.shape[1] if hasattr(emb_matrix, "shape") else 384
                st.metric("Embedding Dimensions", dim)
            with c_idx3:
                st.metric("Storage Format", "Pickle (.pkl)")

            st.markdown("#### 🧪 Standalone Cosine Similarity Tester")
            st.caption("Test how any custom query maps semantically against all indexed FAQs in real time:")

            test_query = st.text_input(
                "Enter test phrase:",
                value="I forgot how to log in to my profile",
                key="tester_query_input",
            )
            if test_query and st.session_state.indexer:
                q_vec = st.session_state.indexer.get_query_embedding(test_query)
                matches = st.session_state.indexer.find_top_k_matches(
                    q_vec, st.session_state.index, k=len(st.session_state.index["ids"])
                )

                ranking_data = []
                for rank, (idx, score) in enumerate(matches, start=1):
                    ranking_data.append(
                        {
                            "Rank": rank,
                            "Title": st.session_state.index["titles"][idx],
                            "Similarity (%)": round(score * 100, 2),
                            "Content Preview": st.session_state.index["contents"][idx][:80] + "...",
                        }
                    )
                st.dataframe(pd.DataFrame(ranking_data), use_container_width=True, height=350)
        else:
            st.info("Load dataset to build or inspect the vector store.")


if __name__ == "__main__":
    main()
