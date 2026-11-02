import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest

try:
    from src.document_loader import LoadDocuments
    from src.indexer import CustomIndex
    from src.rag_chain import RAGChain
except ImportError:
    from document_loader import LoadDocuments
    from indexer import CustomIndex
    from rag_chain import RAGChain


def test_load_documents_success():
    loader = LoadDocuments()
    df = loader.load_faq_data("faq_dataset.csv")
    assert isinstance(df, pd.DataFrame)
    assert set(["id", "title", "content"]).issubset(df.columns)
    assert len(df) == 20


def test_load_documents_file_not_found():
    loader = LoadDocuments()
    with pytest.raises(FileNotFoundError):
        loader.load_faq_data("non_existent_file.csv")


def test_load_documents_invalid_extension():
    loader = LoadDocuments()
    with pytest.raises(ValueError):
        loader.load_faq_data("app.py")


def test_load_documents_missing_columns():
    loader = LoadDocuments()
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write("id,wrong_col\n1,test\n")
        temp_path = f.name
    try:
        with pytest.raises(ValueError):
            loader.load_faq_data(temp_path)
    finally:
        os.remove(temp_path)


def test_load_documents_empty_file():
    loader = LoadDocuments()
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        temp_path = f.name
    try:
        with pytest.raises(ValueError):
            loader.load_faq_data(temp_path)
    finally:
        os.remove(temp_path)


def test_indexer_generate_embeddings():
    indexer = CustomIndex()
    texts = ["How do I reset my password?", "What are the subscription plans?"]
    embeddings = indexer.generate_embeddings(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    # Check L2 normalization
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        assert np.isclose(norm, 1.0, atol=1e-3)


def test_indexer_generate_embeddings_invalid_input():
    indexer = CustomIndex()
    with pytest.raises(ValueError):
        indexer.generate_embeddings([])

    with pytest.raises(ValueError):
        indexer.generate_embeddings(["valid", 123])


def test_indexer_create_and_build_index():
    indexer = CustomIndex()
    df = pd.DataFrame({
        "id": [1, 2],
        "title": ["Resetting Password", "Subscription Plans"],
        "content": ["Go to login and click forgot password", "Monthly and annual options"]
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        index_file = "test_index.pkl"
        created = indexer.create_index(df, index_file, tmpdir)
        assert set(created.keys()) == {"embeddings", "ids", "titles", "contents"}
        assert len(created["ids"]) == 2

        # Build / Load index
        loaded = indexer.build_index(index_file, tmpdir)
        assert set(loaded.keys()) == {"embeddings", "ids", "titles", "contents"}
        assert len(loaded["ids"]) == 2
        assert loaded["titles"] == ["Resetting Password", "Subscription Plans"]


def test_indexer_query_and_search():
    indexer = CustomIndex()
    loader = LoadDocuments()
    df = loader.load_faq_data("faq_dataset.csv")

    with tempfile.TemporaryDirectory() as tmpdir:
        index = indexer.create_index(df, "faq_index.pkl", tmpdir)
        q_emb = indexer.get_query_embedding("How do I reset my password?")
        assert isinstance(q_emb, np.ndarray)
        assert np.isclose(np.linalg.norm(q_emb), 1.0, atol=1e-3)

        top_matches = indexer.find_top_k_matches(q_emb, index, k=3)
        assert len(top_matches) == 3
        # Match 0 should be top ranked: index 0 is "Resetting Password"
        top_idx, top_score = top_matches[0]
        assert top_idx == 0
        assert "password" in index["contents"][top_idx].lower()


def test_indexer_find_top_k_invalid_k():
    indexer = CustomIndex()
    with pytest.raises(ValueError):
        indexer.find_top_k_matches(np.zeros((10,)), {}, k=0)


def test_rag_chain_structure():
    rag = RAGChain()
    chain = rag.create_chain()
    assert chain is not None


def test_rag_chain_query_validation():
    rag = RAGChain()
    with pytest.raises(ValueError):
        rag.query("", ["Some context"])

    with pytest.raises(ValueError):
        rag.query("Valid query", 123)


def test_rag_chain_empty_context_fallback():
    rag = RAGChain()
    res = rag.query("How do I reset my password?", [])
    assert "not have enough information" in res.lower()


def test_rag_chain_gemini_structure():
    rag = RAGChain(model_name="gemini-1.5-flash", provider="gemini")
    chain = rag.create_chain()
    assert chain is not None


def test_rag_chain_gemini_fallback():
    rag = RAGChain(model_name="gemini-1.5-flash", provider="gemini")
    res = rag.query("How do I reset my password?", [])
    assert "not have enough information" in res.lower()

