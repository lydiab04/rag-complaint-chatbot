import pytest
from src.rag_core import RAGConfig, RAGPipeline

def test_rag_config_defaults():
    """Test configuration default values."""
    config = RAGConfig()
    assert config.embedding_model_name == "all-MiniLM-L6-v2"
    assert config.top_k_results == 5
    assert config.temperature == 0.3

def test_empty_question_raises_error():
    """Test that empty queries raise ValueError."""
    pipeline = RAGPipeline()
    with pytest.raises(ValueError, match="Question cannot be empty."):
        pipeline.query("   ")

def test_rag_query_execution():
    """Test valid query response type and structure."""
    pipeline = RAGPipeline()
    answer, docs = pipeline.query("Why do customers complain about credit card fees?")
    assert isinstance(answer, str)
    assert isinstance(docs, list)
    assert len(docs) > 0