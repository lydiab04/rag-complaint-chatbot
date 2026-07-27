from unittest.mock import MagicMock, patch
import pytest
from src.rag_core import RAGConfig, RAGPipeline


def test_rag_config_defaults():
    """Test configuration default values match rag_core defaults."""
    config = RAGConfig()
    assert config.embedding_model_name == "all-MiniLM-L6-v2"
    assert config.generator_model_name == "distilbert/distilgpt2"
    assert config.top_k_results == 5


@patch.object(RAGPipeline, "__init__", lambda self, config=None: None)
def test_empty_question_raises_error():
    """Test that empty or whitespace queries raise a ValueError."""
    pipeline = RAGPipeline()
    with pytest.raises(ValueError, match="Question cannot be empty."):
        pipeline.query("   ")


@patch.object(RAGPipeline, "__init__", lambda self, config=None: None)
def test_rag_query_execution_success():
    """Test successful query execution with mocked collection and generator."""
    pipeline = RAGPipeline()

    # 1. Attach config
    pipeline.config = RAGConfig()

    # 2. Mock embedding_model behavior
    pipeline.embedding_model = MagicMock()
    pipeline.embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

    # 3. Mock Chroma collection query response
    pipeline.collection = MagicMock()
    pipeline.collection.query.return_value = {
        "documents": [["Customer was charged an unexpected late fee on credit card."]],
        "metadatas": [[{"product": "Credit Card"}]]
    }

    # 4. Mock generator behavior
    pipeline.generator = MagicMock()
    pipeline.generator.return_value = [{"generated_text": "The customer experienced unexpected late fee charges."}]

    answer, docs = pipeline.query("Why do customers complain about fees?")

    assert answer is not None
    assert docs is not None