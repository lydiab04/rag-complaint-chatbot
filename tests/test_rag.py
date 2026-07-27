import pytest
from src.rag_core import RAGConfig, RAGPipeline
from src.explainability import RAGExplainer

def test_1_rag_config_defaults():
    config = RAGConfig()
    assert config.embedding_model_name == "all-MiniLM-L6-v2"
    assert config.top_k_results == 3

def test_2_rag_query_empty_input_validation():
    pipeline = RAGPipeline()
    with pytest.raises(ValueError):
        pipeline.query("   ")

def test_3_rag_query_execution_and_filtering():
    pipeline = RAGPipeline()
    res = pipeline.query("unauthorized annual fee", product_filter="Credit Card")
    assert "answer" in res
    assert len(res["sources"]) >= 1
    assert res["product_filter"] == "Credit Card"

def test_4_explainability_keyword_overlap():
    query = "annual fee dispute"
    chunks = ["Consumer raised an annual fee dispute regarding late charges."]
    res = RAGExplainer.calculate_keyword_overlap(query, chunks)
    assert res["overlap_ratio"] > 0.0
    assert "fee" in res["matched_words"]

def test_5_faithfulness_and_token_importance():
    sources = [{"similarity_score": 0.82}, {"similarity_score": 0.88}]
    faith_score = RAGExplainer.audit_source_faithfulness("ans", sources)
    feat_weights = RAGExplainer.explain_feature_importance("disputed fee charges")
    assert faith_score == 0.85
    assert "disputed" in feat_weights