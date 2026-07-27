import streamlit as st
import time
from src.rag_core import RAGPipeline, RAGConfig
from src.explainability import RAGExplainer

st.set_page_config(
    page_title="CrediTrust - CFPB RAG Intelligence",
    page_icon="🏦",
    layout="wide"
)

# Title & Header
st.title("🏦 CrediTrust Financial: CFPB Complaint RAG Assistant")
st.caption("Production-grade Retrieval-Augmented Generation for Regulatory Audits & Consumer Insights")

# Sidebar Configuration
st.sidebar.header("⚙️ Query Configuration")
product_filter = st.sidebar.selectbox(
    "Filter by Financial Product:",
    ["All Products", "Credit Card", "Mortgage", "Student Loan", "Debt Collection", "Checking Account"]
)
top_k = st.sidebar.slider("Retrieved Source Chunks (Top K):", min_value=1, max_value=5, value=3)

# Pipeline Instantiation
@st.cache_resource
def load_pipeline():
    config = RAGConfig(top_k_results=top_k)
    return RAGPipeline(config)

pipeline = load_pipeline()

# Main Query Section
st.subheader("🔎 Search & Compliance Audit")
user_query = st.text_input("Enter compliance query or consumer issue topic:", placeholder="e.g., Unauthorized annual fees and cancellation disputes")

if st.button("Run Audit Query", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a valid query prompt.")
    else:
        with st.spinner("Retrieving vector embeddings & generating answer..."):
            selected_filter = None if product_filter == "All Products" else product_filter
            result = pipeline.query(user_query, product_filter=selected_filter)
            
            # Explainability Metrics
            source_texts = [s["text"] for s in result["sources"]]
            explain_metrics = RAGExplainer.calculate_keyword_overlap(user_query, source_texts)
            faithfulness_score = RAGExplainer.audit_source_faithfulness(result["answer"], result["sources"])

        # Top Row Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Execution Latency", f"{result['metrics']['execution_time_seconds']}s")
        col2.metric("Retrieved Sources", f"{result['metrics']['retrieved_chunks']} chunks")
        col3.metric("Grounding Confidence", f"{faithfulness_score * 100}%")

        # Answer Section
        st.markdown("### 📝 Grounded Response")
        st.info(result["answer"])

        # Source Context Expanders
        st.markdown("### 📑 Verified Source Context (CFPB Complaint Excerpts)")
        for idx, src in enumerate(result["sources"], start=1):
            with st.expander(f"Source #{idx} - Complaint ID: {src['id']} (Similarity: {src['similarity_score']})"):
                st.write(f"**Product:** {src['product']}")
                st.write(f"**Narrative:** {src['text']}")

        # Audit Explainability Tab
        with st.expander("🔬 View Model Explainability & Keyword Attribution"):
            st.write(f"**Keyword Overlap Ratio:** {explain_metrics['overlap_ratio']}")
            st.write(f"**Matched Audit Keywords:** {', '.join(explain_metrics['matched_words']) if explain_metrics['matched_words'] else 'None'}")