# app.py
import streamlit as st
from src.rag_core import run_rag_pipeline

st.set_page_config(page_title="CrediTrust Complaint Intelligence", layout="wide")
st.title("🛡️ CrediTrust Intelligent Complaint Analyzer")
st.caption("Turn real customer feedback into actionable product insights in seconds.")

# Sidebar controls for product-specific multi-querying filtering
product_filter = st.sidebar.selectbox(
    "Filter by Product Line",
    ["All Products", "Credit card", "Personal loan", "Savings account", "Money transfer"]
)
filter_val = None if product_filter == "All Products" else product_filter

query = st.text_input("Enter your business query (e.g., 'What are the main billing disputes regarding Credit Cards?'):")

if st.button("Analyze Feedback"):
    if query:
        with st.spinner("Analyzing unstructured historical narratives..."):
            answer, sources = run_rag_pipeline(query, product_filter=filter_val)
            
            st.subheader("💡 Synthesis & Actionable Insights")
            st.write(answer)
            
            st.markdown("---")
            st.subheader("🔍 Retained Evidence Chunks (Transparency Log)")
            for idx, source in enumerate(sources):
                with st.expander(f"Evidence Excerpt #{idx+1}"):
                    st.write(source)
    else:
        st.warning("Please enter a valid prompt to pull analysis.")