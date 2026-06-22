# src/rag_core.py
import chromadb
from sentence_transformers import SentenceTransformer
import os
from groq import Groq # Example using an ultra-fast free/cheap endpoint

def run_rag_pipeline(query, product_filter=None):
    # Connect to the full pre-built ChromaDB
    chroma_client = chromadb.PersistentClient(path="vector_store/full_store")
    collection = chroma_client.get_collection(name="complaints_full")
    
    # Embed Query
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    query_vector = model.encode(query).tolist()
    
    # Query Database with optional product filtering metadata
    where_clause = {"product_category": product_filter} if product_filter else None
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=5,
        where=where_clause
    )
    
    context_chunks = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    # Synthesize context
    context_str = "\n\n".join([f"[Source ID: {m['complaint_id']} | Product: {m['product_category']}]: {doc}" 
                               for doc, m in zip(context_chunks, metadatas)])
    
    # Prompt Template Strategy
    prompt = f"""You are a senior financial analyst assistant for CrediTrust Financial. 
Your task is to answer strategic questions about customer complaints using only the provided context excerpts.

Context:
{context_str}

Question: {query}

Answer the question thoroughly, highlighting trends and citing source IDs where appropriate. If the context does not contain the answer, state transparently that you do not have enough information.
Answer:"""

    # Call LLM
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8th",
    )
    
    return response.choices[0].message.content, context_chunks