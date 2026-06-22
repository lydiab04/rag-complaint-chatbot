from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

def build_vector_store(sample_df):
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    chroma_client = chromadb.PersistentClient(path="vector_store/")
    collection = chroma_client.get_or_create_collection(name="complaints_sample")
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    for idx, row in sample_df.iterrows():
        chunks = text_splitter.split_text(row['cleaned_narrative'])
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{row['complaint_id']}_chunk_{i}"
            embedding = model.encode(chunk).tolist()
            
            # Pack metadata exactly as required by KPIs
            metadata = {
                "complaint_id": str(row['complaint_id']),
                "product_category": str(row['product']),
                "issue": str(row['issue']),
                "date_received": str(row['date_received'])
            }
            
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[chunk_id]
            )
    print("Vector Store successfully persisted in vector_store/")