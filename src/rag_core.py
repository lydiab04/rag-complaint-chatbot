import os
import chromadb

from sentence_transformers import SentenceTransformer

from transformers import pipeline



embedding_model = SentenceTransformer(

    "all-MiniLM-L6-v2"

)


generator = pipeline(

    "text-generation",

    model="distilbert/distilgpt2"

)




def run_rag_pipeline(question, product_filter=None):
    # 1. Dynamically resolve the absolute path to the project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")

    # 2. Initialize ChromaDB client using the absolute path
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)

    # 3. Get the existing collection safely
    collection = client.get_collection("complaints")

    query_embedding = embedding_model.encode(question).tolist()

    # FIX: If a product filter is provided, format it for ChromaDB's where clause
    where_clause = {"product": product_filter} if product_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where=where_clause  # <-- Pass the filter rule here!
    )

    docs = results["documents"][0]
    context = "\n".join(docs)

    prompt = f"""
You are a financial analyst assistant.

Use ONLY the supplied context.

Context
{context}

Question
{question}

Answer
"""

    response = generator(
        prompt, 
        max_new_tokens=60, 
        temperature=0.3, 
        repetition_penalty=1.2, 
        do_sample=True
    )
    answer = response[0]["generated_text"]
    
    if "Answer" in answer:
        answer = answer.split("Answer")[-1].strip()

    return answer, docs
    # 1. Dynamically resolve the absolute path to the project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")

    # 2. Initialize ChromaDB client using the absolute path
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)

    # 3. Get the existing collection safely
    collection = client.get_collection("complaints")

    query_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    docs = results["documents"][0]
    context = "\n".join(docs)

    prompt = f"""
You are a financial analyst assistant.

Use ONLY the supplied context.

Context
{context}

Question
{question}

Answer
"""

    # Add temperature and repetition penalty to stop looping patterns
    response = generator(
        prompt,
        max_new_tokens=60,          # Keeps the answer concise for your table
        temperature=0.3,            # Low temperature makes it more deterministic/focused
        repetition_penalty=1.2,     # Heavily penalizes repeating the same sentences
        do_sample=True
    )

    full_text = response[0]["generated_text"]
    
    # Extract only the text generated AFTER your prompt's "Answer" tag
    if "Answer" in full_text:
        answer = full_text.split("Answer")[-1].strip()
    else:
        answer = full_text
        
    return answer, docs

    # 1. Dynamically resolve the absolute path to the root 'vector_store' directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")

    # 2. Connect using the safe absolute path
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    
    # 3. This will now look in the correct folder and find your data!
    collection = client.get_collection("complaints")



    query_embedding = embedding_model.encode(

        question

    ).tolist()



    results = collection.query(

        query_embeddings=[query_embedding],

        n_results=5

    )



    docs = results["documents"][0]



    context = "\n".join(docs)




    prompt = f"""

You are a financial analyst assistant.


Use ONLY the supplied context.



Context

{context}



Question

{question}



Answer


"""



    response = generator(

        prompt,

        max_new_tokens=150

    )



    answer = response[0]["generated_text"]



    return answer,docs