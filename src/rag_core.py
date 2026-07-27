import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline


@dataclass(frozen=True)
class RAGConfig:
    """Configuration schema for the CrediTrust RAG Pipeline."""
    embedding_model_name: str = "all-MiniLM-L6-v2"
    generator_model_name: str = "distilbert/distilgpt2"
    collection_name: str = "complaints"
    top_k_results: int = 5
    max_new_tokens: int = 60
    temperature: float = 0.3
    repetition_penalty: float = 1.2


class RAGPipeline:
    """Production RAG Engine for Financial Complaint Analysis."""

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        self.config = config or RAGConfig()
        
        # Resolve path dynamically
        self.base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.vector_store_path: str = os.path.join(self.base_dir, "vector_store")
        
        # Load models
        self.embedding_model: SentenceTransformer = SentenceTransformer(
            self.config.embedding_model_name
        )
        self.generator: Any = pipeline(
            "text-generation",
            model=self.config.generator_model_name
        )
        
        # DB Client
        self.client: chromadb.PersistentClient = chromadb.PersistentClient(
            path=self.vector_store_path
        )
        self.collection: chromadb.Collection = self.client.get_collection(
            self.config.collection_name
        )

    def query(
        self, 
        question: str, 
        product_filter: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Processes a user question, retrieves relevant vector context, 
        and generates a grounded answer.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        query_embedding: List[float] = self.embedding_model.encode(question).tolist()
        where_clause: Optional[Dict[str, str]] = {"product": product_filter} if product_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.config.top_k_results,
            where=where_clause
        )

        docs: List[str] = results["documents"][0] if results["documents"] else []
        context: str = "\n".join(docs)

        prompt: str = (
            "You are a financial analyst assistant.\n"
            "Use ONLY the supplied context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:\n"
        )

        response = self.generator(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            repetition_penalty=self.config.repetition_penalty,
            do_sample=True
        )

        full_text: str = response[0]["generated_text"]
        answer: str = full_text.split("Answer:")[-1].strip() if "Answer:" in full_text else full_text.strip()

        return answer, docs


# Global pipeline instance for fast import
_pipeline_instance: Optional[RAGPipeline] = None

def run_rag_pipeline(question: str, product_filter: Optional[str] = None) -> Tuple[str, List[str]]:
    """Functional wrapper for backward compatibility."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance.query(question, product_filter)