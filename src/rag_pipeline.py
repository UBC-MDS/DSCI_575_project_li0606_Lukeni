import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.bm25 import BM25Retriever
from src.hybrid import HybridRetriever
from src.semantic import SemanticRetriever

load_dotenv()

SYSTEM_PROMPT_V1 = """
You are a helpful Amazon shopping assistant.
Answer the question using ONLY the provided context from Amazon product reviews and metadata.
Do not make up product details that are not in the context.
When possible, mention the product title or ASIN.
"""

SYSTEM_PROMPT_V2 = """
You are a careful product recommendation assistant.
Use only the retrieved Amazon review context below.
If the context is insufficient, say so clearly.
Keep the answer concise, helpful, and grounded in the retrieved evidence.
"""

SYSTEM_PROMPT_V3 = """
You are an Amazon reviews analyst.
Answer the user only from the retrieved reviews and product metadata.
Prefer direct evidence from the retrieved context, and avoid unsupported claims.
"""

@dataclass
class RetrievedDoc:
    product_title: str
    review_title: str
    text: str
    rating: float
    parent_asin: str
    score: float


class SemanticRAGPipeline:
    def __init__(
        self,
        corpus_path: str | None = None,
        faiss_index_path: str | None = None,
        metadata_path: str | None = None,
        *,
        semantic_retriever: SemanticRetriever | None = None,
        model_name: str = None,
        top_k: int = 5,
        system_prompt: str = SYSTEM_PROMPT_V1,
    ) -> None:
        self.top_k = top_k
        self.system_prompt = system_prompt
        self.model_name = model_name or os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

        if semantic_retriever is not None:
            self.retriever = semantic_retriever
        else:
            if not faiss_index_path or not metadata_path:
                raise ValueError(
                    "Provide either `semantic_retriever=...` or both `faiss_index_path` and `metadata_path`."
                )
            self.retriever = SemanticRetriever.load_saved(
                index_path=faiss_index_path,
                metadata_path=metadata_path,
            )

        self.llm = ChatGroq(
            model=self.model_name,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def retrieve(self, query: str):
        return self.retriever.search(query, top_k=self.top_k)

    def build_context(self, docs) -> str:
        blocks = []
        for _, doc in docs.iterrows():
            block = (
                f"ASIN: {doc.get('parent_asin', 'N/A')}\n"
                f"Title: {doc.get('product_title', '')}\n"
                f"Rating: {doc.get('rating', 'N/A')}\n"
                f"Review title: {doc.get('review_title', '')}\n"
                f"Review text: {doc.get('text', '')}\n"
            )
            blocks.append(block)
        return "\n\n".join(blocks)

    def build_prompt(self, query: str, context: str, system_prompt: str | None = None) -> str:
        sp = system_prompt if system_prompt is not None else self.system_prompt
        return f"""{sp}

Context:
{context}

Question:
{query}

Answer using only the context above:
"""

    def generate(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content

    def answer(self, query: str, system_prompt: str | None = None):
        """
        If ``system_prompt`` is None, uses ``self.system_prompt`` (default ``SYSTEM_PROMPT_V1``
        unless the pipeline was constructed with another default).
        Pass ``SYSTEM_PROMPT_V1/V2/V3`` to switch behavior without rebuilding the pipeline.
        """
        docs = self.retrieve(query)
        context = self.build_context(docs)
        prompt = self.build_prompt(query, context, system_prompt=system_prompt)
        answer = self.generate(prompt)

        return {
            "query": query,
            "answer": answer,
            "docs": docs,
            "context": context,
            "prompt": prompt,
        }
    


class HybridRAGPipeline(SemanticRAGPipeline):
    def __init__(
        self,
        corpus_path: str | None = None,
        bm25_index_path: str | None = None,
        bm25_tokens_path: str | None = None,
        faiss_index_path: str | None = None,
        metadata_path: str | None = None,
        *,
        bm25_retriever: BM25Retriever | None = None,
        semantic_retriever: SemanticRetriever | None = None,
        model_name: str = None,
        top_k: int = 5,
        system_prompt: str = SYSTEM_PROMPT_V1,
    ) -> None:
        self.top_k = top_k
        self.system_prompt = system_prompt
        self.model_name = model_name or os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

        if bm25_retriever is not None and semantic_retriever is not None:
            self.retriever = HybridRetriever(
                bm25_retriever=bm25_retriever,
                semantic_retriever=semantic_retriever,
                top_k=top_k,
            )
        else:
            if not all(
                [corpus_path, bm25_index_path, bm25_tokens_path, faiss_index_path, metadata_path]
            ):
                raise ValueError(
                    "Provide `bm25_retriever` and `semantic_retriever`, or all path arguments "
                    "(`corpus_path`, `bm25_index_path`, `bm25_tokens_path`, `faiss_index_path`, `metadata_path`)."
                )
            bm25 = BM25Retriever.load_saved(
                corpus_path=corpus_path,
                index_path=bm25_index_path,
                tokens_path=bm25_tokens_path,
            )
            semantic = SemanticRetriever.load_saved(
                index_path=faiss_index_path,
                metadata_path=metadata_path,
            )
            self.retriever = HybridRetriever(
                bm25_retriever=bm25,
                semantic_retriever=semantic,
                top_k=top_k,
            )

        self.llm = ChatGroq(
            model=self.model_name,
            api_key=os.getenv("GROQ_API_KEY"),
        )