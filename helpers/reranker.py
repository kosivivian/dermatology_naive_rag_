"""
reranker.py

Responsible for:
    1. Receiving candidates from the hybrid retriever
    2. Scoring candidates with a cross-encoder
    3. Returning the highest-ranking documents

Pipeline:

    User Query
        ↓
    Hybrid Retriever
        ↓
    Candidate Documents
        ↓
    Cross-Encoder Reranker
        ↓
    Top-N Documents
        ↓
    LLM
"""

from typing import List

from langchain_core.documents import Document

from langchain_community.cross_encoders import (
    HuggingFaceCrossEncoder
)

from langchain.retrievers.document_compressors import (
    CrossEncoderReranker
)

from langchain.retrievers import (
    ContextualCompressionRetriever
)


class RAGReranker:
    """
    Cross-encoder reranking layer.

    The retriever should return a larger candidate pool
    (for example 10-20 documents).

    The cross-encoder then scores those documents against
    the user's query and keeps only the strongest matches.
    """

    def __init__(
        self,
        base_retriever,
        model_name: str = (
            "cross-encoder/"
            "ms-marco-MiniLM-L-6-v2"
        ),
        top_n: int = 5,
    ):

        self.base_retriever = base_retriever
        self.top_n = top_n

        # -----------------------------------------------------
        # Load cross-encoder
        # -----------------------------------------------------

        print(
            f"Loading reranker model: "
            f"{model_name}"
        )

        self.cross_encoder = (
            HuggingFaceCrossEncoder(
                model_name=model_name
            )
        )

        # -----------------------------------------------------
        # LangChain reranker
        # -----------------------------------------------------

        self.reranker = CrossEncoderReranker(
            model=self.cross_encoder,
            top_n=top_n,
        )

        # -----------------------------------------------------
        # Compression retriever
        # -----------------------------------------------------

        self.compression_retriever = (
            ContextualCompressionRetriever(
                base_retriever=base_retriever,
                base_compressor=self.reranker,
            )
        )

        print(
            "✅ Reranker initialized."
        )

        print(
            f"   Model: {model_name}"
        )

        print(
            f"   Returning top: {top_n}"
        )

    # =========================================================
    # RERANK
    # =========================================================

    def rerank(
        self,
        query: str,
    ) -> List[Document]:
        """
        Retrieve candidates and rerank them.

        Args:
            query:
                User's question.

        Returns:
            Top-N reranked documents.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        documents = (
            self.compression_retriever.invoke(
                query
            )
        )

        return documents

    # =========================================================
    # RETURN LANGCHAIN RETRIEVER
    # =========================================================

    def as_retriever(self):

        return self.compression_retriever


# =============================================================
# FACTORY FUNCTION
# =============================================================

def create_reranker(
    base_retriever,
    model_name: str = (
        "cross-encoder/"
        "ms-marco-MiniLM-L-6-v2"
    ),
    top_n: int = 5,
):

    return RAGReranker(
        base_retriever=base_retriever,
        model_name=model_name,
        top_n=top_n,
    )


# =============================================================
# EXAMPLE USAGE
# =============================================================

if __name__ == "__main__":

    from embedder import (
        EmbeddingPipeline
    )

    from retriever import (
        create_hybrid_retriever
    )

    # ---------------------------------------------------------
    # 1. Load Chroma
    # ---------------------------------------------------------

    embedding_pipeline = (
        EmbeddingPipeline(
            persist_directory="./chroma_db",
            collection_name="dermatology_rag",
            bm25_corpus_path=(
                "./data/bm25_corpus.jsonl"
            ),
        )
    )

    vectorstore = (
        embedding_pipeline.get_vectorstore()
    )

    # ---------------------------------------------------------
    # 2. Create hybrid retriever
    # ---------------------------------------------------------

    hybrid_retriever = (
        create_hybrid_retriever(
            vectorstore=vectorstore,
            chunks_file=(
                "./data/bm25_corpus.jsonl"
            ),
            dense_k=15,
            bm25_k=15,
            weights=[0.7, 0.3],
        )
    )

    # ---------------------------------------------------------
    # 3. Create reranker
    # ---------------------------------------------------------

    reranker = create_reranker(
        base_retriever=(
            hybrid_retriever.as_retriever()
        ),
        model_name=(
            "cross-encoder/"
            "ms-marco-MiniLM-L-6-v2"
        ),
        top_n=5,
    )

    # ---------------------------------------------------------
    # 4. Test
    # ---------------------------------------------------------

    query = (
        "What are the common treatments "
        "for acne?"
    )

    results = reranker.rerank(
        query
    )

    print(
        f"\nFinal reranked documents: "
        f"{len(results)}\n"
    )

    for i, document in enumerate(
        results,
        start=1
    ):

        print(
            f"========== RESULT {i} =========="
        )

        print(
            document.page_content[:700]
        )

        print(
            "\nMetadata:"
        )

        print(
            document.metadata
        )

        print()