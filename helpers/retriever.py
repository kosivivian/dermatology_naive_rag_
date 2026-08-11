"""
retriever.py

Hybrid retrieval layer.

Responsibilities:
    1. Use the persistent Chroma vector store
    2. Create dense vector retrieval
    3. Extract the stored documents from Chroma for BM25
    4. Create sparse BM25 retrieval
    5. Combine dense + sparse retrieval using EnsembleRetriever

Reranking is handled separately in reranker.py.

Architecture:

    Chroma
       │
       ├── Dense Retriever
       │
       └── Stored Documents
                │
                ▼
             BM25
                │
                ▼
        EnsembleRetriever
                │
                ▼
           Reranker
"""


from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


class HybridRetriever:
    """
    Hybrid retriever combining:

        Dense retrieval → Chroma
        Sparse retrieval → BM25

    The two retrieval methods are combined using
    LangChain's EnsembleRetriever.

    Chroma is the single source of truth for the
    document corpus.
    """

    def __init__(
        self,
        vectorstore,
        dense_k: int = 10,
        bm25_k: int = 10,
        weights: Optional[List[float]] = None,
    ):

        self.vectorstore = vectorstore

        self.dense_k = dense_k
        self.bm25_k = bm25_k

        # Default:
        # Dense = 70%
        # BM25 = 30%
        self.weights = (
            weights
            if weights is not None
            else [0.7, 0.3]
        )

        # Validate weights
        if len(self.weights) != 2:
            raise ValueError(
                "weights must contain exactly "
                "two values: [dense_weight, bm25_weight]"
            )

        # -----------------------------------------------------
        # Dense retriever
        # -----------------------------------------------------

        self.dense_retriever = (
            self.vectorstore.as_retriever(
                search_kwargs={
                    "k": dense_k
                }
            )
        )

        # -----------------------------------------------------
        # Load documents from Chroma
        # -----------------------------------------------------

        documents = self._load_documents_from_chroma()

        if not documents:

            raise ValueError(
                "No documents found in Chroma. "
                "Run ingest.py before starting the application."
            )

        # -----------------------------------------------------
        # BM25 retriever
        # -----------------------------------------------------

        self.bm25_retriever = (
            BM25Retriever.from_documents(
                documents
            )
        )

        self.bm25_retriever.k = bm25_k

        # -----------------------------------------------------
        # Hybrid retriever
        # -----------------------------------------------------

        self.hybrid_retriever = (
            EnsembleRetriever(
                retrievers=[
                    self.dense_retriever,
                    self.bm25_retriever,
                ],
                weights=self.weights,
            )
        )

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------

        print(
            "\n✅ Hybrid retriever initialized."
        )

        print(
            f"   Chroma documents: "
            f"{len(documents):,}"
        )

        print(
            f"   Dense retrieval: "
            f"k={dense_k}"
        )

        print(
            f"   BM25 retrieval: "
            f"k={bm25_k}"
        )

        print(
            f"   Weights: "
            f"dense={self.weights[0]}, "
            f"BM25={self.weights[1]}"
        )

    # =========================================================
    # LOAD DOCUMENTS FROM CHROMA
    # =========================================================

    def _load_documents_from_chroma(
        self,
    ) -> List[Document]:
        """
        Load the document corpus directly from Chroma.

        This replaces the old JSON-based BM25 corpus.

        Chroma stores:

            documents
            metadatas
            ids
            embeddings

        We only need:

            documents
            metadatas

        for BM25.
        """

        try:

            collection = (
                self.vectorstore
                ._collection
            )

            results = collection.get(
                include=[
                    "documents",
                    "metadatas",
                ]
            )

        except Exception as e:

            raise RuntimeError(
                "Could not read documents "
                f"from Chroma: {e}"
            )

        documents = (
            results.get(
                "documents"
            )
            or []
        )

        metadatas = (
            results.get(
                "metadatas"
            )
            or []
        )

        if not documents:

            return []

        langchain_documents = []

        for index, text in enumerate(
            documents
        ):

            if not text:
                continue

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                and metadatas[index]
                else {}
            )

            langchain_documents.append(
                Document(
                    page_content=text,
                    metadata=metadata,
                )
            )

        print(
            f"Loaded "
            f"{len(langchain_documents):,} "
            f"documents from Chroma "
            f"for BM25."
        )

        return langchain_documents

    # =========================================================
    # RETRIEVE
    # =========================================================

    def retrieve(
        self,
        query: str,
    ) -> List[Document]:
        """
        Perform hybrid retrieval.

        Returns candidate documents before reranking.
        """

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        documents = (
            self.hybrid_retriever.invoke(
                query
            )
        )

        return documents

    # =========================================================
    # RETURN LANGCHAIN RETRIEVER
    # =========================================================

    def as_retriever(self):

        return self.hybrid_retriever


# =============================================================
# FACTORY FUNCTION
# =============================================================

def create_hybrid_retriever(
    vectorstore,
    chunks_file: str = "./data/bm25_corpus.jsonl",
    dense_k: int = 10,
    bm25_k: int = 10,
    weights: Optional[List[float]] = None,
):

    return HybridRetriever(
        vectorstore=vectorstore,
        dense_k=dense_k,
        bm25_k=bm25_k,
        weights=weights,
    )


# =============================================================
# EXAMPLE USAGE
# =============================================================

if __name__ == "__main__":

    from embedder import EmbeddingPipeline

    # ---------------------------------------------------------
    # Load persistent Chroma
    # ---------------------------------------------------------

    embedding_pipeline = EmbeddingPipeline(
        persist_directory="./chroma_db",
        collection_name="dermatology_rag",
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
    )

    vectorstore = (
        embedding_pipeline.vectorstore
    )

    # ---------------------------------------------------------
    # Create hybrid retriever
    # ---------------------------------------------------------

    retriever = create_hybrid_retriever(
        vectorstore=vectorstore,
        dense_k=10,
        bm25_k=10,
        weights=[0.7, 0.3],
    )

    # ---------------------------------------------------------
    # Test retrieval
    # ---------------------------------------------------------

    query = (
        "What are the common treatments "
        "for acne?"
    )

    documents = retriever.retrieve(
        query
    )

    print(
        f"\nRetrieved "
        f"{len(documents)} "
        f"candidate documents:\n"
    )

    for i, document in enumerate(
        documents,
        start=1
    ):

        print(
            f"--- Candidate {i} ---"
        )

        print(
            document.page_content[:500]
        )

        print(
            "\nMetadata:"
        )

        print(
            document.metadata
        )

        print()
