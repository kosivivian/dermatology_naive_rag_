"""
embedding_pipeline.py

Responsible for:

1. Receiving LangChain Document chunks
2. Generating deterministic chunk IDs
3. Detecting already-indexed chunks
4. Embedding only new chunks
5. Storing embeddings in Chroma
6. Persisting chunks for BM25 retrieval

Document loading and chunking are handled separately.
"""

import hashlib
import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class EmbeddingPipeline:

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "dermatology_rag",
        bm25_corpus_path: str = "./data/bm25_corpus.jsonl",
        model_name: str = (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        embedding_batch_size: int = 32,
    ):

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.bm25_corpus_path = Path(
            bm25_corpus_path
        )

        # Make sure directories exist
        Path(persist_directory).mkdir(
            parents=True,
            exist_ok=True
        )

        self.bm25_corpus_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # -----------------------------------------------------
        # Embedding model
        # -----------------------------------------------------

        print(
            f"Loading embedding model: {model_name}"
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={
                "batch_size": embedding_batch_size
            }
        )

        # -----------------------------------------------------
        # Chroma
        # -----------------------------------------------------

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

        print(
            "✅ Embedding model and Chroma ready."
        )

    # =========================================================
    # CHUNK ID
    # =========================================================

    @staticmethod
    def generate_chunk_id(
        document: Document,
    ) -> str:

        source = document.metadata.get(
            "source_file",
            document.metadata.get(
                "source",
                "unknown"
            )
        )

        page = document.metadata.get(
            "page",
            document.metadata.get(
                "page_number",
                "unknown"
            )
        )

        content = document.page_content.strip()

        raw_id = (
            f"{source}|"
            f"{page}|"
            f"{content}"
        )

        return hashlib.sha256(
            raw_id.encode("utf-8")
        ).hexdigest()

    # =========================================================
    # PREPARE CHUNKS
    # =========================================================

    def prepare_chunks(
        self,
        chunks: List[Document],
    ) -> List[Document]:

        for chunk in chunks:

            chunk_id = self.generate_chunk_id(
                chunk
            )

            chunk.metadata["chunk_id"] = (
                chunk_id
            )

        return chunks

    # =========================================================
    # CHECK CHROMA FOR EXISTING CHUNKS
    # =========================================================

    def get_existing_ids(
        self,
        chunk_ids: List[str],
    ) -> set:

        if not chunk_ids:
            return set()

        existing = self.vectorstore.get(
            ids=chunk_ids,
            include=[]
        )

        return set(
            existing.get("ids", [])
        )

    # =========================================================
    # SAVE CHUNKS FOR BM25
    # =========================================================

    def save_to_bm25_corpus(
        self,
        chunks: List[Document],
    ):
        """
        Append newly indexed chunks to the BM25 corpus.

        JSONL format:
            One document per line.

        This allows the corpus to grow incrementally.
        """

        if not chunks:
            return

        with open(
            self.bm25_corpus_path,
            "a",
            encoding="utf-8"
        ) as f:

            for chunk in chunks:

                record = {
                    "page_content": (
                        chunk.page_content
                    ),
                    "metadata": (
                        chunk.metadata
                    ),
                }

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

        print(
            f"Added {len(chunks)} chunks "
            f"to BM25 corpus."
        )

    # =========================================================
    # INGEST
    # =========================================================

    def ingest(
        self,
        chunks: List[Document],
    ):
        """
    Incrementally ingest chunks into Chroma.

    New chunks are added in batches so that we do not exceed
    Chroma's maximum batch size.

    Existing chunks are skipped based on their deterministic IDs.
    """

        if not chunks:

            print(
                "No chunks received."
            )

            return

        print(
            f"\nReceived {len(chunks)} chunks."
        )

        # -----------------------------------------------------
        # 1. Generate deterministic IDs
        # -----------------------------------------------------

        chunks = self.prepare_chunks(
            chunks
        )

        chunk_ids = [
            chunk.metadata["chunk_id"]
            for chunk in chunks
        ]

        # -----------------------------------------------------
        # 2. Find existing chunks
        # -----------------------------------------------------

        existing_ids = (
            self.get_existing_ids(
                chunk_ids
            )
        )

        # -----------------------------------------------------
        # 3. Keep only NEW chunks
        # -----------------------------------------------------

        new_chunks = [
            chunk
            for chunk in chunks
            if chunk.metadata["chunk_id"]
            not in existing_ids
        ]

        print(
            f"Already indexed: "
            f"{len(existing_ids)}"
        )

        print(
            f"New chunks: "
            f"{len(new_chunks)}"
        )

        if not new_chunks:

            print(
                "Nothing new to embed."
            )

            return

        # -----------------------------------------------------
        # 4. Get IDs for new chunks
        # -----------------------------------------------------

        new_ids = [
            chunk.metadata["chunk_id"]
            for chunk in new_chunks
        ]

        # ---------------------------------------------------------
        # Batch insertion into Chroma
        # ---------------------------------------------------------
        # Chroma has a maximum batch size. Insert in smaller
        # batches instead of sending all chunks at once.

        BATCH_SIZE = 1000

        total_chunks = len(new_chunks)
        total_batches = (
            total_chunks + BATCH_SIZE - 1
        ) // BATCH_SIZE

        print(
            f"Adding {total_chunks:,} chunks "
            f"in {total_batches} batches..."
        )

        for start in range(
            0,
            total_chunks,
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                total_chunks
            )

            batch_documents = new_chunks[start:end]
            batch_ids = new_ids[start:end]

            batch_number = (
                start // BATCH_SIZE
            ) + 1

            print(
                f"  Batch {batch_number}/{total_batches}: "
                f"{len(batch_documents):,} chunks"
            )

            self.vectorstore.add_documents(
                documents=batch_documents,
                ids=batch_ids,
            )

            print(
                f"  ✓ Batch {batch_number} inserted"
            )

        print(
            f"✅ Successfully inserted "
            f"{total_chunks:,} chunks into Chroma."
        )

        # -----------------------------------------------------
        # 6. Store same chunks for BM25
        # -----------------------------------------------------

        self.save_to_bm25_corpus(
            new_chunks
        )

        print(
            "\n✅ Incremental ingestion complete."
        )

    # =========================================================
    # RETRIEVER
    # =========================================================

    def get_vectorstore(self):

        return self.vectorstore

    # =========================================================
    # STATISTICS
    # =========================================================

    def count_documents(self) -> int:

        result = self.vectorstore.get(
            include=[]
        )

        return len(
            result.get("ids", [])
        )


# =============================================================
# EXAMPLE
# =============================================================

if __name__ == "__main__":

    from load_chunk import (
        RAGDocumentProcessor
    )

    # ---------------------------------------------------------
    # 1. Load + chunk
    # ---------------------------------------------------------

    processor = RAGDocumentProcessor(
        chunk_size=1000,
        chunk_overlap=200,
        min_chunk_size=100,
    )

    chunks = processor.process_folder(
        "./documents",
        recursive=False,
    )

    # ---------------------------------------------------------
    # 2. Embed + store
    # ---------------------------------------------------------

    pipeline = EmbeddingPipeline(
        persist_directory="./chroma_db",
        collection_name="dermatology_rag",
        bm25_corpus_path=(
            "./data/bm25_corpus.jsonl"
        ),
        embedding_batch_size=32,
    )

    pipeline.ingest(chunks)

    print(
        f"\nTotal Chroma chunks: "
        f"{pipeline.count_documents()}"
    )