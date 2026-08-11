from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredEPubLoader,
    UnstructuredPowerPointLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGDocumentProcessor:
    """
    Document ingestion and chunking pipeline for RAG.

    Responsibilities:
    1. Load documents using LangChain loaders
    2. Clean extracted text
    3. Enrich metadata
    4. Split documents into retrieval-friendly chunks

    Embeddings and vector storage are handled separately.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".epub",
        ".pptx",
    }

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                ", ",
                " ",
                "",
            ],
        )

    # ---------------------------------------------------------
    # LOADING
    # ---------------------------------------------------------

    def load(
        self,
        file_path: str
    ) -> List[Document]:
        """
        Load a supported document using LangChain loaders.
        """

        path = Path(file_path)
        extension = path.suffix.lower()

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        if extension == ".pdf":
            loader = PyPDFLoader(file_path)

        elif extension == ".epub":
            loader = UnstructuredEPubLoader(file_path)

        elif extension == ".pptx":
            loader = UnstructuredPowerPointLoader(file_path)

        else:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        documents = loader.load()

        print(
            f"Loaded {len(documents)} document sections "
            f"from {path.name}"
        )

        return documents

    # ---------------------------------------------------------
    # CLEANING
    # ---------------------------------------------------------

    def clean_document(
        self,
        document: Document
    ) -> Document:
        """
        Light text cleaning while preserving meaningful structure.
        """

        text = document.page_content

        # Normalize excessive whitespace.
        text = " ".join(text.split())

        document.page_content = text.strip()

        return document

    # ---------------------------------------------------------
    # METADATA
    # ---------------------------------------------------------

    def enrich_metadata(
        self,
        document: Document,
        source_file: str,
    ) -> Document:
        """
        Add useful metadata without replacing
        existing LangChain metadata.
        """

        document.metadata.update({
            "source_file": source_file,
            "file_type": Path(
                source_file
            ).suffix.lower(),
        })

        return document

    # ---------------------------------------------------------
    # FILTERING
    # ---------------------------------------------------------

    def is_valid_document(
        self,
        document: Document
    ) -> bool:
        """
        Remove obviously empty or useless documents.
        """

        text = document.page_content.strip()

        if len(text) < self.min_chunk_size:
            return False

        return True

    # ---------------------------------------------------------
    # CHUNKING
    # ---------------------------------------------------------

    def chunk_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Split documents using LangChain's
        RecursiveCharacterTextSplitter.
        """

        chunks = self.splitter.split_documents(
            documents
        )

        # Add chunk-level metadata.
        for index, chunk in enumerate(chunks):

            chunk.metadata["chunk_index"] = index

            chunk.metadata["chunk_size"] = len(
                chunk.page_content
            )

        return chunks

    # ---------------------------------------------------------
    # COMPLETE SINGLE-FILE PIPELINE
    # ---------------------------------------------------------

    def process(
        self,
        file_path: str
    ) -> List[Document]:
        """
        Complete processing pipeline for ONE file.

        Flow:

            File
             ↓
            Load
             ↓
            Clean
             ↓
            Filter
             ↓
            Metadata enrichment
             ↓
            Chunk
             ↓
            LangChain Documents
        """

        source_file = Path(
            file_path
        ).name

        # 1. Load
        documents = self.load(
            file_path
        )

        # 2. Clean + filter + metadata
        cleaned_documents = []

        for document in documents:

            document = self.clean_document(
                document
            )

            if not self.is_valid_document(
                document
            ):
                continue

            document = self.enrich_metadata(
                document,
                source_file,
            )

            cleaned_documents.append(
                document
            )

        # 3. Chunk
        chunks = self.chunk_documents(
            cleaned_documents
        )

        print(
            f"Created {len(chunks):,} chunks "
            f"from {source_file}"
        )

        return chunks

    # ---------------------------------------------------------
    # FILE DISCOVERY
    # ---------------------------------------------------------

    def get_files(
        self,
        folder_path: str,
        recursive: bool = False,
    ) -> List[Path]:
        """
        Discover supported files without processing them.

        This is useful for registry-aware ingestion because
        ingest.py can discover files first, check their hashes,
        and only then call process() on files that need work.
        """

        folder = Path(
            folder_path
        )

        if not folder.exists():
            raise ValueError(
                f"Folder does not exist: {folder_path}"
            )

        if not folder.is_dir():
            raise ValueError(
                f"Not a directory: {folder_path}"
            )

        files = []

        for extension in self.SUPPORTED_EXTENSIONS:

            pattern = f"*{extension}"

            if recursive:
                files.extend(
                    folder.rglob(pattern)
                )
            else:
                files.extend(
                    folder.glob(pattern)
                )

        return sorted(
            files,
            key=lambda path: path.name.lower()
        )

    # ---------------------------------------------------------
    # FOLDER PROCESSING
    # ---------------------------------------------------------

    def process_folder(
        self,
        folder_path: str,
        recursive: bool = False,
        batch_size: int = 5,
    ) -> List[Document]:
        """
        Process all supported files in batches.

        NOTE:
        Registry-aware ingestion should normally be handled
        by ingest.py. In that case, ingest.py should call
        process() only for files that are new or modified.

        This method remains useful for processing an entire
        folder when registry management is not required.
        """

        files = self.get_files(
            folder_path,
            recursive=recursive,
        )

        print(
            f"Found {len(files)} supported documents."
        )

        if not files:
            return []

        all_chunks = []

        for start in range(
            0,
            len(files),
            batch_size,
        ):

            batch = files[
                start:
                start + batch_size
            ]

            batch_number = (
                start // batch_size
            ) + 1

            total_batches = (
                len(files) + batch_size - 1
            ) // batch_size

            print(
                f"\nProcessing batch "
                f"{batch_number}/{total_batches}"
            )

            for file_path in batch:

                try:

                    chunks = self.process(
                        str(file_path)
                    )

                    all_chunks.extend(
                        chunks
                    )

                except Exception as e:

                    print(
                        f"Error processing "
                        f"{file_path.name}: {e}"
                    )

        print(
            f"\nTotal chunks created: "
            f"{len(all_chunks):,}"
        )

        return all_chunks


# =============================================================
# USAGE
# =============================================================

if __name__ == "__main__":

    processor = RAGDocumentProcessor(
        chunk_size=1000,
        chunk_overlap=200,
        min_chunk_size=100,
    )

    chunks = processor.process_folder(
        "./documents",
        recursive=False,
        batch_size=5,
    )

    for i, chunk in enumerate(
        chunks[:5]
    ):

        print(
            f"\n--- Chunk {i + 1} ---"
        )

        print(
            chunk.page_content[:500]
        )

        print(
            "\nMetadata:"
        )

        print(
            chunk.metadata
        )
