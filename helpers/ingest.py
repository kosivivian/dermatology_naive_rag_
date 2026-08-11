"""
ingest.py

Full document lifecycle management for the Dermatology RAG.

Responsibilities:
    1. Discover documents in ./files
    2. Calculate SHA-256 hash for every file
    3. Compare files against the processing registry
    4. Skip unchanged files
    5. Process new files
    6. Re-process modified files
    7. Remove chunks belonging to modified/deleted files
    8. Embed and store new chunks in Chroma
    9. Update the registry only after successful ingestion
   10. Process files in batches of 5

Document lifecycle:

    NEW
      ↓
    PROCESS
      ↓
    STORED
      ↓
    UNCHANGED → SKIP

    MODIFIED
      ↓
    DELETE OLD CHUNKS
      ↓
    REPROCESS
      ↓
    STORED

    DELETED
      ↓
    DELETE OLD CHUNKS
      ↓
    REMOVE FROM REGISTRY
"""

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from load_chunk import RAGDocumentProcessor
from embedder import EmbeddingPipeline


# =============================================================
# PATH CONFIGURATION
# =============================================================

BASE_DIR = Path(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FILES_DIR = BASE_DIR / "files"

DATA_DIR = BASE_DIR.parent / "data"

REGISTRY_PATH = DATA_DIR / "file_registry.json"

CHROMA_DIR = BASE_DIR / "chroma_db"


# =============================================================
# CONFIGURATION
# =============================================================

BATCH_SIZE = 5

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".epub",
    ".pptx",
}


# =============================================================
# FILE HASHING
# =============================================================

def calculate_file_hash(
    file_path: str
) -> str:
    """
    Calculate SHA-256 hash of a file.

    The hash changes if the contents of the file change.
    """

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


# =============================================================
# REGISTRY
# =============================================================

def load_registry() -> dict:
    """
    Load the document processing registry.

    Returns:
        Dictionary containing previously processed files.
    """

    if not REGISTRY_PATH.exists():
        return {}

    try:

        with open(
            REGISTRY_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except (
        json.JSONDecodeError,
        OSError
    ) as e:

        print(
            f"⚠️ Could not load registry: {e}"
        )

        return {}


def save_registry(
    registry: dict
):
    """
    Save registry atomically.

    Registry is written to a temporary file first
    to reduce the risk of corruption.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_path = (
        str(REGISTRY_PATH)
        + ".tmp"
    )

    with open(
        temp_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            registry,
            f,
            indent=4
        )

    os.replace(
        temp_path,
        REGISTRY_PATH
    )


# =============================================================
# FILE DISCOVERY
# =============================================================

def discover_files(
    folder: Path
) -> list[Path]:
    """
    Discover supported documents in the files directory.

    Only files in the top-level directory are processed.
    """

    if not folder.exists():

        raise FileNotFoundError(
            f"Files directory not found: {folder}"
        )

    if not folder.is_dir():

        raise NotADirectoryError(
            f"Not a directory: {folder}"
        )

    files = []

    for path in folder.iterdir():

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):
            continue

        files.append(path)

    return sorted(
        files,
        key=lambda p: p.name.lower()
    )


# =============================================================
# CHROMA DOCUMENT DELETION
# =============================================================

def delete_file_chunks(
    embedding_pipeline,
    source_file: str,
):
    """
    Delete all Chroma chunks belonging to a source file.

    This is required when a document is modified or deleted.
    """

    collection = (
        embedding_pipeline.vectorstore
        ._collection
    )

    try:

        results = collection.get(
            where={
                "source_file": source_file
            },
            include=[]
        )

        ids = results.get(
            "ids",
            []
        )

        if not ids:

            print(
                f"   No existing chunks found "
                f"for {source_file}"
            )

            return 0

        DELETE_BATCH_SIZE = 1000

        deleted_count = 0

        for start in range(
            0,
            len(ids),
            DELETE_BATCH_SIZE
        ):

            batch_ids = ids[
                start:
                start + DELETE_BATCH_SIZE
            ]

            collection.delete(
                ids=batch_ids
            )

            deleted_count += len(
                batch_ids
            )

        print(
            f"   🗑️ Deleted "
            f"{deleted_count:,} old chunks"
        )

        return deleted_count

    except Exception as e:

        raise RuntimeError(
            f"Failed to delete chunks for "
            f"{source_file}: {e}"
        )


# =============================================================
# DETERMINE FILE STATUS
# =============================================================

def determine_file_status(
    file_path: Path,
    registry: dict,
) -> tuple[str, str]:
    """
    Determine whether a file is:

        new
        unchanged
        modified

    Returns:
        (status, current_hash)
    """

    filename = file_path.name

    current_hash = calculate_file_hash(
        str(file_path)
    )

    previous_record = registry.get(
        filename
    )

    if previous_record is None:

        return (
            "new",
            current_hash
        )

    previous_hash = previous_record.get(
        "hash"
    )

    if previous_hash == current_hash:

        return (
            "unchanged",
            current_hash
        )

    return (
        "modified",
        current_hash
    )


# =============================================================
# UPDATE REGISTRY
# =============================================================

def create_registry_record(
    file_path: Path,
    file_hash: str,
    chunk_count: int,
) -> dict:
    """
    Create the registry record for a successfully
    processed document.
    """

    return {

        "hash": file_hash,

        "size_bytes": (
            file_path.stat().st_size
        ),

        "modified_time": (
            datetime.fromtimestamp(
                file_path.stat().st_mtime,
                tz=timezone.utc
            ).isoformat()
        ),

        "chunk_count": chunk_count,

        "status": "processed",

        "last_processed": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }


# =============================================================
# PROCESS ONE FILE
# =============================================================

def process_file(
    file_path: Path,
    status: str,
    current_hash: str,
    registry: dict,
    processor: RAGDocumentProcessor,
    embedding_pipeline: EmbeddingPipeline,
) -> int:
    """
    Process one new or modified file.

    Returns:
        Number of chunks successfully stored.
    """

    filename = file_path.name

    print(
        f"\n📄 {filename}"
    )

    # ---------------------------------------------------------
    # Handle modified file
    # ---------------------------------------------------------

    if status == "modified":

        print(
            "   🔄 File modified"
        )

        print(
            "   → Removing old chunks..."
        )

        delete_file_chunks(
            embedding_pipeline,
            filename,
        )

    elif status == "new":

        print(
            "   🆕 New file"
        )

    # ---------------------------------------------------------
    # Load + chunk
    # ---------------------------------------------------------

    print(
        "   → Loading and chunking..."
    )

    chunks = processor.process(
        str(file_path)
    )

    print(
        f"   → Created "
        f"{len(chunks):,} chunks"
    )

    if not chunks:

        raise RuntimeError(
            "No valid chunks were created."
        )

    # ---------------------------------------------------------
    # Embed + store
    # ---------------------------------------------------------

    print(
        "   → Embedding and storing..."
    )

    embedding_pipeline.ingest(
        chunks
    )

    # ---------------------------------------------------------
    # Update registry ONLY after successful
    # embedding + storage
    # ---------------------------------------------------------

    registry[filename] = create_registry_record(
        file_path=file_path,
        file_hash=current_hash,
        chunk_count=len(chunks),
    )

    save_registry(
        registry
    )

    print(
        "   ✅ Successfully processed"
    )

    return len(chunks)


# =============================================================
# MAIN INGESTION
# =============================================================

def run_ingestion():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DERMATOLOGY RAG DOCUMENT INGESTION"
    )

    print(
        "=" * 70
    )

    # ---------------------------------------------------------
    # 1. Load registry
    # ---------------------------------------------------------

    registry = load_registry()

    print(
        f"\n📋 Registry contains "
        f"{len(registry)} tracked files."
    )

    # ---------------------------------------------------------
    # 2. Discover current files
    # ---------------------------------------------------------

    current_files = discover_files(
        FILES_DIR
    )

    print(
        f"📁 Found "
        f"{len(current_files)} supported files."
    )

    current_files_by_name = {
        path.name: path
        for path in current_files
    }

    # ---------------------------------------------------------
    # 3. Initialize processor
    # ---------------------------------------------------------

    processor = RAGDocumentProcessor(
        chunk_size=1000,
        chunk_overlap=200,
        min_chunk_size=100,
    )

    # ---------------------------------------------------------
    # 4. Initialize embedding pipeline
    # ---------------------------------------------------------

    embedding_pipeline = EmbeddingPipeline(
        persist_directory=str(
            CHROMA_DIR
        ),
        collection_name="dermatology_rag",
        embedding_batch_size=32,
    )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    new_files = []
    unchanged_files = []
    modified_files = []
    failed_files = []
    deleted_files = []

    total_new_chunks = 0

    # =========================================================
    # PHASE 1 — DETERMINE FILE STATES
    # =========================================================

    print(
        "\n"
        + "-" * 70
    )

    print(
        "CHECKING DOCUMENTS"
    )

    print(
        "-" * 70
    )

    files_to_process = []

    for file_path in current_files:

        filename = file_path.name

        try:

            status, current_hash = (
                determine_file_status(
                    file_path,
                    registry,
                )
            )

            if status == "unchanged":

                print(
                    f"✓ {filename} "
                    f"→ unchanged"
                )

                unchanged_files.append(
                    filename
                )

                continue

            if status == "new":

                print(
                    f"🆕 {filename} "
                    f"→ new"
                )

                new_files.append(
                    filename
                )

            elif status == "modified":

                print(
                    f"🔄 {filename} "
                    f"→ modified"
                )

                modified_files.append(
                    filename
                )

            files_to_process.append(
                (
                    file_path,
                    status,
                    current_hash,
                )
            )

        except Exception as e:

            print(
                f"❌ {filename} "
                f"→ hash check failed: {e}"
            )

            failed_files.append(
                filename
            )

    # =========================================================
    # PHASE 2 — PROCESS NEW / MODIFIED FILES
    # =========================================================

    print(
        "\n"
        + "-" * 70
    )

    print(
        "PROCESSING DOCUMENTS"
    )

    print(
        "-" * 70
    )

    total_to_process = len(
        files_to_process
    )

    if total_to_process == 0:

        print(
            "\n✓ No new or modified documents."
        )

    else:

        total_batches = (
            total_to_process
            + BATCH_SIZE
            - 1
        ) // BATCH_SIZE

        for start in range(
            0,
            total_to_process,
            BATCH_SIZE
        ):

            batch = files_to_process[
                start:
                start + BATCH_SIZE
            ]

            batch_number = (
                start // BATCH_SIZE
            ) + 1

            print(
                f"\n"
                f"📦 Batch "
                f"{batch_number}/{total_batches}"
            )

            for (
                file_path,
                status,
                current_hash,
            ) in batch:

                try:

                    chunks_stored = process_file(
                        file_path=file_path,
                        status=status,
                        current_hash=current_hash,
                        registry=registry,
                        processor=processor,
                        embedding_pipeline=embedding_pipeline,
                    )

                    total_new_chunks += (
                        chunks_stored
                    )

                except Exception as e:

                    print(
                        f"   ❌ Failed: {e}"
                    )

                    failed_files.append(
                        file_path.name
                    )

    # =========================================================
    # PHASE 3 — HANDLE DELETED FILES
    # =========================================================

    print(
        "\n"
        + "-" * 70
    )

    print(
        "CHECKING FOR DELETED DOCUMENTS"
    )

    print(
        "-" * 70
    )

    registered_filenames = set(
        registry.keys()
    )

    current_filenames = set(
        current_files_by_name.keys()
    )

    missing_files = (
        registered_filenames
        - current_filenames
    )

    if not missing_files:

        print(
            "✓ No deleted documents."
        )

    for filename in sorted(
        missing_files
    ):

        print(
            f"\n🗑️ {filename}"
        )

        try:

            print(
                "   → Removing old chunks..."
            )

            delete_file_chunks(
                embedding_pipeline,
                filename,
            )

            del registry[
                filename
            ]

            save_registry(
                registry
            )

            deleted_files.append(
                filename
            )

            print(
                "   ✅ Removed from knowledge base"
            )

        except Exception as e:

            print(
                f"   ❌ Failed: {e}"
            )

            failed_files.append(
                filename
            )

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    total_documents = (
        embedding_pipeline.count_documents()
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "INGESTION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"🆕 New files:        "
        f"{len(new_files)}"
    )

    print(
        f"🔄 Modified files:   "
        f"{len(modified_files)}"
    )

    print(
        f"✓ Unchanged files:   "
        f"{len(unchanged_files)}"
    )

    print(
        f"🗑️ Deleted files:    "
        f"{len(deleted_files)}"
    )

    print(
        f"❌ Failed files:     "
        f"{len(failed_files)}"
    )

    print(
        f"📦 New chunks:       "
        f"{total_new_chunks:,}"
    )

    print(
        f"🗄️ Total Chroma:     "
        f"{total_documents:,}"
    )

    print(
        "=" * 70
    )


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    run_ingestion()
