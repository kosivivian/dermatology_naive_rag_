import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from helpers.doc_loading import load_and_chunk_EPUB, load_and_chunk_pdf
from langchain_core.documents import Document

'''
# Persistent directory for faiss vectorstore
PERSIST_DIR = "./FAISS"

#load the files
#specify the filepath



filepath = "C:/Users/Nebolisa Kosiso/dermatology_naive_rag_/files/Cosmetic Dermatology Products and Procedures (Zoe Diana Draelos) (Z-Library).pdf"

chunks = load_and_chunk_pdf(filepath)



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# If the vectorstore already exists, append to it
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    print("Loading existing FAISS index...")
    db = FAISS.load_local(folder_path=PERSIST_DIR, embeddings=embeddings,allow_dangerous_deserialization=True)
    db.add_documents(chunks)
    db.save_local(PERSIST_DIR)
else:
    print("Creating new FAISS index...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(PERSIST_DIR)

print("FAISS index is ready")



'''
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
def load_faiss():
    """
    load an existing faiss
    """
    db = FAISS.load_local(folder_path="C:/Users/Nebolisa Kosiso/dermatology_naive_rag_/helpers/FAISS", embeddings=embeddings,allow_dangerous_deserialization=True)
    return db

def get_vectorstore_retriever():
    '''
    obtain vectorstore
    '''
    db = load_faiss()
    vectorstore_retriever = db.as_retriever(search_kwargs={"k":5})
    return vectorstore_retriever


def build_chunk_array():
    chunk_array = []

    files_folder = os.path.join(os.path.dirname(__file__), "files")

    for filename in os.listdir(files_folder):
        file_path = os.path.join(files_folder, filename)

        if not os.path.isfile(file_path):
            continue

        #handle pdf
        if filename.lower().endswith(".pdf"):
            print(f"Loading PDF: {filename}")
            chunks = load_and_chunk_pdf(file_path)

        #handle epub files
        elif filename.lower().endswith(".epub"):
            print(f"Loading EPUB: {filename}")
            chunks = load_and_chunk_EPUB(file_path)

        else:
            print(f"Skipping unsuported file: {filename}")
            continue
        # Add chunks to the global chunk_array
        for chunk in chunks:
            doc = Document(
                page_content=chunk.page_content,
                metadata=chunk.metadata
            )
            chunk_array.append(doc)

    print(f"\n✅ Finished! Total chunks collected: {len(chunk_array)}")
    return chunk_array


def get_bm25_retriever():
    '''
    retriever the sparse vector: BM25
    '''
    bm25_retriever = BM25Retriever.from_documents(build_chunk_array()) #
    bm25_retriever.k = 5 #we want the top 5 results
    return bm25_retriever




