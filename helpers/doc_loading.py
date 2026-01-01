from langchain_community.document_loaders import PyPDFLoader, UnstructuredEPubLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load(filepath):
    try:
        loader = PyPDFLoader(filepath)

        documents = loader.load()

        print(f"Loaded {len(documents)} documents.")
        print("---")

        #inspect the first document
        for i, doc in enumerate(documents):
            print(f"--- Document {i+1} ---")
            print("Page Content Snippet:")
            # We print only the first 100 chars to keep the output clean

            print(doc.page_content[:100] + "...")
            print("\nMetadata:")
            print(doc.metadata)
            print("\n")
    except FileNotFoundError:
        print("Error: 'example_report.pdf' not found. Please create this file to run the example.")
   


def load_and_chunk_pdf(file_path):
    """
    loads a document splits into smaller chunks.
    
    Args:
        data: A list of Document objects to be split.
        chunk_size: The maximum size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between chunks.
        
    Returns:
        A list of chunked Document objects.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    '''
    page_number = 150
    for i, chunk in enumerate(chunks):
        if "page" in chunk.metadata and chunk.metadata["page"] == page_number:
            print(f"--- Chunk {i+1} ---")
            print(f"Content: '{chunk.page_content}'")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Metadata: {chunk.metadata}") # Notice the metadata is preserved!
            print("-" * 10)
'''
    return chunks

def load_and_chunk_EPUB(file_path):
    loader = UnstructuredEPubLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)