from langchain.retrievers import EnsembleRetriever
from helpers.storage import get_vectorstore_retriever, get_bm25_retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.pydantic_v1 import Field

def create_reranker():

    #obtain vectorstore retriver
    vs_retriever = get_vectorstore_retriever()

    #obtain sparse-based retriever
    bm25_Retriever = get_bm25_retriever()

    #initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_Retriever,vs_retriever],
        weights=[0.3,0.7]
    )

    #reranking
    # Initialize the cross encoder model for reranking
    # It uses the specified cross-encoder model
    # top_n determines how many documents are returned after reranking
    cross_encoder_model = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # 3. Initialize the LangChain reranker
    reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=3)
    


    # This is our Stage 2 retriever.
    # It chains the base_retriever and the reranker.
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever
    )


    return compression_retriever


class GradeDocuments():
    """Binary score for relevance check on retrieved documents"""
    binary_score: str = Field(description="Document are relevant to the question 'yes' or 'no'")