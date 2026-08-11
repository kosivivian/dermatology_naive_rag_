"""
chain.py

Responsible for:

    1. Receiving the reranker
    2. Retrieving + reranking relevant documents
    3. Formatting retrieved context
    4. Building the LLM prompt
    5. Generating the final answer

Architecture:

    User Question
          ↓
    RAGReranker
          ↓
    Hybrid Retriever
      ↙         ↘
   Chroma      BM25
          ↓
    Cross-Encoder
          ↓
    Top-N Documents
          ↓
    Prompt
          ↓
    Groq LLM
          ↓
       Answer
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq


# =============================================================
# CONTEXT FORMATTER
# =============================================================

def format_docs(docs: List[Document]) -> str:
    """
    Convert retrieved documents into a single context string.

    Metadata is included so the LLM has information about
    where the retrieved content came from.
    """

    formatted_documents = []

    for i, doc in enumerate(docs, start=1):

        metadata = doc.metadata

        source = metadata.get(
            "source_file",
            metadata.get(
                "source",
                "Unknown source"
            )
        )

        page = metadata.get(
            "page_number",
            metadata.get(
                "page",
                "Unknown page"
            )
        )

        formatted_document = (
            f"[Source {i}]\n"
            f"Document: {source}\n"
            f"Page: {page}\n\n"
            f"{doc.page_content}"
        )

        formatted_documents.append(
            formatted_document
        )

    return "\n\n".join(
        formatted_documents
    )


# =============================================================
# PROMPT
# =============================================================

def create_prompt() -> ChatPromptTemplate:
    """
    Create the prompt used by the RAG LLM.
    """

    prompt_template = """
You are a knowledgeable dermatology and skincare
information assistant.

Your role is to help people understand information about
skin, hair, nails, cosmetic dermatology, and skincare in
simple and accessible language.

You must follow these rules:

1. Answer the question using ONLY the information provided
   in the context below.

2. Do not invent facts or fill gaps using information that
   is not present in the context.

3. If the context does not contain enough information to
   answer the question, clearly say that the available
   information does not contain the answer.

4. Explain technical terminology in simple language when
   necessary.

5. Give a clear, helpful and well-structured answer.

6. Do not unnecessarily repeat information.

7. When multiple sources provide relevant information,
   synthesize them into one coherent answer.

8. Do not claim that the retrieved information is a
   diagnosis or personalized medical advice.

9. If the context contains conflicting information, mention
   the conflict rather than choosing a fact without
   explanation.

Context:
----------------
{context}
----------------

Question:
{question}

Answer:
"""

    return ChatPromptTemplate.from_template(
        prompt_template
    )


# =============================================================
# RAG CHAIN
# =============================================================

def create_rag_chain(
    reranker,
    api_key: str,
    model_name: str = "llama-3.1-8b-instant",
    temperature: float = 0,
):
    """
    Create the complete RAG question-answering chain.

    Args:
        reranker:
            RAGReranker instance. The reranker internally
            handles hybrid retrieval and cross-encoder
            reranking.

        api_key:
            Groq API key.

        model_name:
            Groq model used for generation.

        temperature:
            LLM temperature. 0 is recommended for factual
            RAG responses.

    Returns:
        A LangChain LCEL runnable.
    """

    # ---------------------------------------------------------
    # 1. Initialize LLM
    # ---------------------------------------------------------

    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )

    # ---------------------------------------------------------
    # 2. Create prompt
    # ---------------------------------------------------------

    prompt = create_prompt()

    # ---------------------------------------------------------
    # 3. Build RAG chain
    # ---------------------------------------------------------

    rag_chain = (
        {
            "context": (
                reranker
                | format_docs
            ),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# =============================================================
# OPTIONAL: SIMPLE INVOCATION HELPER
# =============================================================

def ask(
    rag_chain,
    question: str,
) -> str:
    """
    Send a question to the RAG chain.
    """

    if not question or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    return rag_chain.invoke(
        question
    )


