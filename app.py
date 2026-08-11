"""
app.py

Streamlit frontend for the Dermatology RAG application.

IMPORTANT:
This file ONLY loads the already-built RAG system.

It does NOT:
    - Load PDFs
    - Chunk documents
    - Generate embeddings
    - Ingest documents
    - Download the knowledge base
    - Modify the Chroma database

Knowledge-base creation is handled separately by ingest.py.

Architecture:

    ingest.py
        ↓
    Chroma + BM25 JSONL corpus
        ↓
    app.py
        ↓
    Hybrid Retriever
        ↓
    Cross-Encoder Reranker
        ↓
    LLM Chain
        ↓
    User
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from helpers.embedder import EmbeddingPipeline
from helpers.retriever import create_hybrid_retriever
from helpers.reranker import create_reranker
from helpers.llm_chain import create_rag_chain


# =============================================================
# ENVIRONMENT
# =============================================================

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")


# =============================================================
# PAGE CONFIG
# =============================================================

st.set_page_config(
    page_title="Dermatology RAG",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧑‍🔬",
)


# =============================================================
# PATH CONFIGURATION
# =============================================================

BASE_DIR = Path(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

HELPERS_DIR = BASE_DIR / "helpers"

CHROMA_DIR = (
    HELPERS_DIR / "chroma_db"
)

BM25_CORPUS = (
    BASE_DIR / "helpers" / "data" / "bm25_corpus.jsonl"
)

IMAGE_PATH = (
    HELPERS_DIR / "files" / "skinwlnes.jpeg"
)


# =============================================================
# HEADER
# =============================================================

if IMAGE_PATH.exists():

    st.image(
        str(IMAGE_PATH),
        width="stretch",
    )

st.title(
    "Dermatology RAG Application"
)


# =============================================================
# ENVIRONMENT VALIDATION
# =============================================================

if not GROQ_API_KEY:

    st.error(
        "GROQ_API_KEY was not found. "
        "Please add it to your .env file."
    )

    st.stop()


# =============================================================
# SESSION STATE
# =============================================================

if "chats" not in st.session_state:

    st.session_state.chats = {}


if "current_chat" not in st.session_state:

    st.session_state.current_chat = None


if "chat_counter" not in st.session_state:

    st.session_state.chat_counter = 0


# =============================================================
# LOAD RAG SYSTEM
# =============================================================

@st.cache_resource
def load_rag_system(
    groq_api_key: str,
):
    """
    Load the EXISTING RAG infrastructure.

    This function does NOT ingest documents.

    It only connects to:

        Existing Chroma database
        Existing BM25 JSONL corpus
        Hybrid retriever
        Reranker
        LLM chain

    The knowledge base must already have been created
    by running ingest.py.
    """

    # =========================================================
    # 1. CHECK CHROMA
    # =========================================================

    if not CHROMA_DIR.exists():

        raise FileNotFoundError(
            f"Chroma database not found at:\n"
            f"{CHROMA_DIR}\n\n"
            f"Run ingest.py first."
        )


    # =========================================================
    # 2. CHECK BM25 CORPUS
    # =========================================================

    if not BM25_CORPUS.exists():

        raise FileNotFoundError(
            f"BM25 corpus not found at:\n"
            f"{BM25_CORPUS}\n\n"
            f"Run ingest.py first."
        )


    # =========================================================
    # 3. LOAD EXISTING CHROMA
    # =========================================================

    embedding_pipeline = EmbeddingPipeline(

        persist_directory=str(
            CHROMA_DIR
        ),

        collection_name=(
            "dermatology_rag"
        ),

        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
    )

    vectorstore = (
        embedding_pipeline.get_vectorstore()
    )


    # =========================================================
    # 4. CREATE HYBRID RETRIEVER
    # =========================================================

    hybrid_retriever = (
        create_hybrid_retriever(

            vectorstore=vectorstore,

            chunks_file=str(
                BM25_CORPUS
            ),

            dense_k=15,

            bm25_k=15,

            weights=[
                0.7,
                0.3,
            ],
        )
    )


    # =========================================================
    # 5. CREATE RERANKER
    # =========================================================

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


    # =========================================================
    # 6. CREATE LLM CHAIN
    # =========================================================

    rag_chain = create_rag_chain(

        reranker=(
            reranker.as_retriever()
        ),

        api_key=groq_api_key,

        model_name=(
            "llama-3.1-8b-instant"
        ),

        temperature=0,
    )


    return rag_chain


# =============================================================
# INITIALIZE RAG SYSTEM
# =============================================================

if "rag_chain" not in st.session_state:

    with st.spinner(
        "⚡ Loading Dermatology knowledge base..."
    ):

        try:

            st.session_state.rag_chain = (
                load_rag_system(
                    GROQ_API_KEY
                )
            )

        except FileNotFoundError as e:

            st.error(str(e))
            st.stop()

        except Exception as e:

            st.error(
                "Failed to load the RAG system."
            )

            st.exception(e)

            st.stop()


# =============================================================
# SIDEBAR
# =============================================================

with st.sidebar:

    st.title(
        "💬 Chat History"
    )


    # ---------------------------------------------------------
    # NEW CHAT
    # ---------------------------------------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True,
    ):

        st.session_state.chat_counter += 1

        new_chat_name = (
            f"Untitled Chat "
            f"{st.session_state.chat_counter}"
        )

        st.session_state.chats[
            new_chat_name
        ] = []

        st.session_state.current_chat = (
            new_chat_name
        )

        st.rerun()


    # ---------------------------------------------------------
    # CHAT LIST
    # ---------------------------------------------------------

    if st.session_state.chats:

        chat_list = list(
            st.session_state.chats.keys()
        )


        # Make sure current chat still exists

        if (
            st.session_state.current_chat
            not in chat_list
        ):

            st.session_state.current_chat = (
                chat_list[0]
            )


        selected_chat = st.radio(

            "Chats",

            chat_list,

            index=chat_list.index(
                st.session_state.current_chat
            ),
        )


        st.session_state.current_chat = (
            selected_chat
        )


# =============================================================
# MAIN CHAT AREA
# =============================================================

if not st.session_state.current_chat:

    st.info(
        "👈 Start a new chat from the sidebar."
    )

    st.stop()


# =============================================================
# CHAT HEADER
# =============================================================

st.header(
    st.session_state.current_chat
)

st.subheader(
    "Ask a question about dermatology"
)


# =============================================================
# DISPLAY CHAT HISTORY
# =============================================================

chat_history = (
    st.session_state.chats[
        st.session_state.current_chat
    ]
)


for message in chat_history:

    if message["role"] == "user":

        with st.chat_message("user"):

            st.write(
                message["content"]
            )

    else:

        with st.chat_message("assistant"):

            st.write(
                message["content"]
            )


# =============================================================
# CHAT INPUT
# =============================================================

question = st.chat_input(
    "Ask me anything about dermatology..."
)


# =============================================================
# PROCESS QUESTION
# =============================================================

if question:

    current_chat = (
        st.session_state.current_chat
    )


    # ---------------------------------------------------------
    # SAVE USER MESSAGE
    # ---------------------------------------------------------

    st.session_state.chats[
        current_chat
    ].append(
        {
            "role": "user",
            "content": question,
        }
    )


    # ---------------------------------------------------------
    # RENAME FIRST CHAT
    # ---------------------------------------------------------

    if (
        current_chat.startswith(
            "Untitled Chat"
        )
        and len(
            st.session_state.chats[
                current_chat
            ]
        ) == 1
    ):

        topic_name = (
            question[:40] + "..."
            if len(question) > 40
            else question
        )


        st.session_state.chats[
            topic_name
        ] = st.session_state.chats.pop(
            current_chat
        )


        st.session_state.current_chat = (
            topic_name
        )


        current_chat = topic_name


    # ---------------------------------------------------------
    # DISPLAY USER MESSAGE
    # ---------------------------------------------------------

    with st.chat_message("user"):

        st.write(question)


    # ---------------------------------------------------------
    # GENERATE ANSWER
    # ---------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            try:

                answer = (
                    st.session_state
                    .rag_chain
                    .invoke(question)
                )

                st.write(answer)


            except Exception as e:

                answer = (
                    "I encountered an error while "
                    "processing your question."
                )

                st.error(
                    "RAG pipeline error."
                )

                st.exception(e)


    # ---------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # ---------------------------------------------------------

    st.session_state.chats[
        current_chat
    ].append(
        {
            "role": "assistant",
            "content": answer,
        }
    )