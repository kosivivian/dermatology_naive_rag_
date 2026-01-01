import streamlit as st
from dotenv import load_dotenv
from helpers.storage import load_faiss
import os
from helpers.reranker import create_reranker
from helpers.chain import create_rag_chain

# ------ Page Config -----
st.set_page_config(page_title="Dermatology RAG", layout="wide",initial_sidebar_state="expanded", page_icon=":woman_scientist:")
st.image("C:/Users/Nebolisa Kosiso/dermatology_naive_rag_/helpers/files/skinwlnes.jpeg", width="stretch")
st.title("Dermatology RAG Application")
load_dotenv() #load the groq api key from environment variables

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#-----Session Setup -----
if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.current_chat = None
    st.session_state.chat_counter = 0

#----Sidebar Chat history -------

with st.sidebar:
    st.title("Chat History")

    #new chat
    if st.button("New chat"):
        st.session_state.chat_counter +=1
        new_chat_name = f"Untitled Chat {st.session_state.chat_counter}"
        st.session_state.chats[new_chat_name] = []
        st.session_state.current_chat = new_chat_name

    #show chats as tabs
    if st.session_state.chats:
        chat_list = list(st.session_state.chats.keys())
        selected_chat = st.radio("Chats", chat_list, index=chat_list.index(st.session_state.current_chat))
        st.session_state.current_chat = selected_chat


#--- Load database + RAG chain ---
if "db" not in st.session_state:
    try:
        st.session_state.db = load_faiss()
    except RuntimeError as e:
        st.error(str(e))

if "rag_chain" not in st.session_state and "db" in st.session_state:
    with st.spinner("⚡ Loading Dermatology knowledge base..."):
        #reranker
        reranker = create_reranker()

        #create rag chain
        st.session_state.rag_chain = create_rag_chain(retriever=reranker,api_key=GROQ_API_KEY)
    
#----Main Chat Area
if "rag_chain" in st.session_state and st.session_state.current_chat:
    st.header(f" {st.session_state.current_chat}")
    st.subheader("Ask a question about dermatology")
    
       
    question = st.chat_input("Ask me anything about dermatology...")

    if question:
        with st.spinner("Generating answer..."):
                #save user question
            st.session_state.chats[st.session_state.current_chat].append((question))

                # If it's the first message, rename the chat to this query (shortened)
            if (st.session_state.current_chat.startswith("Untitled") and len(st.session_state.chats[st.session_state.current_chat]) == 1):
                topic_name = (question[:30] + "...") if len(question) > 30 else question
                # Rename the key
                st.session_state.chats[topic_name] = st.session_state.chats.pop(st.session_state.current_chat)
                st.session_state.current_chat = topic_name

            #bot response
            answer = st.session_state.rag_chain.invoke(question)
            #st.write(answer)

            st.session_state.chats[st.session_state.current_chat].append((answer))
    #display chat history
    #st.markdown("### Conversation")
    for message in st.session_state.chats[st.session_state.current_chat]:
        st.write(f"{message}")
else:
    st.info("Start a new chat from the sidebar or wait for the knowledge base to load")
