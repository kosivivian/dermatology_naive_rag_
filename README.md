# DERMATOLOGY RAG APPLICATION

This is an interactive Naive Retrieval-Augmented Generation (RAG) system that can answer questions on dermatology topic

## FEATURES

- **Interactive Q\&A:** Ask questions about skin, hair and body care in natural language.
- **Advanced Retrieval:** Uses a sophisticated retrieval pipeline with a reranker (**Cross-Encoder**) for highly accurate context finding.
- **Fast Generation:** Powered by the incredibly fast **Groq** API with Llama 3 for near-instant answers.
- **Open-Source Embeddings:** Utilizes a local, open-source model from Hugging Face for text embeddings.
- **Simple UI:** Built with **Streamlit** for a clean and easy-to-use web interface.

- ## TECH STACK

- - **Framework:** LangChain
- **UI:** Streamlit
- **LLM:** Groq (Llama 3 8B)
- **Embedding Model:** Hugging Face `all-MiniLM-L6-v2`
- **Vector Store:** FAISS
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

## GETTING STARTED

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.8 or higher
- Git

### 1\. Clone the Repository

First, clone the project repository to your local machine

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
### 2\. Create a Python Virtual Environment

It is advisable to use a virtual environment to manage project dependencies

```bash
#create the virtual env
python3 -m venv venv

#activate it
#on macOS/Linux:
source venv/bin/activate

#on windows:
venv\Scripts\activate
```

### 3\. Install Dependencies
This project's dependencies are listed in `requirements.txt`.

**(First, if you haven't created a `requirements.txt` file yet, run this command):

```bash
pip freeze > requirements.txt
```
**(or you can just create a new file manually in your root folder)

Now, install the required packages using pip:
```bash
pip install -r requirements.txt
```

### 4\. Set up Environment Variables
The application requires an API key from Groq to use its LLM.

1.  Create a file named `.env` in the root of your project directory. You can do this by copying the example file:

  ```bash
    cp .env.example .env
  ```

  **(if you do not have a `.env.example` file, simply create a new file named `.env)

2.  Get your API key from the [GroqCloud Console](https://console.groq.com/keys)
  
3.  Open the `.env` file and add your API key like this:
    ```bash
    GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```

### 5\. Create the FAISS Vector Store(do this once)

Run this to build the vector store
```bash
python ingest.py
```

If you wish to add more documents, include it in the `ingest.py` and rerun it

### USAGE
With the environment set up and dependencies installed, you can now run the Streamlit application.

```bash
streamlit run app.py
```
The knowledge base contains dermatology-related articles and publications so the chatbot will only be able to answer questions in that respect
You web browser should automatically open with the application running.
  1.   The chain creation occurs at startup so that the app is always ready with the dermatology docs.
  2.   Type in a question in the input field provided and click 'Enter' on your keyboard to get a response.

### PROJECT STRUCTURE
The project is organized in a modular way to keep the code clean and maintainable.

```
.
├── files/                # contains all dermatology docs
├── helpers/
│   ├── __init__.py       # Makes 'helpers' a Python package
│   ├── ingest.py         # creates faiss index with dermatology docs
│   ├── chain.py          # Creates the final RAG chain with the LLM
│   ├── doc_loading.py    # loads dermatology docs in `files/` and splits them into smaller chunks
│   ├── reranker.py       # Creates the retriever and reranker
│   └── storage.py        # Creates the embedding of the docs, creates FAISS vector store, and appends to it.
├── venv/
├── .env                  # Stores API keys (secret, not committed to git)
├── .gitignore            # Specifies files for git to ignore
├── app.py                # The main Streamlit application file
├── requirements.txt      # Project dependencies
└── README.md             # This file

```

---

## HOW IT WORKS

The application follows a standard RAG pipeline:

1.  **Ingestion:** The system has already been preloaded with dermatology documents that serve as its knowledge base, there is room to add more documents and publications.
2.  **Chunking:**  The `chunker` splits the documents into smaller bits.
3.  **Indexing:**  The `vectorstore` helper uses the `all-MiniLM-L6-v2` model to create a numerical vector (embedding) for each chunk and stores them in a FAISS vector database.
4.  **Retrieval & Reranking:**  When a question is asked, the `retriever` finds the most relevant chunks from FAISS. These results are then re-scored by the Cross-Encoder for higher accuracy.
5.  **Generation:**  The top-ranked chunks and the original question are passed to the Groq LLM within a structured prompt, which then generates the final, grounded answer.

   



## REFERENCES AND CREDITS
- **90 Days to Beautiful Hair 50 Dermatologist-Approved Tips to Unlock The Hair of Your Dreams** -- Aguh,
- **Cosmetic Dermatology Products and Procedures (Zoe Diana Draelos) (Z-Library)**
- **Cosmetics and Dermatologic Problems and Solutions, Third Edition (Zoe Diana Draelos) (Z-Library)**
- **Dermatology for Skin of Color (A. Paul Kelly, Susan Taylor) (Z-Library)**
- **Dr. Susan Taylors Rx for Brown Skin Your Prescription for Flawless Skin, Hair, and Nails (Susan C. Taylor) (Z-Library)**
- **Fundamentals of Ethnic Hair The Dermatologists Perspective (Crystal Aguh Ginette A. Okoye) (Z-Library)**
- **HonK.L.E.LeungAlexan_2010_AcneCausesTreatmentandMyths**
- **Put Your Best Face Forward The Ultimate Guide to Skincare from Acne to Anti-Aging (Sandra Lee) (Z-Library)**
- **The new science of perfect skin understanding skin care myths and miracles for radiant skin at any age (Daniel Yarosh) (Z-Library)**
- **HywelC.Williams_2000_PARTIIDescriptivestud_AtopicDermatitisTheEp**
- **HywelC.Williams_2000_PARTIIIAnalyticalstud_AtopicDermatitisTheEp**
- NSK.AI












- 
