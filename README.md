# 🏦 FinSight AI

> A RAG-based Financial Document Q&A Assistant powered by LangChain, FAISS and GPT-4o mini

---

## 📌 What is FinSight AI?

FinSight AI lets you upload any financial PDF — annual reports, earnings call
transcripts, RBI policy documents, SEBI circulars — and ask questions about
it in plain English.

It uses *Retrieval Augmented Generation (RAG)* to find the most relevant
sections of the document and generate accurate, grounded answers using
GPT-4o mini — with page-level source citations so you always know
where the answer came from.

*No hallucination. No guessing. Only answers from your document.*

---

## 🎯 Use Cases

- Analyse company annual reports instantly
- Query RBI / SEBI policy documents
- Extract key insights from earnings call transcripts
- Due diligence on financial filings
- Research any PDF-based financial document

---

## 🖥️ Screenshots

![FinSight AI HomePage](screenshots\homePage.png)

![FinSight AI Demo](screenshots\chat.png)

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o mini |
| Embeddings | OpenAI text-embedding-3-small |
| RAG Framework | LangChain (LCEL) |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| PDF Parsing | LangChain PyPDFLoader |
| Frontend | Streamlit |
| Environment | python-dotenv |

---

## 🏗️ Architecture

User uploads PDF
↓
PyPDFLoader → extracts text page by page
↓
RecursiveCharacterTextSplitter → chunks (1000 chars, 200 overlap)
↓
OpenAI Embeddings → converts chunks to vectors
↓
FAISS Vector Store → stores and indexes all vectors
↓
User asks a question
↓
Question → embedding → FAISS similarity search → Top 4 chunks
↓
Chunks + Question → PromptTemplate → GPT-4o mini
↓
Grounded Answer + Page Citations


## 🚀 Run Locally uisng the below commands in terminal

### 1. Clone the repo

git clone https://github.com/your-username/finsight-ai.git
cd finsight-ai

### 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

### 3. Install dependencies
pip install -r requirements.txt

### 4. Add your OpenAI API key
Create a .env file in the root folder:
OPENAI_API_KEY=sk-your-key-here

### 5. Run the app
streamlit run app.py


## 📁 Project Structure
finsight-ai/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # RAG pipeline (LangChain + FAISS)
├── .env                # API keys (never committed)
├── .gitignore          # Ignores venv, .env, _pycache_
├── requirements.txt    # All dependencies
└── README.md           # This file


## 📖 Key Concepts Used
RAG (Retrieval Augmented Generation) — Grounds LLM answers in your document to eliminate hallucination
LCEL (LangChain Expression Language) — Modern pipe-based chain composition
Vector Embeddings — Semantic search using mathematical similarity instead of keyword matching
FAISS — Millisecond-speed similarity search over thousands of chunks
Chunking with overlap — 200 character overlap ensures answers spanning chunk boundaries are never lost


## 🔮 Future Improvements
[ ] FastAPI backend to expose RAG as a REST API
[ ] Support multiple PDFs simultaneously
[ ] Add conversation memory across questions
[ ] Deploy on AWS / GCP
[ ] Add support for scanned PDFs via OCR
[ ] Streaming responses for better UX


## 👨‍💻 Author
Built by Aanish — Software Engineer transitioning into GenAI/Agentic AI Engineering.
⭐ If you found this useful, give it a star!

---
