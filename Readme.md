# RAG Explorer

A Retrieval-Augmented Generation (RAG) pipeline that lets you upload any PDF and query it using four different retrieval strategies. Built to demonstrate retriever tradeoffs in a production-style pipeline.

---

## What This Project Does

Upload a PDF, select a retriever, ask a question. The app retrieves relevant chunks and generates an answer using GPT-4o-mini. You can switch retrievers and observe how retrieval quality changes.

---

## Tech Stack

- LangChain — pipeline orchestration
- ChromaDB — vector store with disk persistence
- OpenAI — embeddings (text-embedding-ada-002) and generation (gpt-4o-mini)
- PyMuPDF — PDF loading
- Streamlit — frontend

---

## Retriever Implementations

### 1. Similarity (Baseline)
Standard cosine similarity search. Returns top k=3 chunks by similarity score. Used as baseline for comparison.

### 2. MMR (Maximal Marginal Relevance)
Reduces redundancy by diversifying retrieved chunks. Fetches fetch_k=10 candidates, returns k=3 with maximum diversity (lambda_mult=0.7).

**Observation:** On a single-topic corpus (one focused research paper), MMR showed no meaningful improvement over baseline. All chunks are semantically close, leaving no room for diversification. MMR shows value on multi-document, multi-topic corpora.

### 3. Parent Document Retriever
Uses two splitters — child chunks (200 tokens) for precise retrieval, parent chunks (1000 tokens) for context-rich generation. Child chunks are stored in ChromaDB. Parent chunks are stored in an InMemoryStore and passed to the LLM.

**Observation:** Clear improvement over baseline. Retrieved chunks were significantly larger and more coherent, spanning different sections of the document. Boundary cut problem largely resolved.

**Known limitation:** Parent chunk size of 1000 tokens occasionally cuts mid-sentence. Experimenting with 1200 tokens may improve boundary behavior.

### 4. Multi Query Retriever
Generates 3 rephrasings of the user query using a custom prompt — one definition-based, one mechanism-based, one consequence-based. Unions results and deduplicates.

**Observation:** Default LLM rephrasings are too semantically similar, hitting the same chunks. Fixed with a custom prompt enforcing different query angles. On a single-topic corpus, still converges to similar chunks. Shows value on diverse multi-document corpora.

---

## Architectural Decisions

**Chunk size: 500 tokens, overlap: 100 tokens**
Dense academic content requires capturing complete ideas per chunk. Too small (100 tokens) scatters information. Too large (2000 tokens) dilutes similarity scores, hurting retrieval precision.

**Persistent ChromaDB**
Embeddings are persisted to disk to avoid re-embedding on every run. OpenAI embedding costs are non-trivial at scale.

**PyMuPDF over pypdf**
Faster, handles malformed PDFs more gracefully, cleaner text extraction on academic papers.

**gpt-4o-mini**
Cost-efficient for question answering tasks. Temperature set to 0 for deterministic, factual answers.

---

## Known Issues

### RAGAS Dependency Conflict
RAGAS (RAG evaluation framework) was planned for automated evaluation using faithfulness, answer relevancy, context precision, and context recall metrics.

**Issue:** `ragas>=0.2.0` conflicts with `langchain-community>=0.3.0` due to deprecated VertexAI import paths in langchain_community.

```
ImportError: cannot import name 'create_model' from 'langchain_core.utils.pydantic'
```
**Workaround attempted:** Downgrading to `ragas==0.1.21` triggered a metaclass conflict deep in langchain_community's VertexAI implementation.

**Current status:** RAGAS evaluation skipped. Manual evaluation dataset built instead with 3 question-answer pairs grounded in document content.

**Resolution path:** Isolate RAGAS in a separate virtual environment with pinned langchain==0.1.20 and langchain-community==0.0.38.

### Streamlit Rerun Duplicate Chunks
Streamlit reruns the entire script on every user interaction. Without session state, `build_vectorstore` was called multiple times, appending duplicate chunks to ChromaDB.

**Fix:** Wrapped PDF processing and vectorstore creation in `st.session_state` with a file name check to ensure processing happens only once per uploaded file.

---

## Setup

```bash
git clone <your-repo-url>
cd rag-explorer
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Create a `.env` file:
OPENAI_API_KEY=your_key_here

Run:
```bash
streamlit run app.py
```

---

## Project Structure
```
├── app.py                  # Streamlit frontend
├── rag_pipeline.py         # RAG logic, retrievers
├── rag_project.ipynb       # Experimentation notebook
├── requirements.txt
├── .env                    # Not committed
└── .gitignore
```

---

## What I Learned

- Retriever choice depends on data characteristics, not just query type. MMR and Multi Query showed no improvement on a single-topic corpus but would show clear value on diverse multi-document corpora.
- Chunk size is a decision, not a default. 500 tokens was chosen based on content density, not copied from a tutorial.
- Dependency management in LangChain is a real engineering challenge. The ecosystem moves fast and version conflicts are common in production.
- Streamlit's rerun behavior requires explicit state management for expensive operations like embedding and vectorstore creation.