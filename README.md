# 🎬 Advanced RAG Movie Recommendation System

> **Two-Stage Retrieval + Cross-Encoder Reranking + 4-bit Quantized Mistral-7B**

An end-to-end **Retrieval-Augmented Generation (RAG)** system for movie recommendations based on movie plot descriptions.

The system combines **semantic vector retrieval**, **Cross-Encoder reranking**, and a **quantized Mistral-7B language model** to retrieve relevant movies and generate context-grounded recommendations.

---

## 🚀 Project Overview

Traditional semantic search retrieves documents based on embedding similarity, but the initial ranking may not always place the most relevant results first.

This project addresses that problem using a **two-stage retrieval architecture**:

```text
User Query
    │
    ▼
Bi-Encoder Embedding
    │
    ▼
ChromaDB Vector Search
    │
    ▼
Top-10 Candidate Movies
    │
    ▼
MonoBERT Cross-Encoder Reranking
    │
    ▼
Top-5 Relevant Movies
    │
    ▼
Mistral-7B-Instruct
    │
    ▼
Context-Grounded Recommendation
```

The goal is to combine the **speed of dense retrieval** with the **higher relevance scoring of Cross-Encoder reranking**, followed by natural-language generation using an LLM.

---

## ✨ Key Features

* 🔎 **Semantic Search** using dense vector embeddings
* ⚡ **Two-stage retrieval pipeline**
* 🧠 **Cross-Encoder reranking** with MonoBERT
* 🗄️ **ChromaDB** vector database
* 🤖 **Mistral-7B-Instruct** for answer generation
* 📦 **4-bit NF4 quantization** using BitsAndBytes
* 🎯 Context-grounded movie recommendations
* 🚀 GPU-accelerated embedding generation
* 📊 Clear comparison between initial retrieval and reranked results

---

## 🏗️ Architecture

### Stage 1 — Dense Retrieval

Movie plot descriptions are converted into dense vector representations using:

**`all-MiniLM-L6-v2`**

The embeddings are stored in **ChromaDB** using cosine similarity.

For every user query, the system retrieves the top **10 candidate movie chunks** based on semantic similarity.

---

### Stage 2 — Cross-Encoder Reranking

The retrieved candidates are then passed to:

**`castorini/monobert-large-msmarco-finetune-only`**

Unlike a Bi-Encoder, the Cross-Encoder processes the **query and candidate document together**, allowing deeper interaction between the two texts.

The initial 10 candidates are rescored and reordered, and the top **5 candidates** are selected for generation.

---

### Stage 3 — LLM Generation

The reranked movie plots are provided as context to:

**`mistralai/Mistral-7B-Instruct-v0.1`**

The model generates a natural-language recommendation using the retrieved movie context.

To reduce GPU memory requirements, the model is loaded using:

* 4-bit quantization
* NF4 quantization
* Double quantization
* `float16` computation

---

## 📊 Dataset

The project uses the **TMDB-5000 Movies dataset**, containing movie titles and plot overviews.

Only the following fields are used:

| Field            | Purpose                                |
| ---------------- | -------------------------------------- |
| `original_title` | Movie identification                   |
| `overview`       | Text used for retrieval and generation |

After removing missing values, the notebook works with:

**4,800 movies**

The processed plot data is then transformed into **4,799 indexed text chunks**.

---

## 🔄 Data Processing Pipeline

```text
TMDB-5000 Dataset
        │
        ▼
Select Movie Title + Overview
        │
        ▼
Remove Missing Values
        │
        ▼
NLTK Text Chunking
        │
        ▼
all-MiniLM-L6-v2
        │
        ▼
384-dimensional Embeddings
        │
        ▼
ChromaDB
```

The movie overviews are split using `NLTKTextSplitter` while preserving sentence-level structure.

---

## 🧪 Example

### User Query

> What are some good science fiction movies about space exploration, astronauts, or alien planets?

### Initial Retrieval — Top 10

The Bi-Encoder retrieves candidates including:

| Initial Rank | Movie                     |
| -----------: | ------------------------- |
|            1 | Interstellar              |
|            2 | In the Shadow of the Moon |
|            3 | Galaxy Quest              |
|            4 | Gattaca                   |
|            5 | You Only Live Twice       |
|            6 | Aliens in the Attic       |
|            7 | Red Planet                |
|            8 | Mars Attacks!             |

---

### After MonoBERT Reranking — Top 5

The Cross-Encoder changes the ranking based on query-document relevance:

| Final Rank | Movie                         | Initial Rank |
| ---------: | ----------------------------- | -----------: |
|          1 | **Red Planet**                |            7 |
|          2 | **Interstellar**              |            1 |
|          3 | **Galaxy Quest**              |            3 |
|          4 | **In the Shadow of the Moon** |            2 |
|          5 | **Mars Attacks!**             |            8 |

This demonstrates the main purpose of the second retrieval stage: **the Cross-Encoder can reorder candidates retrieved by the initial dense search.**

---

## 🤖 Final Generation

The top reranked movie plots are passed to Mistral-7B together with the user's query.

The prompt instructs the model to:

1. Use the provided movie context.
2. Recommend movies relevant to the user's request.
3. Explain why each movie matches the request.
4. Base the response on the retrieved plot information.

Example output begins with recommendations such as:

* **Red Planet**
* **Interstellar**
* **Galaxy Quest**
* **In the Shadow of the Moon**
* **Mars Attacks!**

---

## 🛠️ Tech Stack

| Technology                | Purpose                   |
| ------------------------- | ------------------------- |
| Python                    | Core implementation       |
| Pandas                    | Data processing           |
| PyTorch                   | Deep learning backend     |
| Sentence Transformers     | Text embeddings           |
| `all-MiniLM-L6-v2`        | Bi-Encoder retrieval      |
| ChromaDB                  | Vector database           |
| MonoBERT                  | Cross-Encoder reranking   |
| Hugging Face Transformers | Model loading & inference |
| Mistral-7B-Instruct       | Text generation           |
| BitsAndBytes              | 4-bit quantization        |
| NLTK                      | Text processing           |
| LangChain Text Splitters  | Document chunking         |

---

## 📁 Project Structure

```text
advanced-rag-movie-recommendation/
│
├── two_stage_movie_rag.ipynb
└── README.md
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ramezaboud/advanced-rag-movie-recommendation.git
cd advanced-rag-movie-recommendation
```

### 2. Install the required dependencies

The notebook installs the main dependencies automatically, including:

```bash
pip install -U sentence-transformers
pip install -U chromadb
pip install -q langchain-text-splitters nltk
pip install -U bitsandbytes accelerate
```

### 3. Open the notebook

Run:

```text
two_stage_movie_rag.ipynb
```

The notebook downloads the required dataset and Hugging Face models during execution.

### 4. GPU Recommendation

A CUDA-enabled GPU is recommended because the pipeline loads:

* MonoBERT
* Mistral-7B
* Sentence Transformer embeddings

Mistral-7B is loaded using 4-bit quantization to reduce GPU memory requirements.

---

## 🧩 RAG Components

### Retriever

```text
all-MiniLM-L6-v2
```

Fast Bi-Encoder used to generate embeddings and perform initial candidate retrieval.

### Vector Database

```text
ChromaDB
```

Stores movie chunks and their corresponding embeddings using cosine similarity.

### Reranker

```text
castorini/monobert-large-msmarco-finetune-only
```

A Cross-Encoder used to assign relevance scores to the retrieved candidates.

### Generator

```text
mistralai/Mistral-7B-Instruct-v0.1
```

Generates the final recommendation from the reranked movie context.

---

## 💡 Why Two-Stage Retrieval?

A single vector search stage is efficient, but embedding similarity does not always produce the ideal ranking.

This architecture separates retrieval into two stages:

```text
Stage 1
Fast candidate retrieval
        ↓
Top 10

Stage 2
Deep query-document relevance scoring
        ↓
Top 5
```

This allows the system to first retrieve a broader candidate set and then apply a more computationally expensive relevance model to refine the ranking.

---

## ⚠️ Limitations

This project is designed as an **end-to-end RAG demonstration and portfolio project**, rather than a production recommendation platform.

Current limitations include:

* Recommendations are based primarily on movie plot descriptions.
* The system does not use user history or personalized preferences.
* Retrieval quality is demonstrated through a single example query.
* No formal retrieval benchmark such as Recall@K, MRR, or NDCG is included.
* LLM outputs are generated from retrieved context but are not guaranteed to be completely hallucination-free.
* The current implementation runs as a notebook-based pipeline rather than a deployed API or application.

---

## 🔮 Future Improvements

Potential extensions include:

* 📈 Evaluate retrieval with **Recall@K, MRR, and NDCG**
* 🧪 Compare Bi-Encoder retrieval against Bi-Encoder + Cross-Encoder reranking
* 🎭 Add movie genres, cast, directors, and keywords to the retrieval context
* 👤 Add personalized recommendations based on user preferences
* 💾 Persist the ChromaDB collection
* 🌐 Build a REST API using FastAPI
* 🎨 Create an interactive Streamlit interface
* ⚙️ Optimize inference latency
* 🔍 Add query expansion and hybrid retrieval
* 📊 Add RAG evaluation using dedicated evaluation frameworks

---

## 📌 Project Highlights

**Architecture**

```text
Bi-Encoder Retrieval
        +
Cross-Encoder Reranking
        +
Quantized LLM Generation
```

**Retrieval:** `all-MiniLM-L6-v2`
**Vector DB:** `ChromaDB`
**Reranker:** `MonoBERT`
**Generator:** `Mistral-7B-Instruct`
**Quantization:** `4-bit NF4`

---

## 👨‍💻 Author

**Ramez Fawzy**

Machine Learning Engineer | AI & NLP Enthusiast

GitHub: [@ramezaboud](https://github.com/ramezaboud)

---

## ⭐ If you found this project useful

Feel free to explore the notebook, experiment with different queries, and extend the retrieval and evaluation pipeline.
