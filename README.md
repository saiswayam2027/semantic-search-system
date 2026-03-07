# Semantic Search System with Fuzzy Clustering and Semantic Cache

## Overview

This project implements a **lightweight semantic search system** built on the **20 Newsgroups dataset**.  
The system performs semantic retrieval using vector embeddings, fuzzy clustering, and a semantic caching layer exposed through a FastAPI service.

The objective is to design a system capable of understanding **semantic similarity between queries**, reducing redundant computations through an intelligent caching mechanism.

Dataset:  
https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

---

# System Architecture

The system consists of four major components:

1. **Embedding + Vector Database**
2. **Fuzzy Clustering**
3. **Semantic Cache**
4. **FastAPI API Service**

Pipeline:

```
User Query
     ↓
Query Embedding
     ↓
Semantic Cache Lookup
     ↓
(Cache Miss)
Cluster Identification
     ↓
Vector Search (ChromaDB)
     ↓
Results Returned + Stored in Cache
```

---

# Part 1 — Embeddings & Vector Database

Documents are converted into vector embeddings using the **Sentence Transformers model**:

```
all-MiniLM-L6-v2
```

Reasons for choosing this model:

- Lightweight and fast
- Good semantic representation quality
- Suitable for real-time inference

Embeddings are stored in **ChromaDB**, a lightweight vector database that supports efficient similarity search.

---

# Part 2 — Fuzzy Clustering

The semantic structure of the corpus is discovered using:

```
TF-IDF + Non-negative Matrix Factorization (NMF)
```

Unlike hard clustering, NMF produces **soft topic distributions**, allowing a document to belong to multiple clusters.

Example cluster distribution:

```
Document Topic Distribution

Cluster 2 → 0.42
Cluster 9 → 0.33
Cluster 13 → 0.21
Cluster 5 → 0.04
```

This reflects the overlapping semantic nature of the dataset.

---

# Part 3 — Semantic Cache

Traditional caches only work when queries match exactly.

This project implements a **semantic cache** that recognizes queries with similar meaning.

Example:

Query 1:
```
Tell me about space exploration missions
```

Query 2:
```
NASA missions and space travel
```

Even though phrasing differs, the system detects semantic similarity and retrieves cached results.

Similarity is computed using:

```
Cosine similarity between query embeddings
```

The cache tracks:

```
total_entries
hit_count
miss_count
hit_rate
```

No external caching systems (Redis / Memcached) were used.

---

# Part 4 — FastAPI Service

The system exposes a REST API using **FastAPI**.

## POST /query

Request:

```json
{
  "query": "Tell me about space exploration missions"
}
```

Response:

```json
{
  "query": "...",
  "cache_hit": true,
  "matched_query": "...",
  "similarity_score": 0.91,
  "result": "...",
  "dominant_cluster": 3
}
```

---

## GET /cache/stats

Returns cache statistics.

Example:

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

---

## DELETE /cache

Clears the cache and resets statistics.

Response:

```json
{
  "message": "Cache flushed."
}
```

---

# Installation

### 1. Clone repository

```
git clone <repo-url>
cd semantic-search-system
```

### 2. Create virtual environment

```
python -m venv venv
source venv/bin/activate
```

Windows:

```
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

# Running the System

Run the notebook and execute cells sequentially.

The FastAPI server will start automatically when the server cell is executed.

Once running, the API is available at:

```
http://localhost:8001
```

Swagger documentation:

```
http://localhost:8001/docs
```

---

# Example Queries

```
Tell me about space exploration missions
NASA missions and space travel
How does encryption work?
What are graphics cards used for?
```

---

# Technologies Used

- Python
- Sentence Transformers
- ChromaDB
- Scikit-Learn
- NMF Topic Modeling
- FastAPI
- NumPy / Pandas

---

# Key Highlights

- Semantic query understanding
- Fuzzy clustering of documents
- Custom semantic cache (no external libraries)
- Real-time API service
- Vector database integration

---

# Author

Sai Swayam Pradhan  
B.Tech Computer Science — VIT Chennai
