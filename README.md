# Semantic Search API with Fuzzy Clustering and Semantic Cache

## Overview

This project implements a semantic search system exposed through a FastAPI service.
The system uses sentence embeddings to understand the meaning of queries, clusters them into topics, and uses a semantic cache to reuse results for similar queries.

The goal is to reduce redundant computation by returning cached results for semantically similar queries.

## Features

* Sentence embeddings using SentenceTransformers
* Topic clustering using TF-IDF + NMF
* Semantic cache using cosine similarity
* FastAPI REST API
* Docker support
* Automatic cache statistics

---

## Project Structure

semantic-search-system
│
├── semantic_search.ipynb   # notebook used for model creation
├── main.py                 # FastAPI server implementation
├── requirements.txt        # dependencies
├── Dockerfile              # container configuration
├── README.md               # project documentation
└── .gitignore              # ignored files

Note: `clusterer.pkl` is not included in the repository because of GitHub file size limits.

---

## Installation

Clone the repository:

git clone <your-repository-url>

cd semantic-search-system

Install dependencies:

pip install -r requirements.txt

---

## Running the API

Start the FastAPI server:

uvicorn main:app --host 0.0.0.0 --port 8000

Open the interactive API documentation:

http://localhost:8000/docs

---

## API Endpoints

### Query Endpoint

POST /query

Example request:

{
"query": "Explain NASA missions"
}

Example response:

{
"query": "Explain NASA missions",
"cache_hit": true,
"matched_query": "Tell me about space exploration missions",
"similarity_score": 0.80,
"result": "Query processed. Dominant cluster: 0",
"dominant_cluster": 0
}

---

### Cache Statistics

GET /cache/stats

Returns statistics about cache usage.

---

### Clear Cache

DELETE /cache

Clears the semantic cache.

---

## Running with Docker

Build the container:

docker build -t semantic-search .

Run the container:

docker run -p 8000:8000 semantic-search

Then open:

http://localhost:8000/docs

---

## Technologies Used

* Python
* FastAPI
* SentenceTransformers
* Scikit-learn
* NumPy
* Docker
---

# Author

Sai Swayam Pradhan  
B.Tech Computer Science — VIT Chennai
