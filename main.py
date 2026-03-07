from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load clustering model
with open("clusterer.pkl", "rb") as f:
    vectorizer, nmf = pickle.load(f)

# Simple semantic cache
cache = []
cache_embeddings = []

hit_count = 0
miss_count = 0
SIM_THRESHOLD = 0.75


class QueryRequest(BaseModel):
    query: str


def get_dominant_cluster(text):
    X = vectorizer.transform([text])
    topic_dist = nmf.transform(X)
    return int(np.argmax(topic_dist))


@app.post("/query")
async def query(req: QueryRequest):

    global hit_count, miss_count

    query_text = req.query
    query_embedding = embed_model.encode(query_text)

    # Check cache
    if cache_embeddings:
        sims = cosine_similarity([query_embedding], cache_embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score > SIM_THRESHOLD:
            hit_count += 1
            cached = cache[best_idx]

            return {
                "query": query_text,
                "cache_hit": True,
                "matched_query": cached["query"],
                "similarity_score": round(best_score, 4),
                "result": cached["result"],
                "dominant_cluster": cached["cluster"]
            }

    miss_count += 1

    dominant_cluster = get_dominant_cluster(query_text)

    result = f"Query processed. Dominant cluster: {dominant_cluster}"

    cache.append({
        "query": query_text,
        "cluster": dominant_cluster,
        "result": result
    })

    cache_embeddings.append(query_embedding)

    return {
        "query": query_text,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result,
        "dominant_cluster": dominant_cluster
    }


@app.get("/cache/stats")
async def cache_stats():

    total = len(cache)
    hit_rate = hit_count / (hit_count + miss_count) if (hit_count + miss_count) else 0

    return {
        "total_entries": total,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "hit_rate": round(hit_rate, 4)
    }


@app.delete("/cache")
async def clear_cache():

    global cache, cache_embeddings, hit_count, miss_count

    cache = []
    cache_embeddings = []
    hit_count = 0
    miss_count = 0

    return {"message": "Cache flushed."}