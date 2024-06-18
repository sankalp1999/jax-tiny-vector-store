import time
from functools import partial
from typing import List
import jax
from jax import lax
import jax.numpy as jnp
from sentence_transformers import SentenceTransformer
from documents import document

def timeit(func):
    def inner(*args, **kwargs):
        t_start = time.time()
        res = func(*args, **kwargs)
        t_exec = time.time() - t_start
        return res, t_exec * 1000
    return inner

# @jax.jit
def cosine_similarity(v1, v2):
    return jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2))

# @partial(jax.jit, static_argnums=(2,))
# def get_topk_similar(store, query, k):
#     scores = cosine_similarity(store, query)
#     top_indices = jnp.argsort(scores)[-k:][::-1]
#     return top_indices, scores[top_indices]  # Return indices and scores directly

def get_topk_similar(store, query, k):
    scores = cosine_similarity(store, query)
    topk_values, topk_indices = lax.top_k(scores, k) # this is recommended
    return topk_indices[::-1], topk_values[::-1]

class Vectorstore:
    def __init__(self, docs: List[str], embedder: SentenceTransformer):
        self.docs = docs
        self.embedder = embedder
        encoded_docs = [embedder.encode(doc) for doc in docs] 
        self._store = jnp.array(encoded_docs) # batch is actually slow on cpu, might be faster on gpu

    @timeit
    def search(self, query: str, k: int = 10):
        q_emb = self.embedder.encode(query)
        topk_indices, scores = self._search(q_emb, k)
        return [self.docs[int(idx)] for idx in topk_indices], scores.tolist() 

    @partial(jax.jit, static_argnums=(0, 2))
    def _search(self, query, k):
        topk_indices, scores = get_topk_similar(self._store, query, k)
        return topk_indices, scores

    def __repr__(self):
        return f"Vectorstore(embedder={self.embedder})"

# Usage
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

docs = document.split('\n')
print(f"Building vectorstore for {len(docs)} documents...")
print()

vs = Vectorstore(docs, embedder=model)

query = "What did emma do in this story?"
(topk_docs, scores), exectime = vs.search(query)

print(f"\nMost similar documents: {list(topk_docs)[0]}")
print(f"Scores (higher is better): {list(scores)}")
print(f"\nSearch time: {exectime} ms")

