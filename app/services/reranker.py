from sentence_transformers import CrossEncoder
from typing import List, Dict

print("[RERANKER] Loading cross-encoder model...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("[RERANKER] Cross-encoder loaded successfully")


def rerank_chunks(question: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Takes question + 20 candidate chunks from Qdrant.
    Scores each question+chunk pair together.
    Returns top_k most relevant chunks in ranked order.
    """
    if not chunks:
        return []

    # Build [question, chunk_text] pairs for the model
    pairs = [[question, chunk["text"]] for chunk in chunks]

    # Score every pair — higher = more relevant to the question
    scores = reranker.predict(pairs)

    # Attach score to each chunk
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    # Sort highest score first, return top_k
    ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

    print(f"[RERANKER] Reranked {len(chunks)} chunks → keeping top {top_k}")
    for i, chunk in enumerate(ranked[:top_k]):
        print(f"[RERANKER] #{i+1} score={chunk['rerank_score']:.3f} | {chunk['text'][:80]}...")

    return ranked[:top_k]