from sentence_transformers import SentenceTransformer
from typing import List, Dict

print("[EMBEDDER] Loading sentence-transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("[EMBEDDER] Model loaded successfully")


def embed_chunks(chunks: List[Dict]) -> List[List[float]]:
    texts = [chunk["text"] for chunk in chunks]
    print(f"[EMBEDDER] Embedding {len(texts)} chunks...")
    vectors = model.encode(texts, show_progress_bar=True)
    print(f"[EMBEDDER] Done. Each vector has {len(vectors[0])} dimensions")
    return [vector.tolist() for vector in vectors]


def embed_single_text(text: str) -> List[float]:
    vector = model.encode(text)
    return vector.tolist()