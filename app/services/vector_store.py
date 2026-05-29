from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import List, Dict
import uuid

client = QdrantClient(host="qdrant", port=6333)

COLLECTION_NAME = "documents"
VECTOR_SIZE = 384


def ensure_collection_exists():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        print(f"[QDRANT] Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print("[QDRANT] Collection created")
    else:
        print(f"[QDRANT] Collection '{COLLECTION_NAME}' already exists")


def store_chunks(chunks: List[Dict], vectors: List[List[float]], filename: str):
    ensure_collection_exists()
    points = []
    for chunk, vector in zip(chunks, vectors):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text":        chunk["text"],
                "page_number": chunk["page_number"],
                "chunk_id":    chunk["chunk_id"],
                "filename":    filename,
                "word_count":  chunk["word_count"]
            }
        )
        points.append(point)
    print(f"[QDRANT] Storing {len(points)} points...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[QDRANT] Successfully stored {len(points)} chunks from {filename}")


def search_similar_chunks(query_vector: List[float], top_k: int = 20) -> List[Dict]:
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    return [
        {
            "text":        hit.payload["text"],
            "filename":    hit.payload["filename"],
            "page_number": hit.payload["page_number"],
            "score":       hit.score
        }
        for hit in results
    ]