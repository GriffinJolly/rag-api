from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import List, Dict
import uuid

from app.config import QDRANT_HOST, QDRANT_PORT

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

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
    ensure_collection_exists()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )
    chunks = []
    for hit in results.points:
        payload = hit.payload or {}
        chunks.append(
            {
                "text":        payload.get("text", ""),
                "filename":    payload.get("filename", ""),
                "page_number": payload.get("page_number", 0),
                "score":       hit.score
            }
        )

    return chunks