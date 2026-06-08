from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ValidationError
from typing import List, Dict
import httpx

from app.config import OLLAMA_URL, OLLAMA_MODEL
from app.services.embedder import embed_single_text
from app.services.vector_store import search_similar_chunks
from app.services.reranker import rerank_chunks

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class SourceChunk(BaseModel):
    text: str
    filename: str
    page_number: int
    rerank_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]


def build_prompt(question: str, chunks: List[Dict]) -> str:
    """
    Assembles the context + question into a prompt.
    The chunks are injected so Llama answers from
    YOUR document, not its training data.
    """
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Passage {i+1} — {chunk['filename']}, page {chunk['page_number']}]\n"
            f"{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    return f"""You are a helpful assistant that answers questions based strictly on the provided document passages.

PASSAGES:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer using ONLY the information in the passages above
- If the answer is not in the passages, say "I could not find this in the document"
- Be concise and direct
- Reference which passage supports your answer

ANSWER:"""


async def call_ollama(prompt: str) -> str:
    """
    Sends prompt to local Ollama server.
    Timeout is 120s — local models are slower than OpenAI.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers questions from documents."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": False
                }
            )
            response.raise_for_status()
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Ollama connection failed: {exc}"
            )
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            raise HTTPException(
                status_code=502,
                detail=f"Ollama request failed ({exc.response.status_code}): {detail}"
            )

        try:
            data = response.json()
        except ValueError:
            raise HTTPException(
                status_code=502,
                detail="Invalid JSON response returned by Ollama"
            )
        answer = None

        if isinstance(data, dict):
            if "message" in data and isinstance(data["message"], dict):
                answer = data["message"].get("content")
            elif "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                first_choice = data["choices"][0]
                if isinstance(first_choice, dict):
                    if "message" in first_choice and isinstance(first_choice["message"], dict):
                        answer = first_choice["message"].get("content")
                    else:
                        answer = first_choice.get("content")
            elif "output" in data:
                output = data["output"]
                if isinstance(output, list) and output:
                    first_output = output[0]
                    if isinstance(first_output, dict):
                        answer = first_output.get("content")
                    else:
                        answer = str(first_output)

        if not isinstance(answer, str) or not answer.strip():
            raise HTTPException(
                status_code=502,
                detail=f"Unexpected Ollama response format: {data}"
            )

        return answer.strip()


async def parse_query_request(request: Request) -> QueryRequest:
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            payload = await request.json()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
    else:
        form = await request.form()
        payload = dict(form)

    try:
        return QueryRequest(**payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors())


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: Request):
    query = await parse_query_request(request)

    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    print(f"\n[QUERY] Question: {query.question}")

    try:
        # Step 1 — embed the question into a vector
        print("[QUERY] Embedding question...")
        try:
            question_vector = embed_single_text(query.question)
        except Exception as exc:
            print(f"[QUERY] Embedding failed: {exc}")
            raise HTTPException(status_code=503, detail="Embedding service unavailable")

        # Step 2 — find top 20 similar chunks from Qdrant
        print("[QUERY] Searching Qdrant...")
        try:
            candidate_chunks = search_similar_chunks(question_vector, top_k=20)
        except Exception as exc:
            print(f"[QUERY] Qdrant search failed: {exc}")
            raise HTTPException(status_code=503, detail="Vector store unavailable")

        if not candidate_chunks:
            raise HTTPException(
                status_code=404,
                detail="No documents found. Please upload a PDF first."
            )

        # Step 3 — rerank 20 → top 5
        print("[QUERY] Reranking...")
        try:
            top_chunks = rerank_chunks(query.question, candidate_chunks, top_k=query.top_k)
        except Exception as exc:
            print(f"[QUERY] Reranking failed: {exc}")
            raise HTTPException(status_code=503, detail="Reranker service unavailable")

        # Step 4 — build prompt and call Llama 3 via Ollama
        print("[QUERY] Calling Ollama...")
        prompt = build_prompt(query.question, top_chunks)
        answer = await call_ollama(prompt)
        print(f"[QUERY] Answer: {answer[:100]}...")

        # Step 5 — return answer + sources used
        return QueryResponse(
            question=query.question,
            answer=answer,
            sources=[
                SourceChunk(
                    text=chunk["text"],
                    filename=chunk["filename"],
                    page_number=chunk["page_number"],
                    rerank_score=chunk["rerank_score"]
                )
                for chunk in top_chunks
            ]
        )
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[QUERY] Unexpected error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the query."
        )