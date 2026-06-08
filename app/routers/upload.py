import base64
from typing import Optional

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Request
from app.services.chunker import extract_text_from_pdf, chunk_text
from app.services.embedder import embed_chunks
from app.services.vector_store import store_chunks

router = APIRouter()


def process_pdf_background(filename: str, file_bytes: bytes):
    print(f"\n[BACKGROUND] Starting to process: {filename}")

    # Step 1 — extract text (Phase 3)
    pages = extract_text_from_pdf(file_bytes)
    print(f"[BACKGROUND] Extracted {len(pages)} pages")

    # Step 2 — chunk text (Phase 3)
    chunks = chunk_text(pages, chunk_size=500, overlap=50)
    print(f"[BACKGROUND] Created {len(chunks)} chunks")

    # Step 3 — embed chunks (Phase 4) ← NEW
    vectors = embed_chunks(chunks)
    print(f"[BACKGROUND] Generated {len(vectors)} vectors")

    # Step 4 — store in Qdrant (Phase 4) ← NEW
    store_chunks(chunks, vectors, filename)

    print(f"[BACKGROUND] Pipeline complete for: {filename}")


@router.post("/upload", status_code=202)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None)
):
    filename = None
    file_bytes = None

    if file is not None:
        filename = file.filename
        file_bytes = await file.read()
    else:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                payload = await request.json()
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid JSON body")

            filename = payload.get("filename")
            content_base64 = payload.get("content_base64")

            if not filename or not content_base64:
                raise HTTPException(
                    status_code=400,
                    detail="JSON upload must include 'filename' and 'content_base64' fields"
                )

            try:
                file_bytes = base64.b64decode(content_base64)
            except (ValueError, TypeError):
                raise HTTPException(
                    status_code=400,
                    detail="content_base64 is not valid Base64"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Upload must be a multipart/form-data file or a JSON payload with base64 content"
            )

    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported right now"
        )

    if not file_bytes or len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    background_tasks.add_task(process_pdf_background, filename, file_bytes)

    return {
        "status":   "accepted",
        "filename": filename,
        "message":  "PDF received. Processing in background."
    }