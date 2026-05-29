from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
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
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported right now"
        )

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    background_tasks.add_task(process_pdf_background, file.filename, file_bytes)

    return {
        "status":   "accepted",
        "filename": file.filename,
        "message":  "PDF received. Processing in background."
    }