from fastapi import FastAPI
from app.routers import upload

app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload PDFs, ask questions, get answers.",
    version="0.1.0"
)

app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])

@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG API is running"}