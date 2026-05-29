import pypdf
import io
from typing import List, Dict

def extract_text_from_pdf(file_bytes: bytes) -> List[Dict]:
    pages = []
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"text": text, "page_number": page_num + 1})
    return pages

def chunk_text(pages: List[Dict], chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    chunks = []
    chunk_id = 0
    for page in pages:
        words = page["text"].split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunks.append({
                "chunk_id":    chunk_id,
                "text":        " ".join(chunk_words),
                "page_number": page["page_number"],
                "word_count":  len(chunk_words)
            })
            chunk_id += 1
            start += chunk_size - overlap
    return chunks