from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.routers import upload, query

app = FastAPI(
    title="RAG Document Q&A API",
    description="Upload PDFs, ask questions, get answers.",
    version="0.1.0"
)

app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])

@app.get("/", response_class=HTMLResponse)
def home_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>RAG Document Q&A</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f7fb; color: #1f2937; }
            .container { max-width: 900px; margin: 0 auto; padding: 30px; }
            h1 { margin-bottom: 0.5rem; }
            .card { background: white; border-radius: 12px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); padding: 24px; margin-bottom: 24px; }
            label { display: block; margin-top: 16px; font-weight: 600; }
            input[type="text"], input[type="number"], textarea { width: 100%; padding: 10px 12px; border: 1px solid #d1d5db; border-radius: 8px; margin-top: 8px; }
            button { background: #2563eb; border: none; color: white; padding: 12px 18px; border-radius: 8px; cursor: pointer; margin-top: 16px; }
            button:hover { background: #1d4ed8; }
            .hint { color: #4b5563; margin-top: 8px; }
            .output { margin-top: 18px; padding: 16px; border: 1px solid #e5e7eb; border-radius: 10px; background: #f8fafc; }
            .message { margin-top: 12px; padding: 14px 16px; border-radius: 10px; font-weight: 500; }
            .message.success { background: #ecfdf5; border: 1px solid #22c55e; color: #166534; }
            .message.error { background: #fee2e2; border: 1px solid #ef4444; color: #991b1b; }
            .message.info { background: #e0f2fe; border: 1px solid #38bdf8; color: #0369a1; }
            .source-card { margin-top: 12px; padding: 12px 14px; border: 1px solid #e5e7eb; border-radius: 10px; background: #ffffff; }
            .source-card strong { display: block; margin-bottom: 8px; }
            .answer-text { white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RAG Document Q&A</h1>
            <p class="hint">Upload a PDF document first, then ask a question about its contents.</p>

            <div class="card">
                <h2>1. Upload a PDF document</h2>
                <p class="hint">This sends the file to <code>/api/v1/upload</code> and starts background processing.</p>
                <form id="upload-form" method="post" action="/api/v1/upload" enctype="multipart/form-data">
                    <label for="pdf-file">PDF file</label>
                    <input id="pdf-file" name="file" type="file" accept="application/pdf" required />
                    <button type="submit">Upload PDF</button>
                </form>
                <div id="upload-result" class="output"></div>
            </div>

            <div class="card">
                <h2>2. Ask a question</h2>
                <p class="hint">This sends the question to <code>/api/v1/query</code> and returns an answer from the uploaded document.</p>
                <form id="query-form" method="post" action="/api/v1/query">
                    <label for="question">Question</label>
                    <textarea id="question" name="question" rows="3" placeholder="What is the main topic?" required></textarea>
                    <label for="top_k">Top K</label>
                    <input id="top_k" name="top_k" type="number" min="1" max="20" value="5" required />
                    <button type="submit">Submit Query</button>
                </form>
                <div id="query-result" class="output"></div>
            </div>

            <div class="card">
                <h2>Tips</h2>
                <ul>
                    <li>Upload a PDF only once; processing runs in the background.</li>
                    <li>If the document is still processing, queries may return "No documents found".</li>
                    <li>Use the OpenAPI docs at <a href="/docs">/docs</a> for raw API examples.</li>
                </ul>
            </div>
        </div>

        <script>
            const uploadForm = document.getElementById('upload-form');
            const uploadResult = document.getElementById('upload-result');
            const queryForm = document.getElementById('query-form');
            const queryResult = document.getElementById('query-result');

            function renderMessage(container, text, type = 'info') {
                container.innerHTML = `<div class="message ${type}">${text}</div>`;
            }

            function renderQueryResult(data) {
                if (!data || typeof data !== 'object') {
                    queryResult.innerHTML = `<div class="message error">Unexpected response format</div>`;
                    return;
                }

                if (data.error || data.detail) {
                    renderMessage(queryResult, `Error: ${data.error || data.detail}`, 'error');
                    return;
                }

                let html = `<div class="message success">Answer received</div>`;
                html += `<div class="output answer-text"><strong>Answer:</strong> ${data.answer}</div>`;
                if (Array.isArray(data.sources) && data.sources.length) {
                    html += '<h3>Sources</h3>';
                    data.sources.forEach((source, index) => {
                        html += `
                            <div class="source-card">
                                <strong>${index + 1}. ${source.filename} — page ${source.page_number}</strong>
                                <div>${source.text}</div>
                                <div style="margin-top:8px;font-size:0.9rem;color:#475569;">Score: ${source.rerank_score}</div>
                            </div>`;
                    });
                } else {
                    html += '<div class="message info">No sources were returned.</div>';
                }
                queryResult.innerHTML = html;
            }

            async function parseJsonResponse(response) {
                const text = await response.text();
                try {
                    return JSON.parse(text);
                } catch (err) {
                    return { error: 'Invalid JSON response', detail: text };
                }
            }

            uploadForm.addEventListener('submit', async (event) => {
                event.preventDefault();
                renderMessage(uploadResult, 'Uploading PDF...', 'info');
                const formData = new FormData(uploadForm);
                try {
                    const response = await fetch('/api/v1/upload', { method: 'POST', body: formData });
                    const data = await parseJsonResponse(response);
                    if (response.ok) {
                        renderMessage(uploadResult, `Upload accepted: ${data.filename}. Processing in background.`, 'success');
                    } else {
                        renderMessage(uploadResult, `Upload failed: ${data.detail || JSON.stringify(data)}`, 'error');
                    }
                } catch (error) {
                    renderMessage(uploadResult, `Upload failed: ${error.message || error}`, 'error');
                }
            });

            queryForm.addEventListener('submit', async (event) => {
                event.preventDefault();
                renderMessage(queryResult, 'Submitting query... Please wait.', 'info');
                const formData = new FormData(queryForm);
                const payload = { question: formData.get('question'), top_k: Number(formData.get('top_k')) };
                try {
                    const response = await fetch('/api/v1/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
                    const data = await parseJsonResponse(response);
                    if (!response.ok) {
                        renderMessage(queryResult, `Query failed: ${data.detail || data.error || JSON.stringify(data)}`, 'error');
                        return;
                    }
                    renderQueryResult(data);
                } catch (error) {
                    renderMessage(queryResult, `Query failed: ${error.message || error}`, 'error');
                }
            });
        </script>
    </body>
    </html>
    """
