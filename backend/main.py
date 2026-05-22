from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from core import orchestrator
from core.schemas import CaseInfo
from services.storage_service import StorageService


app = FastAPI(title="Divorce Camp Multi-Agent Backend")
storage = StorageService()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Hackathon MVP: wildcard CORS is acceptable for local demo.
# Restrict allow_origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health_check():
    return {
        "status": "ok",
        "message": "Divorce Camp Multi-Agent Backend is running",
    }


@app.get("/test")
async def test_screen():
    return FileResponse(
        "static/index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.post("/cases")
async def create_case(case_info: CaseInfo):
    case_id = storage.create_case(case_info.model_dump())
    return {"case_id": case_id}


@app.post("/cases/{case_id}/upload")
async def upload_files(
    case_id: str,
    files: list[UploadFile] = File(...),
    doc_type: str | None = Form(None),
):
    try:
        storage.load_case_info(case_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="case_id not found") from exc

    uploaded = []
    for file in files:
        uploaded.append(storage.save_uploaded_file(case_id, file, doc_type))
    return {"case_id": case_id, "uploaded_files": uploaded}


@app.get("/cases/{case_id}/files")
async def get_uploaded_files(case_id: str):
    try:
        return {"case_id": case_id, "uploaded_files": storage.load_uploaded_files(case_id)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="case_id not found") from exc


@app.post("/legal/init")
async def init_legal_knowledge():
    try:
        return await orchestrator.init_legal_knowledge()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="legal_basis.json not found") from exc


@app.post("/legal/reset")
async def reset_legal_knowledge():
    return await orchestrator.reset_legal_knowledge()


@app.get("/legal/search")
async def search_legal_knowledge(q: str, issue_type: str | None = None, top_k: int = 5):
    matches = await orchestrator.search_legal_knowledge(q, issue_type, top_k)
    return {
        "query": q,
        "issue_type": issue_type,
        "top_k": top_k,
        "match_count": len(matches),
        "matches": matches,
    }


@app.get("/pinecone/status")
async def pinecone_status():
    return await orchestrator.pinecone_status()


@app.post("/cases/{case_id}/run")
async def run_case(case_id: str):
    try:
        report = await orchestrator.run_case(case_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="case_id or required case file not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"pipeline failed: {type(exc).__name__}: {exc}") from exc
    return {
        "case_id": case_id,
        "status": "completed",
        "report": report,
    }


@app.get("/cases/{case_id}/report")
async def get_report(case_id: str):
    try:
        return storage.load_report(case_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="report not found") from exc
