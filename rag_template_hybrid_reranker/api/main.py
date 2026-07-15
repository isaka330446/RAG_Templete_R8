# reranker入りRAG APIとログ確認APIをFastAPIで公開します。
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from api.config import load_settings
from api.schemas import AskRequest, AskResponse
from api.rag_engine import RAGEngine

app = FastAPI(title="RAG Template API", version="1.0.0")
settings = load_settings()
api_settings = settings.get("api", {})

app.add_middleware(
    CORSMiddleware,
    allow_origins=api_settings.get("cors_allow_origins", ["http://localhost:8501", "http://127.0.0.1:8501"]),
    allow_credentials=bool(api_settings.get("allow_credentials", False)),
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = RAGEngine()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = engine.ask(
        question=req.question,
        corpus_ids=req.corpus_ids,
        top_k=req.top_k,
        show_debug=req.show_debug,
        session_id=req.session_id,
        history=[m.model_dump() for m in req.history],
    )
    return result


@app.get("/logs/recent")
def recent_logs(limit: int = Query(default=50, ge=1, le=500)):
    return {"logs": engine.log_store.recent(limit=limit)}
