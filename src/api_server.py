"""FastAPI backend exposing RAG operations for the frontend."""

from __future__ import annotations

import asyncio
import json
import queue

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api_models import (
    ChatRequest,
    ChatResponse,
    ClearResponse,
    IngestJobResponse,
    IngestStartRequest,
    IngestStartResponse,
    SourceItem,
    StatusResponse,
)
from src.ingest_jobs import IngestJobManager
from src.services.rag_service import clear_documents, get_status, query_documents

app = FastAPI(title="RAG Agent API", version="0.1.0")
job_manager = IngestJobManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(**get_status())


@app.post("/clear", response_model=ClearResponse)
def clear() -> ClearResponse:
    return ClearResponse(cleared=clear_documents())


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        answer, docs = query_documents(
            question=payload.question,
            search_mode=payload.mode,
            title_filter=payload.filter_title,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    sources: list[SourceItem] = []
    if payload.show_sources:
        for doc in docs:
            sources.append(
                SourceItem(
                    source=doc.metadata.get("source", "unknown"),
                    title=doc.metadata.get("title"),
                    content_preview=doc.page_content[:220],
                )
            )
    return ChatResponse(answer=answer, sources=sources)


@app.post("/ingest", response_model=IngestStartResponse)
def ingest(payload: IngestStartRequest) -> IngestStartResponse:
    job = job_manager.start_job(source=payload.source)
    return IngestStartResponse(job_id=job.job_id, status=job.status)


@app.get("/ingest/{job_id}", response_model=IngestJobResponse)
def ingest_status(job_id: str) -> IngestJobResponse:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")

    return IngestJobResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        source=job.source,
        result=job.result,
        error=job.error,
    )


@app.get("/ingest/{job_id}/events")
async def ingest_events(job_id: str) -> StreamingResponse:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")

    async def event_stream() -> asyncio.AsyncGenerator[str, None]:
        snapshot = {
            "job_id": job.job_id,
            "status": job.status,
            "source": job.source,
            "progress": job.progress,
            "stage": "snapshot",
            "message": job.message,
            "result": job.result,
            "error": job.error,
        }
        yield f"data: {json.dumps(snapshot)}\n\n"

        while True:
            try:
                event = await asyncio.to_thread(job.events.get, True, 0.5)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                yield ": heartbeat\n\n"

            if job.done_event.is_set() and job.events.empty():
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def run() -> None:
    """Run the API server with uvicorn."""
    import uvicorn

    uvicorn.run("src.api_server:app", host="127.0.0.1", port=8001, reload=False)


if __name__ == "__main__":
    run()

