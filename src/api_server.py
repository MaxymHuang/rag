"""FastAPI backend exposing RAG operations for the frontend."""

from __future__ import annotations

import asyncio
import json
import queue

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api_models import (
    AccessMetadata,
    AdminMigrationRequest,
    AdminMigrationResponse,
    AdminModelsResponse,
    AdminStatusResponse,
    AdminSystemConfigResponse,
    AdminSystemConfigUpdateRequest,
    AdminSystemConfigUpdateResponse,
    ChatRequest,
    ChatResponse,
    ClearResponse,
    IngestJobResponse,
    IngestStartRequest,
    IngestStartResponse,
    ModelSelectRequest,
    ModelSelectResponse,
    ModelsResponse,
    SourceItem,
    StatusResponse,
)
from src.config import (
    AVAILABLE_EMBEDDING_MODELS,
    AVAILABLE_VECTOR_DB_PROVIDERS,
    EMBEDDING_MODEL,
    VECTOR_DB_PROVIDER,
)
from src.ingest_jobs import IngestJobManager
from src.services.model_service import get_models, select_model
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


def _parse_csv_values(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _build_access_metadata() -> AccessMetadata:
    return AccessMetadata(
        access_mode="prepare_rbac_phase",
        requires_auth=False,
        permissions=["admin:read", "admin:write", "admin:ingest", "admin:clear", "admin:system"],
    )


def require_admin_access() -> AccessMetadata:
    """Placeholder admin gate for next-phase RBAC integration."""
    return _build_access_metadata()


def _build_admin_system_config(access: AccessMetadata) -> AdminSystemConfigResponse:
    embedding_options = _parse_csv_values(AVAILABLE_EMBEDDING_MODELS)
    if EMBEDDING_MODEL not in embedding_options:
        embedding_options.append(EMBEDDING_MODEL)

    vector_options_raw = _parse_csv_values(AVAILABLE_VECTOR_DB_PROVIDERS)
    vector_options = [option for option in vector_options_raw if option == "chroma"] or ["chroma"]
    active_vector_provider = VECTOR_DB_PROVIDER if VECTOR_DB_PROVIDER in vector_options else "chroma"

    return AdminSystemConfigResponse(
        embedding_model=EMBEDDING_MODEL,
        embedding_model_options=embedding_options,
        vector_db_provider=active_vector_provider,
        vector_db_provider_options=vector_options,  # type: ignore[arg-type]
        migration_supported=True,
        access=access,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(**get_status())


@app.get("/admin/status", response_model=AdminStatusResponse)
def admin_status(access: AccessMetadata = Depends(require_admin_access)) -> AdminStatusResponse:
    data = get_status()
    return AdminStatusResponse(**data, access=access)


@app.post("/clear", response_model=ClearResponse)
def clear() -> ClearResponse:
    try:
        return ClearResponse(cleared=clear_documents())
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/admin/clear", response_model=ClearResponse)
def admin_clear(_: AccessMetadata = Depends(require_admin_access)) -> ClearResponse:
    try:
        return ClearResponse(cleared=clear_documents())
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        answer, docs = query_documents(
            question=payload.question,
            search_mode=payload.mode,
            title_filter=payload.filter_title,
            history=[{"role": item.role, "content": item.content} for item in payload.history],
            context_sources=payload.context_sources,
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


@app.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    data = get_models()
    return ModelsResponse(current=data["current"], available=data["available"])


@app.get("/admin/models", response_model=AdminModelsResponse)
def admin_models(access: AccessMetadata = Depends(require_admin_access)) -> AdminModelsResponse:
    data = get_models()
    return AdminModelsResponse(current=data["current"], available=data["available"], access=access)


@app.post("/models/select", response_model=ModelSelectResponse)
def models_select(payload: ModelSelectRequest) -> ModelSelectResponse:
    try:
        data = select_model(payload.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModelSelectResponse(current=data["current"], available=data["available"])


@app.post("/admin/models/select", response_model=AdminModelsResponse)
def admin_models_select(
    payload: ModelSelectRequest,
    access: AccessMetadata = Depends(require_admin_access),
) -> AdminModelsResponse:
    try:
        data = select_model(payload.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AdminModelsResponse(current=data["current"], available=data["available"], access=access)


@app.post("/ingest", response_model=IngestStartResponse)
def ingest(payload: IngestStartRequest) -> IngestStartResponse:
    job = job_manager.start_job(source=payload.source)
    return IngestStartResponse(job_id=job.job_id, status=job.status)


@app.post("/admin/ingest", response_model=IngestStartResponse)
def admin_ingest(
    payload: IngestStartRequest,
    _: AccessMetadata = Depends(require_admin_access),
) -> IngestStartResponse:
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


@app.get("/admin/ingest/{job_id}", response_model=IngestJobResponse)
def admin_ingest_status(
    job_id: str,
    _: AccessMetadata = Depends(require_admin_access),
) -> IngestJobResponse:
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


@app.get("/admin/ingest/{job_id}/events")
async def admin_ingest_events(
    job_id: str,
    _: AccessMetadata = Depends(require_admin_access),
) -> StreamingResponse:
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


@app.get("/admin/system/config", response_model=AdminSystemConfigResponse)
def admin_system_config(
    access: AccessMetadata = Depends(require_admin_access),
) -> AdminSystemConfigResponse:
    return _build_admin_system_config(access)


@app.patch("/admin/system/config", response_model=AdminSystemConfigUpdateResponse)
def admin_system_config_update(
    payload: AdminSystemConfigUpdateRequest,
    access: AccessMetadata = Depends(require_admin_access),
) -> AdminSystemConfigUpdateResponse:
    config = _build_admin_system_config(access)
    message_parts: list[str] = []

    if payload.embedding_model and payload.embedding_model != config.embedding_model:
        message_parts.append("Embedding model update is prepared and will be applied in the next phase.")
    if payload.vector_db_provider and payload.vector_db_provider != config.vector_db_provider:
        message_parts.append("Vector DB provider migration is prepared and will be applied in the next phase.")
    if not message_parts:
        message_parts.append("No config changes requested.")

    return AdminSystemConfigUpdateResponse(
        applied=False,
        message=" ".join(message_parts),
        config=config,
    )


@app.post("/admin/system/migrate", response_model=AdminMigrationResponse)
def admin_system_migrate(
    payload: AdminMigrationRequest,
    access: AccessMetadata = Depends(require_admin_access),
) -> AdminMigrationResponse:
    if payload.action == "reindex":
        job = job_manager.start_job(source=payload.source)
        return AdminMigrationResponse(
            started=True,
            message="Reindex migration started using ingest pipeline.",
            job_id=job.job_id,
            access=access,
        )

    target = payload.target_vector_db_provider or "chroma"
    return AdminMigrationResponse(
        started=False,
        message=(
            f"Vector DB migration to '{target}' is planned for the next phase. "
            "Current release includes endpoint scaffolding and access-control hooks."
        ),
        job_id=None,
        access=access,
    )


def run() -> None:
    """Run the API server with uvicorn."""
    import uvicorn

    uvicorn.run("src.api_server:app", host="127.0.0.1", port=8001, reload=False)


if __name__ == "__main__":
    run()

