"""Ingestion job manager with progress updates for API consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import queue
import threading
import uuid
from typing import Any

from src.api_models import IngestSource
from src.services.rag_service import ingest_documents


@dataclass
class IngestJob:
    job_id: str
    source: IngestSource
    status: str = "queued"
    progress: int = 0
    message: str = "Queued"
    result: dict[str, Any] | None = None
    error: str | None = None
    events: queue.Queue[dict[str, Any]] = field(default_factory=queue.Queue)
    done_event: threading.Event = field(default_factory=threading.Event)


class IngestJobManager:
    """Manage ingestion jobs and progress events."""

    def __init__(self) -> None:
        self._jobs: dict[str, IngestJob] = {}
        self._lock = threading.Lock()

    def start_job(self, source: IngestSource) -> IngestJob:
        job_id = str(uuid.uuid4())
        job = IngestJob(job_id=job_id, source=source)
        self._emit(job, status="queued", progress=0, stage="queued", message="Queued")
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> IngestJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run_job(self, job: IngestJob) -> None:
        self._emit(job, status="running", progress=1, stage="started", message="Starting ingestion")

        try:
            result = ingest_documents(
                source=job.source,
                progress_callback=lambda stage, progress, message: self._emit(
                    job,
                    status="running",
                    progress=progress,
                    stage=stage,
                    message=message,
                ),
            )
            job.result = result
            self._emit(job, status="completed", progress=100, stage="completed", message="Ingestion complete")
        except Exception as exc:  # noqa: BLE001
            job.error = str(exc)
            self._emit(job, status="failed", progress=max(job.progress, 1), stage="failed", message=job.error)
        finally:
            job.done_event.set()

    def _emit(self, job: IngestJob, status: str, progress: int, stage: str, message: str) -> None:
        job.status = status
        job.progress = max(0, min(100, progress))
        job.message = message
        payload = {
            "job_id": job.job_id,
            "status": job.status,
            "source": job.source,
            "progress": job.progress,
            "stage": stage,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "result": job.result,
            "error": job.error,
        }
        job.events.put(payload)

