"""Pydantic models for the API server."""

from typing import Literal

from pydantic import BaseModel, Field


IngestSource = Literal["all", "local", "notion"]
QueryMode = Literal["hybrid", "vector", "keyword"]
ChatRole = Literal["user", "assistant"]


class ChatMessage(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: QueryMode = "hybrid"
    show_sources: bool = False
    filter_title: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)


class SourceItem(BaseModel):
    source: str
    title: str | None = None
    content_preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)


class StatusResponse(BaseModel):
    documents_directory: str
    notion_configured: bool
    notion_database_id: str
    chunk_count: int
    embedding_model: str
    llm_model: str


class ClearResponse(BaseModel):
    cleared: bool


class IngestStartRequest(BaseModel):
    source: IngestSource = "all"


class IngestStartResponse(BaseModel):
    job_id: str
    status: str


class IngestJobResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    source: IngestSource
    result: dict | None = None
    error: str | None = None


class ModelsResponse(BaseModel):
    current: str
    available: list[str]


class ModelSelectRequest(BaseModel):
    model: str = Field(min_length=1)


class ModelSelectResponse(BaseModel):
    current: str
    available: list[str]

