"""Pydantic models for the API server."""

from typing import Literal

from pydantic import BaseModel, Field


IngestSource = Literal["all", "local", "notion"]
QueryMode = Literal["hybrid", "vector", "keyword"]
ChatRole = Literal["user", "assistant"]
ContextSource = Literal["local", "notion"]
VectorDbProvider = Literal["chroma"]
AdminMigrationAction = Literal["reindex", "vector_db_migration"]


class ChatMessage(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: QueryMode = "hybrid"
    show_sources: bool = False
    filter_title: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)
    context_sources: list[ContextSource] = Field(default_factory=lambda: ["local", "notion"])


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


class AccessMetadata(BaseModel):
    access_mode: str
    requires_auth: bool
    permissions: list[str] = Field(default_factory=list)


class AdminStatusResponse(StatusResponse):
    access: AccessMetadata


class AdminModelsResponse(ModelsResponse):
    access: AccessMetadata


class AdminSystemConfigResponse(BaseModel):
    embedding_model: str
    embedding_model_options: list[str]
    vector_db_provider: VectorDbProvider
    vector_db_provider_options: list[VectorDbProvider]
    migration_supported: bool
    access: AccessMetadata


class AdminSystemConfigUpdateRequest(BaseModel):
    embedding_model: str | None = None
    vector_db_provider: VectorDbProvider | None = None


class AdminSystemConfigUpdateResponse(BaseModel):
    applied: bool
    message: str
    config: AdminSystemConfigResponse


class AdminMigrationRequest(BaseModel):
    action: AdminMigrationAction
    source: IngestSource = "all"
    target_vector_db_provider: VectorDbProvider | None = None


class AdminMigrationResponse(BaseModel):
    started: bool
    message: str
    job_id: str | None = None
    access: AccessMetadata

