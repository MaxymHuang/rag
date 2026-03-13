export type QueryMode = "hybrid" | "vector" | "keyword";
export type IngestSource = "all" | "local" | "notion";
export type ContextSource = "local" | "notion";
export type ChatRole = "user" | "assistant";

export interface SourceItem {
  source: string;
  title?: string | null;
  content_preview: string;
}

export interface ChatResponse {
  answer: string;
  sources: SourceItem[];
}

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export interface ChatRequest {
  question: string;
  mode: QueryMode;
  showSources: boolean;
  filterTitle?: string;
  history?: ChatMessage[];
  contextSources?: ContextSource[];
}

export interface StatusResponse {
  documents_directory: string;
  notion_configured: boolean;
  notion_database_id: string;
  chunk_count: number;
  embedding_model: string;
  llm_model: string;
}

export interface AccessMetadata {
  accessMode: string;
  requiresAuth: boolean;
  permissions: string[];
}

export interface AdminStatusResponse extends StatusResponse {
  access: AccessMetadata;
}

export interface IngestStartResponse {
  job_id: string;
  status: string;
}

export interface IngestEvent {
  job_id: string;
  status: string;
  source: IngestSource;
  progress: number;
  stage: string;
  message: string;
  timestamp?: string;
  result?: Record<string, unknown>;
  error?: string | null;
}

export interface ModelsResponse {
  current: string;
  available: string[];
}

export interface AdminModelsResponse extends ModelsResponse {
  access: AccessMetadata;
}

export type VectorDbProvider = "chroma";

export interface AdminSystemConfigResponse {
  embedding_model: string;
  embedding_model_options: string[];
  vector_db_provider: VectorDbProvider;
  vector_db_provider_options: VectorDbProvider[];
  migration_supported: boolean;
  access: AccessMetadata;
}

export interface AdminSystemConfigUpdateRequest {
  embedding_model?: string;
  vector_db_provider?: VectorDbProvider;
}

export interface AdminSystemConfigUpdateResponse {
  applied: boolean;
  message: string;
  config: AdminSystemConfigResponse;
}

export type AdminMigrationAction = "reindex" | "vector_db_migration";

export interface AdminMigrationRequest {
  action: AdminMigrationAction;
  source: IngestSource;
  target_vector_db_provider?: VectorDbProvider;
}

export interface AdminMigrationResponse {
  started: boolean;
  message: string;
  job_id?: string | null;
  access: AccessMetadata;
}

export interface ClearResponse {
  cleared: boolean;
}

