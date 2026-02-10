export type QueryMode = "hybrid" | "vector" | "keyword";
export type IngestSource = "all" | "local" | "notion";
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

export interface StatusResponse {
  documents_directory: string;
  notion_configured: boolean;
  notion_database_id: string;
  chunk_count: number;
  embedding_model: string;
  llm_model: string;
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

