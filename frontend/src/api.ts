import type {
  ChatResponse,
  ChatRequest,
  ClearResponse,
  IngestEvent,
  IngestSource,
  IngestStartResponse,
  ModelsResponse,
  StatusResponse
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8001";

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { detail?: string };
    throw new Error(body.detail ?? `Request failed with ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function fetchStatus(): Promise<StatusResponse> {
  const response = await fetch(`${API_BASE_URL}/status`);
  return readJson<StatusResponse>(response);
}

export async function sendChat(params: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: params.question,
      mode: params.mode,
      show_sources: params.showSources,
      filter_title: params.filterTitle || null,
      history: params.history ?? [],
      context_sources: params.contextSources ?? ["local", "notion"]
    })
  });
  return readJson<ChatResponse>(response);
}

export async function fetchModels(): Promise<ModelsResponse> {
  const response = await fetch(`${API_BASE_URL}/models`);
  return readJson<ModelsResponse>(response);
}

export async function selectModel(model: string): Promise<ModelsResponse> {
  const response = await fetch(`${API_BASE_URL}/models/select`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model })
  });
  return readJson<ModelsResponse>(response);
}

export async function startIngest(source: IngestSource): Promise<IngestStartResponse> {
  const response = await fetch(`${API_BASE_URL}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source })
  });
  return readJson<IngestStartResponse>(response);
}

export async function clearIngestedData(): Promise<ClearResponse> {
  const response = await fetch(`${API_BASE_URL}/clear`, {
    method: "POST"
  });
  return readJson<ClearResponse>(response);
}

export function createIngestEventSource(
  jobId: string,
  onEvent: (event: IngestEvent) => void,
  onError: (event: Event) => void
): EventSource {
  const source = new EventSource(`${API_BASE_URL}/ingest/${jobId}/events`);
  source.onmessage = (message) => {
    try {
      onEvent(JSON.parse(message.data) as IngestEvent);
    } catch {
      // Ignore malformed events.
    }
  };
  source.onerror = onError;
  return source;
}

