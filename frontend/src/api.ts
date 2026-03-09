import type {
  AccessMetadata,
  AdminMigrationRequest,
  AdminMigrationResponse,
  AdminModelsResponse,
  AdminStatusResponse,
  AdminSystemConfigResponse,
  AdminSystemConfigUpdateRequest,
  AdminSystemConfigUpdateResponse,
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

interface RawAccessMetadata {
  access_mode: string;
  requires_auth: boolean;
  permissions: string[];
}

interface RawAdminStatusResponse extends Omit<AdminStatusResponse, "access"> {
  access: RawAccessMetadata;
}

interface RawAdminModelsResponse extends Omit<AdminModelsResponse, "access"> {
  access: RawAccessMetadata;
}

interface RawAdminSystemConfigResponse extends Omit<AdminSystemConfigResponse, "access"> {
  access: RawAccessMetadata;
}

interface RawAdminSystemConfigUpdateResponse extends Omit<AdminSystemConfigUpdateResponse, "config"> {
  config: RawAdminSystemConfigResponse;
}

interface RawAdminMigrationResponse extends Omit<AdminMigrationResponse, "access"> {
  access: RawAccessMetadata;
}

function mapAccessMetadata(input: RawAccessMetadata): AccessMetadata {
  return {
    accessMode: input.access_mode,
    requiresAuth: input.requires_auth,
    permissions: input.permissions
  };
}

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

export async function fetchAdminStatus(): Promise<AdminStatusResponse> {
  const response = await fetch(`${API_BASE_URL}/admin/status`);
  const raw = await readJson<RawAdminStatusResponse>(response);
  return { ...raw, access: mapAccessMetadata(raw.access) };
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

export async function fetchAdminModels(): Promise<AdminModelsResponse> {
  const response = await fetch(`${API_BASE_URL}/admin/models`);
  const raw = await readJson<RawAdminModelsResponse>(response);
  return { ...raw, access: mapAccessMetadata(raw.access) };
}

export async function selectModel(model: string): Promise<ModelsResponse> {
  const response = await fetch(`${API_BASE_URL}/models/select`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model })
  });
  return readJson<ModelsResponse>(response);
}

export async function selectAdminModel(model: string): Promise<AdminModelsResponse> {
  const response = await fetch(`${API_BASE_URL}/admin/models/select`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model })
  });
  const raw = await readJson<RawAdminModelsResponse>(response);
  return { ...raw, access: mapAccessMetadata(raw.access) };
}

export async function startIngest(source: IngestSource): Promise<IngestStartResponse> {
  const response = await fetch(`${API_BASE_URL}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source })
  });
  return readJson<IngestStartResponse>(response);
}

export async function startAdminIngest(source: IngestSource): Promise<IngestStartResponse> {
  const response = await fetch(`${API_BASE_URL}/admin/ingest`, {
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

export async function clearAdminIngestedData(): Promise<ClearResponse> {
  const response = await fetch(`${API_BASE_URL}/admin/clear`, {
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

export function createAdminIngestEventSource(
  jobId: string,
  onEvent: (event: IngestEvent) => void,
  onError: (event: Event) => void
): EventSource {
  const source = new EventSource(`${API_BASE_URL}/admin/ingest/${jobId}/events`);
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

export async function fetchAdminSystemConfig(): Promise<AdminSystemConfigResponse> {
  const response = await fetch(`${API_BASE_URL}/admin/system/config`);
  const raw = await readJson<RawAdminSystemConfigResponse>(response);
  return { ...raw, access: mapAccessMetadata(raw.access) };
}

export async function updateAdminSystemConfig(
  payload: AdminSystemConfigUpdateRequest
): Promise<AdminSystemConfigUpdateResponse> {
  const response = await fetch(`${API_BASE_URL}/admin/system/config`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const raw = await readJson<RawAdminSystemConfigUpdateResponse>(response);
  return {
    ...raw,
    config: {
      ...raw.config,
      access: mapAccessMetadata(raw.config.access)
    }
  };
}

export async function runAdminMigration(
  payload: AdminMigrationRequest
): Promise<AdminMigrationResponse> {
  const response = await fetch(`${API_BASE_URL}/admin/system/migrate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const raw = await readJson<RawAdminMigrationResponse>(response);
  return { ...raw, access: mapAccessMetadata(raw.access) };
}

