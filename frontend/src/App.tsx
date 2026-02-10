import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

import { createIngestEventSource, fetchStatus, sendChat, startIngest } from "./api";
import type { IngestEvent, IngestSource, QueryMode, StatusResponse } from "./types";

const SEARCH_MODES: QueryMode[] = ["hybrid", "vector", "keyword"];
const INGEST_SOURCES: IngestSource[] = ["all", "local", "notion"];

export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [statusError, setStatusError] = useState<string>("");
  const [question, setQuestion] = useState("");
  const [filterTitle, setFilterTitle] = useState("");
  const [mode, setMode] = useState<QueryMode>("hybrid");
  const [showSources, setShowSources] = useState(true);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Array<{ source: string; title?: string | null; content_preview: string }>>(
    []
  );
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");

  const [ingestSource, setIngestSource] = useState<IngestSource>("all");
  const [ingestJobId, setIngestJobId] = useState("");
  const [ingestStatus, setIngestStatus] = useState("idle");
  const [ingestMessage, setIngestMessage] = useState("Waiting");
  const [ingestProgress, setIngestProgress] = useState(0);
  const [ingestError, setIngestError] = useState("");

  const eventSourceRef = useRef<EventSource | null>(null);

  const canAsk = useMemo(() => question.trim().length > 0 && !chatLoading, [chatLoading, question]);

  async function loadStatus() {
    try {
      setStatusError("");
      const data = await fetchStatus();
      setStatus(data);
    } catch (error) {
      setStatusError(error instanceof Error ? error.message : "Failed to load status");
    }
  }

  useEffect(() => {
    void loadStatus();
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  async function onSubmitChat(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!question.trim()) {
      return;
    }
    setChatLoading(true);
    setChatError("");
    try {
      const response = await sendChat({
        question,
        mode,
        showSources,
        filterTitle: filterTitle.trim() || undefined
      });
      setAnswer(response.answer);
      setSources(response.sources);
    } catch (error) {
      setChatError(error instanceof Error ? error.message : "Chat failed");
    } finally {
      setChatLoading(false);
    }
  }

  async function onStartIngest() {
    setIngestError("");
    setIngestProgress(0);
    setIngestStatus("queued");
    setIngestMessage("Starting ingestion...");
    eventSourceRef.current?.close();

    try {
      const job = await startIngest(ingestSource);
      setIngestJobId(job.job_id);

      const stream = createIngestEventSource(
        job.job_id,
        (evt: IngestEvent) => {
          setIngestStatus(evt.status);
          setIngestProgress(evt.progress);
          setIngestMessage(evt.message || evt.stage);
          if (evt.status === "completed" || evt.status === "failed") {
            stream.close();
            void loadStatus();
          }
        },
        () => {
          setIngestError("Connection to ingestion events was interrupted");
        }
      );

      eventSourceRef.current = stream;
    } catch (error) {
      setIngestStatus("failed");
      setIngestError(error instanceof Error ? error.message : "Unable to start ingestion");
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-5xl space-y-6 px-4 py-8">
        <header className="rounded-lg border border-slate-800 bg-slate-900 p-4">
          <h1 className="text-2xl font-semibold">RAG Agent</h1>
          <p className="text-sm text-slate-400">Chat over local + Notion documents with ingestion progress tracking.</p>
        </header>

        <section className="rounded-lg border border-slate-800 bg-slate-900 p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-lg font-medium">Backend Status</h2>
            <button
              className="rounded bg-slate-700 px-3 py-1 text-sm hover:bg-slate-600"
              onClick={() => void loadStatus()}
              type="button"
            >
              Refresh
            </button>
          </div>
          {statusError ? (
            <p className="text-sm text-red-400">{statusError}</p>
          ) : status ? (
            <div className="grid gap-2 text-sm text-slate-300 md:grid-cols-2">
              <p>Chunks: {status.chunk_count}</p>
              <p>Notion configured: {status.notion_configured ? "Yes" : "No"}</p>
              <p className="truncate">Embedding model: {status.embedding_model}</p>
              <p className="truncate">LLM model: {status.llm_model}</p>
            </div>
          ) : (
            <p className="text-sm text-slate-400">Loading status...</p>
          )}
        </section>

        <section className="rounded-lg border border-slate-800 bg-slate-900 p-4">
          <h2 className="mb-3 text-lg font-medium">Ingest Documents</h2>
          <div className="flex flex-wrap items-center gap-3">
            <select
              className="rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
              value={ingestSource}
              onChange={(event) => setIngestSource(event.target.value as IngestSource)}
            >
              {INGEST_SOURCES.map((source) => (
                <option key={source} value={source}>
                  {source}
                </option>
              ))}
            </select>
            <button className="rounded bg-blue-600 px-4 py-2 text-sm hover:bg-blue-500" type="button" onClick={onStartIngest}>
              Ingest
            </button>
          </div>
          <div className="mt-4 h-3 w-full overflow-hidden rounded bg-slate-800">
            <div
              className="h-full bg-emerald-500 transition-all duration-300"
              style={{ width: `${Math.max(0, Math.min(100, ingestProgress))}%` }}
            />
          </div>
          <p className="mt-2 text-sm text-slate-300">
            {ingestStatus} ({ingestProgress}%): {ingestMessage}
          </p>
          {ingestJobId ? <p className="text-xs text-slate-500">Job: {ingestJobId}</p> : null}
          {ingestError ? <p className="text-sm text-red-400">{ingestError}</p> : null}
        </section>

        <section className="rounded-lg border border-slate-800 bg-slate-900 p-4">
          <h2 className="mb-3 text-lg font-medium">Chat</h2>
          <form className="space-y-3" onSubmit={onSubmitChat}>
            <textarea
              className="min-h-24 w-full rounded border border-slate-700 bg-slate-800 p-3 text-sm"
              placeholder="Ask about your indexed documents..."
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
            />
            <div className="flex flex-wrap items-center gap-3">
              <select
                className="rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
                value={mode}
                onChange={(event) => setMode(event.target.value as QueryMode)}
              >
                {SEARCH_MODES.map((searchMode) => (
                  <option key={searchMode} value={searchMode}>
                    {searchMode}
                  </option>
                ))}
              </select>
              <input
                className="rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
                placeholder="Optional title filter"
                value={filterTitle}
                onChange={(event) => setFilterTitle(event.target.value)}
              />
              <label className="flex items-center gap-2 text-sm text-slate-300">
                <input checked={showSources} onChange={(event) => setShowSources(event.target.checked)} type="checkbox" />
                Show sources
              </label>
              <button
                className="rounded bg-emerald-600 px-4 py-2 text-sm disabled:opacity-50"
                disabled={!canAsk}
                type="submit"
              >
                {chatLoading ? "Asking..." : "Ask"}
              </button>
            </div>
          </form>
          {chatError ? <p className="mt-3 text-sm text-red-400">{chatError}</p> : null}
          {answer ? (
            <article className="mt-4 space-y-3 rounded border border-slate-700 bg-slate-800 p-4">
              <h3 className="font-medium">Answer</h3>
              <p className="whitespace-pre-wrap text-sm text-slate-200">{answer}</p>
            </article>
          ) : null}
          {sources.length > 0 ? (
            <div className="mt-4 space-y-2">
              <h3 className="font-medium">Sources</h3>
              {sources.map((source, idx) => (
                <div className="rounded border border-slate-700 bg-slate-800 p-3 text-sm" key={`${source.source}-${idx}`}>
                  <p className="font-medium text-slate-100">{source.source}</p>
                  {source.title ? <p className="text-slate-400">title: {source.title}</p> : null}
                  <p className="mt-1 text-slate-300">{source.content_preview}</p>
                </div>
              ))}
            </div>
          ) : null}
        </section>
      </div>
    </main>
  );
}

