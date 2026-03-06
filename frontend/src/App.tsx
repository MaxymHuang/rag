import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { clearIngestedData, createIngestEventSource, fetchModels, fetchStatus, selectModel, sendChat, startIngest } from "./api";
import type { ChatRole, ContextSource, IngestEvent, IngestSource, QueryMode, StatusResponse } from "./types";

const SEARCH_MODES: QueryMode[] = ["hybrid", "vector", "keyword"];
const INGEST_SOURCES: IngestSource[] = ["all", "local", "notion"];
const CONTEXT_SOURCES: ContextSource[] = ["local", "notion"];

interface ChatMessage {
  id: number;
  role: ChatRole;
  content: string;
  sources: Array<{ source: string; title?: string | null; content_preview: string }>;
}

export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [statusError, setStatusError] = useState<string>("");
  const [question, setQuestion] = useState("");
  const [filterTitle, setFilterTitle] = useState("");
  const [mode, setMode] = useState<QueryMode>("hybrid");
  const [showSources, setShowSources] = useState(true);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [modelLoading, setModelLoading] = useState(false);
  const [modelError, setModelError] = useState("");

  const [ingestSource, setIngestSource] = useState<IngestSource>("all");
  const [ingestJobId, setIngestJobId] = useState("");
  const [ingestStatus, setIngestStatus] = useState("idle");
  const [ingestMessage, setIngestMessage] = useState("Waiting");
  const [ingestProgress, setIngestProgress] = useState(0);
  const [ingestError, setIngestError] = useState("");
  const [clearLoading, setClearLoading] = useState(false);
  const [clearError, setClearError] = useState("");
  const [contextSources, setContextSources] = useState<ContextSource[]>(["local", "notion"]);

  const eventSourceRef = useRef<EventSource | null>(null);
  const messageIdRef = useRef(1);

  const canAsk = useMemo(
    () => question.trim().length > 0 && !chatLoading && contextSources.length > 0,
    [chatLoading, contextSources.length, question]
  );
  const isBackendReady =
    status !== null &&
    status.chunk_count > 0 &&
    status.notion_configured &&
    status.embedding_model.trim().length > 0 &&
    status.llm_model.trim().length > 0;
  const missingReadinessItems =
    status === null
      ? []
      : [
          ...(status.chunk_count > 0 ? [] : ["No chunks indexed"]),
          ...(status.notion_configured ? [] : ["Notion not configured"]),
          ...(status.embedding_model.trim().length > 0 ? [] : ["Embedding model missing"]),
          ...(status.llm_model.trim().length > 0 ? [] : ["LLM model missing"])
        ];

  async function loadStatus() {
    try {
      setStatusError("");
      const data = await fetchStatus();
      setStatus(data);
    } catch (error) {
      setStatusError(error instanceof Error ? error.message : "Failed to load status");
    }
  }

  async function loadModels(showError = true) {
    try {
      setModelError("");
      const data = await fetchModels();
      setModels(data.available);
      setSelectedModel(data.current);
      setStatus((previous) => (previous ? { ...previous, llm_model: data.current } : previous));
    } catch (error) {
      if (showError) {
        setModelError(error instanceof Error ? error.message : "Failed to load model options");
      }
    }
  }

  useEffect(() => {
    void loadStatus();
    void loadModels();
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  useEffect(() => {
    const pollId = window.setInterval(() => {
      void loadModels(false);
    }, 15000);
    const onFocus = () => {
      void loadModels(false);
    };
    window.addEventListener("focus", onFocus);
    return () => {
      window.clearInterval(pollId);
      window.removeEventListener("focus", onFocus);
    };
  }, []);

  async function onSubmitChat(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!question.trim()) {
      return;
    }
    const trimmedQuestion = question.trim();
    const currentHistory = messages.map((item) => ({ role: item.role, content: item.content }));
    const userMessage: ChatMessage = {
      id: messageIdRef.current++,
      role: "user",
      content: trimmedQuestion,
      sources: []
    };
    const thinkingMessageId = messageIdRef.current++;
    const thinkingMessage: ChatMessage = {
      id: thinkingMessageId,
      role: "assistant",
      content: "thinking...",
      sources: []
    };

    setMessages((previous) => [...previous, userMessage, thinkingMessage]);
    setQuestion("");
    setChatLoading(true);
    setChatError("");
    try {
      const response = await sendChat({
        question: trimmedQuestion,
        mode,
        showSources,
        filterTitle: filterTitle.trim() || undefined,
        history: currentHistory,
        contextSources
      });
      const assistantMessage: ChatMessage = {
        id: thinkingMessageId,
        role: "assistant",
        content: response.answer,
        sources: response.sources
      };
      setMessages((previous) =>
        previous.map((message) => (message.id === thinkingMessageId ? assistantMessage : message))
      );
    } catch (error) {
      setMessages((previous) => previous.filter((message) => message.id !== thinkingMessageId));
      setChatError(error instanceof Error ? error.message : "Chat failed");
    } finally {
      setChatLoading(false);
    }
  }

  async function onSelectModel(nextModel: string) {
    try {
      setModelLoading(true);
      setModelError("");
      const data = await selectModel(nextModel);
      setSelectedModel(data.current);
      setModels(data.available);
      setStatus((previous) => (previous ? { ...previous, llm_model: data.current } : previous));
    } catch (error) {
      setModelError(error instanceof Error ? error.message : "Failed to switch model");
    } finally {
      setModelLoading(false);
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

  function onToggleContextSource(source: ContextSource) {
    setContextSources((previous) => {
      if (previous.includes(source)) {
        if (previous.length === 1) {
          return previous;
        }
        return previous.filter((item) => item !== source);
      }
      return [...previous, source];
    });
  }

  async function onClearIngest() {
    try {
      setClearLoading(true);
      setClearError("");
      setIngestError("");
      eventSourceRef.current?.close();
      await clearIngestedData();
      setIngestJobId("");
      setIngestStatus("idle");
      setIngestMessage("Vector store cleared");
      setIngestProgress(0);
      await loadStatus();
    } catch (error) {
      setClearError(error instanceof Error ? error.message : "Unable to clear ingested data");
    } finally {
      setClearLoading(false);
    }
  }

  const panelClass = "rounded-2xl border border-neutral-800 bg-black p-5 shadow-sm";
  const inputClass =
    "rounded-xl border border-neutral-700 bg-black px-3 py-2 text-sm text-white outline-none transition-all duration-200 focus:border-neutral-500 focus:ring-2 focus:ring-neutral-700/70";

  return (
    <main className="min-h-screen bg-black text-white">
      <div className="mx-auto max-w-6xl space-y-6 px-4 py-8">
        <header className={panelClass}>
          <h1 className="text-2xl font-semibold tracking-tight text-white">AXI Expert</h1>
          <p className="mt-1 text-sm text-neutral-400">Chat over local and Notion documents with live ingestion tracking.</p>
        </header>

        <section className={panelClass}>
          <div className="mb-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-medium">Agent Status</h2>
              {isBackendReady ? (
                <span className="rounded-full bg-neutral-800 px-2 py-0.5 text-xs font-medium text-white ring-1 ring-neutral-700">
                  Ready
                </span>
              ) : (
                <span className="rounded-full bg-neutral-900 px-2 py-0.5 text-xs font-medium text-neutral-200 ring-1 ring-neutral-700">
                  Not Ready
                </span>
              )}
            </div>
            <button
              className="rounded-xl border border-neutral-700 bg-black px-3 py-1.5 text-sm text-white transition hover:bg-neutral-900"
              onClick={() => void loadStatus()}
              type="button"
            >
              Refresh
            </button>
          </div>
          {statusError ? (
            <p className="text-sm text-white">{statusError}</p>
          ) : status ? (
            <>
              <div className="grid gap-2 text-sm text-neutral-300 md:grid-cols-2">
                <p>Chunks: {status.chunk_count}</p>
                <p>Notion configured: {status.notion_configured ? "Yes" : "No"}</p>
                <p className="truncate">Embedding model: {status.embedding_model}</p>
                <p className="truncate">LLM model: {status.llm_model}</p>
              </div>
              {!isBackendReady ? <p className="mt-3 text-sm text-white">Missing: {missingReadinessItems.join(", ")}</p> : null}
            </>
          ) : (
            <p className="text-sm text-neutral-400">Loading status...</p>
          )}
        </section>

        <section className={panelClass}>
          <h2 className="mb-3 text-lg font-medium">Ingest Documents</h2>
          <div className="flex flex-wrap items-center gap-3">
            <select
              className={inputClass}
              value={ingestSource}
              onChange={(event) => setIngestSource(event.target.value as IngestSource)}
            >
              {INGEST_SOURCES.map((source) => (
                <option key={source} value={source}>
                  {source}
                </option>
              ))}
            </select>
            <button
              className="rounded-xl border border-neutral-600 bg-white px-4 py-2 text-sm font-medium text-black transition hover:bg-neutral-200"
              type="button"
              onClick={onStartIngest}
            >
              Ingest
            </button>
            <button
              className="rounded-xl border border-neutral-700 bg-black px-4 py-2 text-sm font-medium text-white transition hover:bg-neutral-900 disabled:cursor-not-allowed disabled:opacity-60"
              type="button"
              onClick={() => void onClearIngest()}
              disabled={clearLoading}
            >
              {clearLoading ? "Clearing..." : "Clear"}
            </button>
          </div>
          <div className="mt-4 h-3 w-full overflow-hidden rounded-full bg-neutral-900">
            <div
              className="h-full bg-white transition-all duration-300"
              style={{ width: `${Math.max(0, Math.min(100, ingestProgress))}%` }}
            />
          </div>
          <p className="mt-2 text-sm text-neutral-300">
            {ingestStatus} ({ingestProgress}%): {ingestMessage}
          </p>
          {ingestJobId ? <p className="text-xs text-neutral-500">Job: {ingestJobId}</p> : null}
          {ingestError ? <p className="mt-1 text-sm text-white">{ingestError}</p> : null}
          {clearError ? <p className="mt-1 text-sm text-white">{clearError}</p> : null}
        </section>

        <section className={panelClass}>
          <h2 className="mb-3 text-lg font-medium">Chat</h2>
          {chatError ? <p className="mb-2 text-sm text-white">{chatError}</p> : null}
          {modelError ? <p className="mb-2 text-sm text-white">{modelError}</p> : null}
          <div className="mt-4 max-h-[34rem] space-y-3 overflow-y-auto pr-1">
            {messages.map((message) => (
              <article
                className={`rounded-2xl border p-4 transition-all ${
                  message.role === "user"
                    ? "border-neutral-700 bg-neutral-950"
                    : "border-neutral-800 bg-black"
                }`}
                key={message.id}
              >
                <p className="mb-2 text-xs uppercase tracking-wide text-neutral-400">{message.role}</p>
                {message.role === "assistant" ? (
                  <div className="prose prose-invert prose-sm max-w-none break-words">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        table: ({ children }) => (
                          <div className="my-3 overflow-x-auto">
                            <table className="min-w-full border-collapse border border-slate-600 text-left text-xs">{children}</table>
                          </div>
                        ),
                        th: ({ children }) => <th className="border border-slate-600 bg-slate-700/70 px-2 py-1">{children}</th>,
                        td: ({ children }) => <td className="border border-slate-700 px-2 py-1 align-top">{children}</td>,
                        code: ({ className, children, ...props }) => {
                          const isBlock = className?.includes("language-");
                          if (isBlock) {
                            return (
                              <code className="block overflow-x-auto rounded-xl bg-black px-3 py-2 text-white" {...props}>
                                {children}
                              </code>
                            );
                          }
                          return (
                            <code className="rounded bg-neutral-800 px-1 py-0.5 text-white" {...props}>
                              {children}
                            </code>
                          );
                        },
                        p: ({ children }) => <p className="mb-2 whitespace-pre-wrap">{children}</p>,
                        li: ({ children }) => <li className="mb-1">{children}</li>
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap text-sm text-white">{message.content}</p>
                )}
                {message.role === "assistant" && message.sources.length > 0 ? (
                  <details className="mt-3 rounded-xl border border-neutral-800 bg-black p-2">
                    <summary className="cursor-pointer text-sm font-medium text-neutral-200">Reference</summary>
                    <div className="mt-2 space-y-2">
                      {message.sources.map((source, idx) => (
                        <div className="rounded-xl border border-neutral-800 bg-neutral-950 p-3 text-sm" key={`${source.source}-${idx}`}>
                          <p className="font-medium text-white">{source.source}</p>
                          {source.title ? <p className="text-neutral-400">title: {source.title}</p> : null}
                          <p className="mt-1 text-neutral-300">{source.content_preview}</p>
                        </div>
                      ))}
                    </div>
                  </details>
                ) : null}
              </article>
            ))}
          </div>
          <form className="mt-4 space-y-3 border-t border-neutral-800 pt-4" onSubmit={onSubmitChat}>
            <textarea
              className="min-h-24 w-full rounded-xl border border-neutral-700 bg-black p-3 text-sm text-white outline-none transition-all focus:border-neutral-500 focus:ring-2 focus:ring-neutral-700/70"
              placeholder="Ask about your indexed documents..."
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
            />
            <div className="flex flex-wrap items-center gap-3">
              <select className={inputClass} value={mode} onChange={(event) => setMode(event.target.value as QueryMode)}>
                {SEARCH_MODES.map((searchMode) => (
                  <option key={searchMode} value={searchMode}>
                    {searchMode}
                  </option>
                ))}
              </select>
              <input
                className={inputClass}
                placeholder="Optional title filter"
                value={filterTitle}
                onChange={(event) => setFilterTitle(event.target.value)}
              />
              <div className="flex items-center gap-2 rounded-xl border border-neutral-800 bg-black px-3 py-2">
                <span className="text-xs uppercase tracking-wide text-neutral-400">Context</span>
                {CONTEXT_SOURCES.map((source) => (
                  <label className="flex items-center gap-1 text-xs text-neutral-200" key={source}>
                    <input
                      checked={contextSources.includes(source)}
                      onChange={() => onToggleContextSource(source)}
                      type="checkbox"
                    />
                    {source}
                  </label>
                ))}
              </div>
              <label className="flex items-center gap-2 text-sm text-neutral-300">
                <input checked={showSources} onChange={(event) => setShowSources(event.target.checked)} type="checkbox" />
                Show sources
              </label>
              <select
                className={`${inputClass} disabled:opacity-50`}
                value={selectedModel}
                disabled={modelLoading || models.length === 0}
                onChange={(event) => void onSelectModel(event.target.value)}
              >
                {models.map((modelName) => (
                  <option key={modelName} value={modelName}>
                    {modelName}
                  </option>
                ))}
              </select>
              <button
                aria-label="Send"
                className="inline-flex items-center justify-center rounded-xl border border-neutral-600 bg-white px-3 py-2 text-sm text-black transition hover:bg-neutral-200 disabled:cursor-not-allowed disabled:opacity-60"
                disabled={!canAsk}
                type="submit"
              >
                <svg
                  aria-hidden="true"
                  className={`h-5 w-5 ${chatLoading ? "animate-pulse" : ""}`}
                  fill="none"
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  viewBox="0 0 24 24"
                >
                  <path d="m22 2-7 20-4-9-9-4Z" />
                  <path d="M22 2 11 13" />
                </svg>
                <span className="sr-only">{chatLoading ? "Sending" : "Send"}</span>
              </button>
            </div>
          </form>
        </section>
      </div>
    </main>
  );
}

