import type { AdminStatusResponse } from "../../types";

interface StatusPanelProps {
  status: AdminStatusResponse | null;
  statusError: string;
  onRefresh: () => void;
}

export function StatusPanel({ status, statusError, onRefresh }: StatusPanelProps) {
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

  return (
    <section className="rounded-2xl border border-neutral-800 bg-black p-5 shadow-sm">
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
          onClick={onRefresh}
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
            <p>
              LLM GPU offload (llama.cpp):{" "}
              {status.llamacpp_gpu_offload ? "yes" : "no"}
              {status.llamacpp_n_gpu_layers != null ? ` (n_gpu_layers=${status.llamacpp_n_gpu_layers})` : ""}
            </p>
          </div>
          {status.llamacpp_gpu_offload === false &&
          status.llamacpp_n_gpu_layers != null &&
          status.llamacpp_n_gpu_layers !== 0 ? (
            <p className="mt-3 text-sm text-amber-200">
              This Python build of llama-cpp-python has no GPU backend; LLAMACPP_N_GPU_LAYERS is ignored and inference
              runs on CPU. Reinstall llama-cpp-python with CUDA (see README, section LLM GPU acceleration).
            </p>
          ) : null}
          {!isBackendReady ? <p className="mt-3 text-sm text-white">Missing: {missingReadinessItems.join(", ")}</p> : null}
        </>
      ) : (
        <p className="text-sm text-neutral-400">Loading status...</p>
      )}
    </section>
  );
}

