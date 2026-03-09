import type { AdminSystemConfigResponse, VectorDbProvider } from "../../types";

interface SystemSettingsPanelProps {
  config: AdminSystemConfigResponse | null;
  loading: boolean;
  error: string;
  feedback: string;
  onSave: (payload: { embedding_model?: string; vector_db_provider?: VectorDbProvider }) => void;
}

export function SystemSettingsPanel({ config, loading, error, feedback, onSave }: SystemSettingsPanelProps) {
  if (!config) {
    return (
      <section className="rounded-2xl border border-neutral-800 bg-black p-5 shadow-sm">
        <h2 className="mb-3 text-lg font-medium">System Settings</h2>
        {error ? <p className="text-sm text-white">{error}</p> : <p className="text-sm text-neutral-400">Loading config...</p>}
      </section>
    );
  }

  return (
    <section className="rounded-2xl border border-neutral-800 bg-black p-5 shadow-sm">
      <h2 className="mb-3 text-lg font-medium">System Settings</h2>
      <div className="grid gap-3 md:grid-cols-2">
        <label className="space-y-2 text-sm text-neutral-300">
          <span>Embedding Model</span>
          <select
            className="w-full rounded-xl border border-neutral-700 bg-black px-3 py-2 text-sm text-white outline-none transition-all duration-200 focus:border-neutral-500 focus:ring-2 focus:ring-neutral-700/70"
            defaultValue={config.embedding_model}
            disabled={loading}
            onChange={(event) => onSave({ embedding_model: event.target.value })}
          >
            {config.embedding_model_options.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <label className="space-y-2 text-sm text-neutral-300">
          <span>Vector DB</span>
          <select
            className="w-full rounded-xl border border-neutral-700 bg-black px-3 py-2 text-sm text-white outline-none transition-all duration-200 focus:border-neutral-500 focus:ring-2 focus:ring-neutral-700/70"
            defaultValue={config.vector_db_provider}
            disabled={loading}
            onChange={(event) => onSave({ vector_db_provider: event.target.value as VectorDbProvider })}
          >
            {config.vector_db_provider_options.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
      </div>
      {feedback ? <p className="mt-3 text-sm text-neutral-300">{feedback}</p> : null}
      {!feedback ? <p className="mt-3 text-xs text-neutral-500">Updates are scaffolded now and fully applied in the next phase.</p> : null}
    </section>
  );
}

