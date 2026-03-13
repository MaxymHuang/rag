interface ModelSelectionPanelProps {
  models: string[];
  selectedModel: string;
  loading: boolean;
  error: string;
  onSelectModel: (model: string) => void;
}

export function ModelSelectionPanel({
  models,
  selectedModel,
  loading,
  error,
  onSelectModel
}: ModelSelectionPanelProps) {
  return (
    <section className="rounded-2xl border border-neutral-800 bg-black p-5 shadow-sm">
      <h2 className="mb-3 text-lg font-medium">Model Selection</h2>
      {error ? <p className="mb-3 text-sm text-white">{error}</p> : null}
      <div className="flex flex-wrap items-center gap-3">
        <select
          className="rounded-xl border border-neutral-700 bg-black px-3 py-2 text-sm text-white outline-none transition-all duration-200 focus:border-neutral-500 focus:ring-2 focus:ring-neutral-700/70 disabled:opacity-50"
          value={selectedModel}
          disabled={loading || models.length === 0}
          onChange={(event) => onSelectModel(event.target.value)}
        >
          {models.map((modelName) => (
            <option key={modelName} value={modelName}>
              {modelName}
            </option>
          ))}
        </select>
        {loading ? <span className="text-xs text-neutral-400">Updating model...</span> : null}
      </div>
    </section>
  );
}

