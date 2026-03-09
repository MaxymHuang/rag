interface DangerZonePanelProps {
  clearLoading: boolean;
  clearError: string;
  onClear: () => void;
}

export function DangerZonePanel({ clearLoading, clearError, onClear }: DangerZonePanelProps) {
  return (
    <section className="rounded-2xl border border-neutral-800 bg-black p-5 shadow-sm">
      <h2 className="mb-3 text-lg font-medium text-white">Danger Zone</h2>
      <p className="mb-3 text-sm text-neutral-300">
        Clear removes all indexed vectors. You must ingest again before chat can answer.
      </p>
      <button
        className="rounded-xl border border-neutral-700 bg-black px-4 py-2 text-sm font-medium text-white transition hover:bg-neutral-900 disabled:cursor-not-allowed disabled:opacity-60"
        type="button"
        onClick={onClear}
        disabled={clearLoading}
      >
        {clearLoading ? "Clearing..." : "Clear Index"}
      </button>
      {clearError ? <p className="mt-2 text-sm text-white">{clearError}</p> : null}
    </section>
  );
}

