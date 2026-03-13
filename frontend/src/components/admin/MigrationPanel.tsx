interface MigrationPanelProps {
  loading: boolean;
  message: string;
  onReindex: () => void;
  onVectorMigration: () => void;
}

export function MigrationPanel({ loading, message, onReindex, onVectorMigration }: MigrationPanelProps) {
  return (
    <section className="rounded-2xl border border-neutral-800 bg-black p-5 shadow-sm">
      <h2 className="mb-3 text-lg font-medium">Migration</h2>
      <p className="mb-3 text-sm text-neutral-300">
        Prepare migration workflows for embedding and vector DB updates.
      </p>
      <div className="flex flex-wrap items-center gap-3">
        <button
          className="rounded-xl border border-neutral-600 bg-white px-4 py-2 text-sm font-medium text-black transition hover:bg-neutral-200 disabled:cursor-not-allowed disabled:opacity-60"
          type="button"
          onClick={onReindex}
          disabled={loading}
        >
          Reindex Now
        </button>
        <button
          className="rounded-xl border border-neutral-700 bg-black px-4 py-2 text-sm font-medium text-white transition hover:bg-neutral-900 disabled:cursor-not-allowed disabled:opacity-60"
          type="button"
          onClick={onVectorMigration}
          disabled={loading}
        >
          Vector DB Migration
        </button>
      </div>
      {message ? <p className="mt-3 text-sm text-neutral-300">{message}</p> : null}
    </section>
  );
}

