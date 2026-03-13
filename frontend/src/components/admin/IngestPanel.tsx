import type { IngestSource } from "../../types";

const INGEST_SOURCES: IngestSource[] = ["all", "local", "notion"];

interface IngestPanelProps {
  ingestSource: IngestSource;
  ingestStatus: string;
  ingestMessage: string;
  ingestProgress: number;
  ingestJobId: string;
  ingestError: string;
  onChangeSource: (source: IngestSource) => void;
  onStartIngest: () => void;
}

export function IngestPanel({
  ingestSource,
  ingestStatus,
  ingestMessage,
  ingestProgress,
  ingestJobId,
  ingestError,
  onChangeSource,
  onStartIngest
}: IngestPanelProps) {
  return (
    <section className="rounded-2xl border border-neutral-800 bg-black p-5 shadow-sm">
      <h2 className="mb-3 text-lg font-medium">Ingest Documents</h2>
      <div className="flex flex-wrap items-center gap-3">
        <select
          className="rounded-xl border border-neutral-700 bg-black px-3 py-2 text-sm text-white outline-none transition-all duration-200 focus:border-neutral-500 focus:ring-2 focus:ring-neutral-700/70"
          value={ingestSource}
          onChange={(event) => onChangeSource(event.target.value as IngestSource)}
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
    </section>
  );
}

