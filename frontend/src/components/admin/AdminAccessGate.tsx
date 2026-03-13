import type { ReactNode } from "react";
import type { AccessMetadata } from "../../types";

interface AdminAccessGateProps {
  access: AccessMetadata | null;
  loading: boolean;
  error: string;
  children: ReactNode;
}

export function AdminAccessGate({ access, loading, error, children }: AdminAccessGateProps) {
  if (loading) {
    return <p className="text-sm text-neutral-400">Loading admin access...</p>;
  }

  if (error) {
    return <p className="text-sm text-white">{error}</p>;
  }

  if (!access) {
    return <p className="text-sm text-white">Unable to verify admin access metadata.</p>;
  }

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-neutral-800 bg-black p-4">
        <p className="text-xs uppercase tracking-wide text-neutral-500">Access Mode</p>
        <p className="mt-1 text-sm text-neutral-200">{access.accessMode}</p>
        <p className="mt-1 text-xs text-neutral-500">
          {access.requiresAuth
            ? "Authentication is required for admin access."
            : "Auth is not enforced yet. RBAC hooks are wired for next phase."}
        </p>
      </div>
      {children}
    </div>
  );
}

