"use client";

import React, { useEffect } from "react";
import { ShieldAlert, RefreshCw, RotateCcw } from "lucide-react";

interface ErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function RootRouteError({ error, reset }: ErrorProps) {
  useEffect(() => {
    console.warn("[Next.js App Error Boundary] Caught segment rendering exception:", error?.message);
  }, [error]);

  return (
    <div className="min-h-[500px] w-full flex items-center justify-center p-6 bg-slate-950/90 backdrop-blur-md rounded-2xl border border-red-500/30 text-slate-100 shadow-2xl my-6">
      <div className="max-w-lg text-center space-y-5">
        <div className="mx-auto w-16 h-16 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center text-red-400 shadow-lg shadow-red-500/10">
          <ShieldAlert className="w-8 h-8" />
        </div>

        <div className="space-y-2">
          <h2 className="text-xl font-bold tracking-tight text-white">
            Clinical Interface Protection Active
          </h2>
          <p className="text-xs text-slate-400 leading-relaxed">
            An unexpected client-side rendering exception was safely caught and contained.
            Federated model parameters, privacy enclaves, and hospital datasets remain secure.
          </p>
        </div>

        {error?.message && (
          <div className="text-left bg-slate-900 rounded-lg p-3 text-xs font-mono text-slate-400 border border-slate-800 overflow-x-auto max-h-32">
            <p className="text-red-400 font-semibold mb-1">
              {error.name || "Error"}: {error.message}
            </p>
          </div>
        )}

        <div className="flex items-center justify-center gap-3 pt-2">
          <button
            type="button"
            onClick={() => reset()}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold shadow-lg shadow-cyan-600/20 transition-all hover:scale-105 active:scale-95"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Recover View</span>
          </button>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold border border-slate-700 transition-all"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reload Page</span>
          </button>
        </div>
      </div>
    </div>
  );
}

