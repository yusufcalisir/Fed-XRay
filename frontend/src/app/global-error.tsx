"use client";

import React, { useEffect } from "react";
import { ShieldAlert, RefreshCw, RotateCcw } from "lucide-react";

interface GlobalErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function GlobalError({ error, reset }: GlobalErrorProps) {
  useEffect(() => {
    console.warn("[Next.js Global Root Crash Guard] Caught root layout exception:", error?.message);
  }, [error]);

  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center p-4 font-sans antialiased">
        <div className="max-w-xl w-full bg-slate-900/90 border border-red-500/30 rounded-2xl p-6 sm:p-8 text-center shadow-2xl space-y-5">
          <div className="mx-auto w-16 h-16 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center text-red-400 shadow-lg shadow-red-500/10">
            <ShieldAlert className="w-8 h-8" />
          </div>

          <div className="space-y-2">
            <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-white">
              Fed-XRay Root Runtime Protection Active
            </h1>
            <p className="text-xs sm:text-sm text-slate-400 leading-relaxed">
              The root application layout contained an unexpected runtime exception.
              All client enclaves, model parameters, and local data partitions remain untouched and secure.
            </p>
          </div>

          {error?.message && (
            <div className="text-left bg-slate-950 rounded-lg p-3 text-xs font-mono text-slate-400 border border-slate-800 overflow-x-auto max-h-32">
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
              <span>Recover Application</span>
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
      </body>
    </html>
  );
}

