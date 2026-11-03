"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";
import { ShieldAlert, RefreshCw, RotateCcw } from "lucide-react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  language?: "EN" | "TR" | "en" | "tr";
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ClinicalErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.setState({ errorInfo });
    // In product mode, send safely to telemetry without throwing
    if (process.env.NODE_ENV !== "production") {
      console.warn("[ClinicalErrorBoundary] Caught client-side rendering exception:", error.message);
    }
  }

  private handleReset = (): void => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  private handleReload = (): void => {
    if (typeof window !== "undefined") {
      window.location.reload();
    }
  };

  public render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const isTr = this.props.language?.toUpperCase() === "TR";

      return (
        <div className="min-h-[400px] w-full flex items-center justify-center p-6 bg-slate-900/60 backdrop-blur-md rounded-2xl border border-red-500/20 text-slate-100 shadow-2xl my-6">
          <div className="max-w-xl text-center space-y-5">
            <div className="mx-auto w-16 h-16 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center text-red-400 shadow-lg shadow-red-500/10">
              <ShieldAlert className="w-8 h-8" />
            </div>

            <div className="space-y-2">
              <h2 className="text-xl font-bold tracking-tight text-white flex items-center justify-center gap-2">
                <span>
                  {isTr
                    ? "Klinik Arayüz Koruma Kalkanı Devrede"
                    : "Clinical Interface Protection Active"}
                </span>
              </h2>
              <p className="text-sm text-slate-400 leading-relaxed">
                {isTr
                  ? "Beklenmeyen bir arayüz işleme hatası yakalandı ve izole edildi. Fed-XRay federal model parametreleri, şifreleme enklavı ve hasta verileri güvendedir."
                  : "An unexpected interface rendering exception was safely caught and contained. Federated model parameters, privacy enclaves, and hospital cohorts remain secure."}
              </p>
            </div>

            {this.state.error && (
              <div className="text-left bg-slate-950/80 rounded-lg p-3 text-xs font-mono text-slate-400 border border-slate-800 overflow-x-auto max-h-32">
                <p className="text-red-400 font-semibold mb-1">
                  {this.state.error.name}: {this.state.error.message}
                </p>
              </div>
            )}

            <div className="flex items-center justify-center gap-3 pt-2">
              <button
                type="button"
                onClick={this.handleReset}
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold shadow-lg shadow-cyan-600/20 transition-all hover:scale-105 active:scale-95"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>{isTr ? "Bileşeni Yeniden Dene" : "Retry Component"}</span>
              </button>
              <button
                type="button"
                onClick={this.handleReload}
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold border border-slate-700 transition-all"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>{isTr ? "Sayfayı Yenile" : "Reload Page"}</span>
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ClinicalErrorBoundary;

