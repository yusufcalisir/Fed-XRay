"use client";

import React, { Component, ReactNode, Suspense, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { AlertCircle, RefreshCw } from "lucide-react";
import { useLanguage } from "@/context/LanguageContext";
import type { LungViewMode } from "@/lib/scanTexture";

const LungScene = dynamic(() => import("./LungScene"), {
  ssr: false,
  loading: () => null,
});

export interface LungViewer3DProps {
  rawImage?: number[][] | null;
  heatmap?: number[][] | null;
  mode?: LungViewMode;
  opacity?: number;
  colormap?: string;
  sway?: boolean;
  labels?: boolean;
  showHint?: boolean;
  className?: string;
}

function detectWebGL(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return !!(canvas.getContext("webgl2") || canvas.getContext("webgl"));
  } catch {
    return false;
  }
}

interface SceneBoundaryProps {
  fallback: (error: Error | null, reset: () => void) => ReactNode;
  children: ReactNode;
}

interface SceneBoundaryState {
  failed: boolean;
  error: Error | null;
}

class SceneBoundary extends Component<SceneBoundaryProps, SceneBoundaryState> {
  state: SceneBoundaryState = { failed: false, error: null };

  static getDerivedStateFromError(error: Error): SceneBoundaryState {
    return { failed: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("[LungViewer3D SceneBoundary Caught]", error, errorInfo);
  }

  reset = () => {
    this.setState({ failed: false, error: null });
  };

  render() {
    return this.state.failed ? this.props.fallback(this.state.error, this.reset) : this.props.children;
  }
}

export default function LungViewer3D({
  rawImage = null,
  heatmap = null,
  mode = "anatomy",
  opacity = 0.7,
  colormap = "Hot",
  sway = true,
  labels = true,
  showHint = true,
  className = "",
}: LungViewer3DProps) {
  const { t } = useLanguage();
  const [webgl, setWebgl] = useState<boolean | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setWebgl(detectWebGL());
  }, []);

  const message = (text: string) => (
    <div className="flex h-full w-full items-center justify-center p-6 text-center text-xs text-[var(--text-muted)]">{text}</div>
  );

  return (
    <div
      ref={containerRef}
      className={`relative isolate w-full h-full overflow-hidden rounded-xl ${className || ""}`}
      style={{ background: "var(--viewer-bg)" }}
      data-testid="lung-viewer-3d"
    >
      {webgl === null && message(t("lung3d_loading"))}
      {webgl === false && message(t("lung3d_webgl_error"))}
      {webgl && (
        <SceneBoundary
          fallback={(_error, reset) => (
            <div className="flex h-full w-full flex-col items-center justify-center p-6 text-center text-xs text-[var(--text-muted)] gap-2">
              <AlertCircle className="w-5 h-5 text-amber-500 mb-0.5" />
              <p className="max-w-xs text-[11px] leading-relaxed">
                {t("lung3d_render_error")}
              </p>
              <button
                type="button"
                onClick={reset}
                className="mt-1 inline-flex items-center gap-1.5 px-3 py-1 rounded-lg border border-[var(--border-card)] bg-[var(--bg-card)] text-[11px] font-semibold text-[var(--text-heading)] hover:bg-[var(--bg-card-inner)] transition-colors shadow-xs"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>{t("lung3d_retry")}</span>
              </button>
            </div>
          )}
        >
          <Suspense fallback={message(t("lung3d_loading"))}>
            <LungScene
              containerRef={containerRef}
              rawImage={rawImage}
              heatmap={heatmap}
              mode={mode}
              opacity={opacity}
              colormap={colormap}
              sway={sway}
              labels={labels}
              peakLabel={t("lung3d_peak")}
              rightLabel={t("lung3d_right")}
              leftLabel={t("lung3d_left")}
              tracheaLabel={t("lung3d_trachea")}
            />
          </Suspense>
        </SceneBoundary>
      )}
      {webgl && showHint && (
        <div className="pointer-events-none absolute bottom-2 right-3 font-mono text-[10px] text-[var(--text-muted)]">{t("lung3d_hint")}</div>
      )}
    </div>
  );
}

