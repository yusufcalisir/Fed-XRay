"use client";

import React, { Component, ReactNode, Suspense, useEffect, useState } from "react";
import { useLanguage } from "@/context/LanguageContext";
import type { LungViewMode } from "@/lib/scanTexture";

const LungScene = React.lazy(() => import("./LungScene"));

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

class SceneBoundary extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
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

  useEffect(() => {
    setWebgl(detectWebGL());
  }, []);

  const message = (text: string) => (
    <div className="flex h-full w-full items-center justify-center p-6 text-center text-xs text-[var(--text-muted)]">{text}</div>
  );

  return (
    <div className={`relative overflow-hidden ${className}`} data-testid="lung-viewer-3d">
      {webgl === null && message(t("lung3d_loading"))}
      {webgl === false && message(t("lung3d_webgl_error"))}
      {webgl && (
        <SceneBoundary fallback={message(t("lung3d_webgl_error"))}>
          <Suspense fallback={message(t("lung3d_loading"))}>
            <LungScene
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
        <div className="pointer-events-none absolute bottom-2 right-3 font-mono text-[10px] text-white/50">{t("lung3d_hint")}</div>
      )}
    </div>
  );
}

