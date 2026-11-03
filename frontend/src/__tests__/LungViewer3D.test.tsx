import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@/test/test-utils";
import LungViewer3D from "@/components/LungViewer3D";

describe("LungViewer3D Component", () => {
  it("renders container element with correct test id and background styling", () => {
    render(<LungViewer3D showHint={false} />);
    const container = screen.getByTestId("lung-viewer-3d");
    expect(container).toBeInTheDocument();
  });

  it("renders WebGL error fallback when WebGL context cannot be created", () => {
    // Force getContext to return null for webgl/webgl2
    const originalGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = vi.fn().mockImplementation((type: string) => {
      if (type === "webgl" || type === "webgl2") return null;
      if (type === "2d") {
        return {
          createImageData: () => ({ data: new Uint8ClampedArray(4) }),
          putImageData: () => {},
        } as any;
      }
      return null;
    }) as any;

    try {
      render(<LungViewer3D />);
      expect(screen.getByText(/3D rendering requires WebGL/i)).toBeInTheDocument();
    } finally {
      HTMLCanvasElement.prototype.getContext = originalGetContext;
    }
  });

  it("renders hint element when showHint is true and WebGL is available", () => {
    const originalGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = vi.fn().mockImplementation((type: string) => {
      if (type === "webgl" || type === "webgl2") return {} as any;
      if (type === "2d") {
        return {
          createImageData: () => ({ data: new Uint8ClampedArray(4) }),
          putImageData: () => {},
        } as any;
      }
      return null;
    }) as any;

    try {
      render(<LungViewer3D showHint={true} />);
      expect(screen.getByText(/Drag to rotate | Scroll to zoom/i)).toBeInTheDocument();
    } finally {
      HTMLCanvasElement.prototype.getContext = originalGetContext;
    }
  });
});
