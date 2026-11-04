import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  resolveUpstreamBackendUrl,
  createGatewayErrorResponse,
  proxyToUpstreamBackend,
} from "@/lib/backendGateway";

describe("Backend Gateway Forwarder (DEFECT-01 Remediation)", () => {
  const originalEnv = process.env;

  beforeEach(() => {
    vi.restoreAllMocks();
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it("resolves default local backend URL when no environment variable is provided in dev", () => {
    delete process.env.NEXT_PUBLIC_API_URL;
    (process.env as any).NODE_ENV = "development";

    const url = resolveUpstreamBackendUrl();
    expect(url).toBe("http://127.0.0.1:8000");
  });

  it("sanitizes trailing slashes from valid custom upstream backend URL", () => {
    process.env.NEXT_PUBLIC_API_URL = "http://localhost:8000///";

    const url = resolveUpstreamBackendUrl();
    expect(url).toBe("http://localhost:8000");
  });

  it("filters out expired placeholder or temporary tunnels and falls back to default", () => {
    process.env.NEXT_PUBLIC_API_URL = "https://globe-celebration-poems-shakira.trycloudflare.com";
    (process.env as any).NODE_ENV = "development";

    const url = resolveUpstreamBackendUrl();
    expect(url).toBe("http://127.0.0.1:8000");
  });

  it("creates standard RFC-compliant 503 Service Unavailable error response", async () => {
    const res = createGatewayErrorResponse("FastAPI backend connection refused", 503, {
      errorCode: "BACKEND_UNAVAILABLE",
      targetEndpoint: "http://127.0.0.1:8000/api/health",
      retryAfterSeconds: 5,
    });

    expect(res.status).toBe(503);
    expect(res.headers.get("Cache-Control")).toContain("no-store");
    expect(res.headers.get("Retry-After")).toBe("5");

    const body = await res.json();
    expect(body.status).toBe("unavailable");
    expect(body.error_code).toBe("BACKEND_UNAVAILABLE");
    expect(body.detail).toBe("FastAPI backend connection refused");
    expect(body.target_endpoint).toBe("http://127.0.0.1:8000/api/health");
  });

  it("creates standard 504 Gateway Timeout response when request times out", async () => {
    const res = createGatewayErrorResponse("Upstream timed out", 504);

    expect(res.status).toBe(504);
    const body = await res.json();
    expect(body.status).toBe("timeout");
    expect(body.error_code).toBe("GATEWAY_TIMEOUT");
  });

  it("transparently forwards successful JSON response from upstream", async () => {
    const mockUpstreamPayload = {
      status: "healthy",
      version: "2.1.0",
      active_architecture: "vit_tiny",
    };

    global.fetch = vi.fn().mockResolvedValue({
      status: 200,
      headers: new Headers({ "content-type": "application/json" }),
      json: async () => mockUpstreamPayload,
    } as any);

    const res = await proxyToUpstreamBackend(null, "/api/health");
    expect(res.status).toBe(200);

    const data = await res.json();
    expect(data.status).toBe("healthy");
    expect(data.version).toBe("2.1.0");
    expect(data.active_architecture).toBe("vit_tiny");
  });

  it("transparently forwards upstream HTTP error status without masking as 200", async () => {
    const mockErrorDetail = { detail: "Held-out test set not loaded" };

    global.fetch = vi.fn().mockResolvedValue({
      status: 409,
      headers: new Headers({ "content-type": "application/json" }),
      json: async () => mockErrorDetail,
    } as any);

    const res = await proxyToUpstreamBackend(null, "/api/cdss/diagnose", { method: "POST" });
    expect(res.status).toBe(409);

    const data = await res.json();
    expect(data.detail).toBe("Held-out test set not loaded");
  });

  it("returns HTTP 503 with honest error description when upstream throws connection error", async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error("connect ECONNREFUSED 127.0.0.1:8000"));

    const res = await proxyToUpstreamBackend(null, "/api/health");
    expect(res.status).toBe(503);

    const data = await res.json();
    expect(data.status).toBe("unavailable");
    expect(data.error_code).toBe("BACKEND_UNAVAILABLE");
    expect(data.detail).toContain("Fed-XRay federated inference backend is unreachable");
  });

  it("returns HTTP 504 when upstream request is aborted due to timeout", async () => {
    const abortErr = new Error("The operation was aborted");
    abortErr.name = "AbortError";
    global.fetch = vi.fn().mockRejectedValue(abortErr);

    const res = await proxyToUpstreamBackend(null, "/api/cohorts/generate", { timeoutMs: 100 });
    expect(res.status).toBe(504);

    const data = await res.json();
    expect(data.status).toBe("timeout");
    expect(data.error_code).toBe("GATEWAY_TIMEOUT");
    expect(data.detail).toContain("timed out");
  });
});

