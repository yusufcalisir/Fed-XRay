import { NextResponse } from "next/server";

export interface GatewayErrorPayload {
  status: "unavailable" | "error" | "timeout";
  detail: string;
  error_code: string;
  target_endpoint: string;
  timestamp: number;
  retry_after_seconds?: number;
}

/**
 * Resolve the authoritative upstream FastAPI federated backend URL.
 * Sanitizes expired, dead, or placeholder tunnels and falls back to
 * environment or standard loopback configuration.
 */
export function resolveUpstreamBackendUrl(): string {
  const envUrl = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/+$/, "");

  const isDead = Boolean(
    envUrl && (
      envUrl.includes("globe-celebration-poems-shakira") ||
      envUrl.includes("trycloudflare.com") ||
      envUrl.includes("placeholder")
    )
  );

  const isLoopback = Boolean(
    envUrl && (
      envUrl.includes("127.0.0.1") ||
      envUrl.includes("localhost") ||
      envUrl.includes("0.0.0.0")
    )
  );

  const defaultBackend =
    process.env.NODE_ENV === "production"
      ? "https://fed-xray-api.onrender.com"
      : "http://127.0.0.1:8000";

  // In production cloud deployments (e.g. Vercel), server cannot reach client loopback
  if (process.env.NODE_ENV === "production" && isLoopback) {
    return defaultBackend;
  }

  return isDead || !envUrl ? defaultBackend : envUrl;
}

/**
 * Generate a standardized, RFC 7807-compatible error response for failed
 * upstream gateway forwarding without masking failure as successful HTTP 200.
 */
export function createGatewayErrorResponse(
  detail: string,
  statusCode: number = 503,
  options?: {
    errorCode?: string;
    targetEndpoint?: string;
    retryAfterSeconds?: number;
  }
): NextResponse {
  const target = options?.targetEndpoint || resolveUpstreamBackendUrl();
  const errorCode = options?.errorCode || (statusCode === 504 ? "GATEWAY_TIMEOUT" : "BACKEND_UNAVAILABLE");
  const retryAfter = options?.retryAfterSeconds ?? 5;

  const payload: GatewayErrorPayload = {
    status: statusCode === 504 ? "timeout" : "unavailable",
    detail,
    error_code: errorCode,
    target_endpoint: target,
    timestamp: Date.now() / 1000,
    retry_after_seconds: retryAfter,
  };

  return NextResponse.json(payload, {
    status: statusCode,
    headers: {
      "Cache-Control": "no-store, no-cache, must-revalidate",
      "Retry-After": String(retryAfter),
    },
  });
}

export interface ProxyForwardOptions {
  timeoutMs?: number;
  method?: string;
  body?: unknown;
  customHeaders?: Record<string, string>;
}

/**
 * Transparently forward an incoming request to the upstream FastAPI backend.
 * Preserves genuine HTTP status codes and error payloads without synthesizing
 * fake data or masking outages under HTTP 200.
 */
export async function proxyToUpstreamBackend(
  incomingRequest: Request | null,
  endpointPath: string,
  options: ProxyForwardOptions = {}
): Promise<NextResponse> {
  const upstreamBase = resolveUpstreamBackendUrl();
  const normalizedPath = endpointPath.startsWith("/") ? endpointPath : `/${endpointPath}`;
  const targetUrl = `${upstreamBase}${normalizedPath}`;

  const timeoutMs = options.timeoutMs ?? 10000;
  const method = options.method || incomingRequest?.method || "GET";

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const forwardHeaders: Record<string, string> = {
      Accept: "application/json",
      ...options.customHeaders,
    };

    let forwardBody: string | undefined = undefined;

    if (options.body !== undefined) {
      forwardHeaders["Content-Type"] = "application/json";
      forwardBody = typeof options.body === "string" ? options.body : JSON.stringify(options.body);
    } else if (incomingRequest && ["POST", "PUT", "PATCH"].includes(method.toUpperCase())) {
      try {
        const rawJson = await incomingRequest.json();
        forwardHeaders["Content-Type"] = "application/json";
        forwardBody = JSON.stringify(rawJson);
      } catch {
        // Request had no parseable JSON body; proceed with empty body
      }
    }

    const upstreamResponse = await fetch(targetUrl, {
      method,
      headers: forwardHeaders,
      body: forwardBody,
      signal: controller.signal,
      cache: "no-store",
    });

    clearTimeout(timeoutId);

    // Forward upstream response faithfully
    const contentType = upstreamResponse.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      const data = await upstreamResponse.json();
      return NextResponse.json(data, {
        status: upstreamResponse.status,
        headers: {
          "Cache-Control": "no-store, no-cache, must-revalidate",
        },
      });
    }

    // Binary or text forward
    const responseBytes = await upstreamResponse.arrayBuffer();
    return new NextResponse(responseBytes, {
      status: upstreamResponse.status,
      headers: {
        "Content-Type": contentType || "application/octet-stream",
        "Cache-Control": "no-store, no-cache, must-revalidate",
      },
    });
  } catch (error: unknown) {
    clearTimeout(timeoutId);

    const isAbort = error instanceof Error && error.name === "AbortError";
    const statusCode = isAbort ? 504 : 503;
    const errorDetail = isAbort
      ? `Upstream federated backend timed out after ${timeoutMs}ms at ${targetUrl}.`
      : `Fed-XRay federated inference backend is unreachable at ${upstreamBase}. Service is offline or container waking from sleep.`;

    return createGatewayErrorResponse(errorDetail, statusCode, {
      errorCode: isAbort ? "GATEWAY_TIMEOUT" : "BACKEND_UNAVAILABLE",
      targetEndpoint: targetUrl,
      retryAfterSeconds: isAbort ? 2 : 5,
    });
  }
}
