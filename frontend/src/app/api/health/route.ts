import { proxyToUpstreamBackend } from "@/lib/backendGateway";

export const dynamic = "force-dynamic";

/**
 * Health check reverse-proxy endpoint.
 * Transparently queries the active Python FastAPI backend /api/health endpoint.
 * Returns HTTP 503 Service Unavailable if the backend is offline or unresponsive,
 * strictly eliminating false "healthy" standby responses under DEFECT-01.
 */
export async function GET() {
  return proxyToUpstreamBackend(null, "/api/health", { timeoutMs: 2500 });
}
