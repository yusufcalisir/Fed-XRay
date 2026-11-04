import { proxyToUpstreamBackend } from "@/lib/backendGateway";

export const dynamic = "force-dynamic";

/**
 * System state reverse-proxy endpoint.
 * Fetches authoritative consortium state from FastAPI /api/state.
 * Fails honestly with HTTP 503 if the backend is unreachable, strictly
 * eliminating mock cohort fixture substitution under DEFECT-01.
 */
export async function GET() {
  return proxyToUpstreamBackend(null, "/api/state", { timeoutMs: 3000 });
}
