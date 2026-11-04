import { NextRequest } from "next/server";
import { proxyToUpstreamBackend } from "@/lib/backendGateway";

export const dynamic = "force-dynamic";

/**
 * Cohort generation reverse-proxy endpoint.
 * Forwards request to FastAPI /api/cohorts/generate for authentic Strategy E
 * clinical data ingestion. Returns genuine status code and fails honestly
 * with HTTP 503 if backend is unavailable, eliminating fake static fixture
 * substitution under DEFECT-01.
 */
export async function POST(request: NextRequest) {
  return proxyToUpstreamBackend(request, "/api/cohorts/generate", { timeoutMs: 15000 });
}
