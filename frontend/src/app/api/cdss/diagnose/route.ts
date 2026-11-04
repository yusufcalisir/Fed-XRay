import { NextRequest } from "next/server";
import { proxyToUpstreamBackend } from "@/lib/backendGateway";

export const dynamic = "force-dynamic";

/**
 * Diagnostic inference reverse-proxy endpoint.
 * Forwards scan inference request to FastAPI /api/cdss/diagnose.
 * Returns authentic model logits, Grad-CAM maps, and clinical findings.
 * Fails honestly with HTTP 503 if backend is unavailable or sleeping.
 */
export async function POST(request: NextRequest) {
  return proxyToUpstreamBackend(request, "/api/cdss/diagnose", { timeoutMs: 60000 });
}
