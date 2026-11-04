import { NextRequest } from "next/server";
import { proxyToUpstreamBackend } from "@/lib/backendGateway";

export const dynamic = "force-dynamic";

/**
 * Case-based retrieval (Federated RAG) reverse-proxy endpoint.
 * Retrieves verified twin cases from FastAPI /api/cdss/rag-similar.
 * Fails honestly with HTTP 503 if the backend is unavailable or times out,
 * eliminating false HTTP 200 empty responses under DEFECT-01.
 */
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const topK = searchParams.get("top_k") || "2";
  return proxyToUpstreamBackend(request, `/api/cdss/rag-similar?top_k=${topK}`, { timeoutMs: 15000 });
}
