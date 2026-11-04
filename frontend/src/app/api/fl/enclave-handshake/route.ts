import { proxyToUpstreamBackend } from "@/lib/backendGateway";

export const dynamic = "force-dynamic";

/**
 * Enclave cryptographic handshake reverse-proxy endpoint.
 * Fetches Option J cryptographic verification from FastAPI /api/fl/enclave-handshake.
 * Fails honestly with HTTP 503 if the backend is unavailable, eliminating
 * hardcoded SHA-256 digests and canned node fingerprints under DEFECT-01.
 */
export async function GET() {
  return proxyToUpstreamBackend(null, "/api/fl/enclave-handshake", { timeoutMs: 5000 });
}
