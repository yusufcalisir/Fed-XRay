/** @type {import('next').NextConfig} */
const rawUrl = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/$/, '') || '';
// Sanitize: never proxy to known expired or placeholder tunnel domains
const isDeadTunnel =
  rawUrl.includes('globe-celebration-poems-shakira') ||
  rawUrl.includes('trycloudflare.com') ||
  rawUrl.includes('placeholder');
const defaultBackend =
  process.env.NODE_ENV === 'production'
    ? 'https://fed-xray-api.onrender.com'
    : 'http://127.0.0.1:8000';
const backendUrl = isDeadTunnel || !rawUrl ? defaultBackend : rawUrl;

const nextConfig = {
  output: process.env.DOCKER_BUILD === '1' ? 'standalone' : undefined,
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
