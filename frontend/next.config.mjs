/** @type {import('next').NextConfig} */
const rawUrl = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/$/, '') || '';
// Sanitize: never proxy to known expired or placeholder tunnel domains
const isDeadTunnel =
  rawUrl.includes('globe-celebration-poems-shakira') ||
  rawUrl.includes('trycloudflare.com/api') ||
  rawUrl.includes('placeholder');
const backendUrl = isDeadTunnel || !rawUrl ? 'http://127.0.0.1:8000' : rawUrl;

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
