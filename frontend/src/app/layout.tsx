import type { Metadata, Viewport } from "next";
import "./globals.css";
import { LanguageProvider } from "@/context/LanguageContext";
import { ApiProvider } from "@/context/ApiContext";
import ClientRuntimeShield from "@/components/ClientRuntimeShield";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export const metadata: Metadata = {
  title: "Fed-XRay | Decentralized Healthcare AI Infrastructure",
  description: "Privacy-Preserving Federated Medical Imaging & Multimodal CDSS Platform",
  icons: {
    icon: [
      { url: "/icon.svg?v=4", type: "image/svg+xml" },
      { url: "/favicon.ico?v=4", sizes: "any" },
    ],
    shortcut: "/icon.svg?v=4",
    apple: "/icon.svg?v=4",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased relative min-h-screen" suppressHydrationWarning>
        <LanguageProvider>
          <ApiProvider>
            <ClientRuntimeShield>
              {/* Technical Precision Blueprint Grid Pattern */}
              <div
                className="fixed inset-0 pointer-events-none z-0 bg-grid-pattern bg-grid-mask opacity-80 dark:opacity-90"
                aria-hidden="true"
              />
              <div className="relative z-10 flex flex-col min-h-screen">
                {children}
              </div>
            </ClientRuntimeShield>
          </ApiProvider>
        </LanguageProvider>
      </body>
    </html>
  );
}
