import type { Metadata } from "next";
import "./globals.css";
import { LanguageProvider } from "@/context/LanguageContext";
import { ApiProvider } from "@/context/ApiContext";
import ClientRuntimeShield from "@/components/ClientRuntimeShield";

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
      <body className="antialiased" suppressHydrationWarning>
        <LanguageProvider>
          <ApiProvider>
            <ClientRuntimeShield>
              {children}
            </ClientRuntimeShield>
          </ApiProvider>
        </LanguageProvider>
      </body>
    </html>
  );
}
