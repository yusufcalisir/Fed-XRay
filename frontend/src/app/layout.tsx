import type { Metadata } from "next";
import "./globals.css";
import { LanguageProvider } from "@/context/LanguageContext";
import { ApiProvider } from "@/context/ApiContext";

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
    <html lang="en">
      <body className="antialiased">
        <LanguageProvider>
          <ApiProvider>
            {children}
          </ApiProvider>
        </LanguageProvider>
      </body>
    </html>
  );
}
