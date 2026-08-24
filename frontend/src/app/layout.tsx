import type { Metadata } from "next";
import "./globals.css";
import { LanguageProvider } from "@/context/LanguageContext";
import { ApiProvider } from "@/context/ApiContext";

export const metadata: Metadata = {
  title: "Fed-XRay | AI Radiologist Network",
  description: "Privacy-Preserving Federated Medical Imaging & Multimodal CDSS Platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased selection:bg-brand-500 selection:text-white">
        <LanguageProvider>
          <ApiProvider>
            {children}
          </ApiProvider>
        </LanguageProvider>
      </body>
    </html>
  );
}
