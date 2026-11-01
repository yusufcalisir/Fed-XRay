import React from "react";
import { render, RenderOptions } from "@testing-library/react";
import { LanguageProvider } from "@/context/LanguageContext";

interface WrapperProps {
  children: React.ReactNode;
}

export function AllTheProviders({ children }: WrapperProps) {
  return (
    <LanguageProvider>
      {children}
    </LanguageProvider>
  );
}

export function customRender(
  ui: React.ReactElement,
  options?: Omit<RenderOptions, "wrapper">
) {
  return render(ui, { wrapper: AllTheProviders, ...options });
}

export * from "@testing-library/react";
export { customRender as render };
