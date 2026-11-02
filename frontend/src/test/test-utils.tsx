import React from "react";
import { render, RenderOptions } from "@testing-library/react";
import { LanguageProvider } from "@/context/LanguageContext";
import { ApiProvider } from "@/context/ApiContext";

interface WrapperProps {
  children: React.ReactNode;
}

export function AllTheProviders({ children }: WrapperProps) {
  return (
    <ApiProvider>
      <LanguageProvider>
        {children}
      </LanguageProvider>
    </ApiProvider>
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
