"use client";

import React, { useEffect, ReactNode } from "react";
import { installRuntimeShield } from "@/lib/runtimeShield";
import ClinicalErrorBoundary from "./ClinicalErrorBoundary";

interface ClientRuntimeShieldProps {
  children: ReactNode;
}

export function ClientRuntimeShield({ children }: ClientRuntimeShieldProps) {
  useEffect(() => {
    const uninstall = installRuntimeShield();
    return () => {
      uninstall();
    };
  }, []);

  return <ClinicalErrorBoundary>{children}</ClinicalErrorBoundary>;
}

export default ClientRuntimeShield;
