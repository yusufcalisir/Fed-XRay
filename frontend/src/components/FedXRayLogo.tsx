"use client";

import React from "react";

export interface FedXRayLogoProps {
  size?: number;
  className?: string;
  showText?: boolean;
}

/**
 * Fed-XRay Official Clinical Emblem.
 * Symbolizes:
 * 1. Privacy-Preserving Federated Enclave (Clinical Security Shield)
 * 2. Precision Radiological Sensing & Diagnostic Pulse (Waveform Sensor)
 */
export default function FedXRayLogo({
  size = 24,
  className = "",
  showText = false,
}: FedXRayLogoProps) {
  return (
    <div className={`inline-flex items-center gap-2 select-none ${className}`}>
      {/* Precision Medical Shield & Radiant Scan Emblem */}
      <svg
        width={size}
        height={size}
        viewBox="0 0 32 32"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="shrink-0"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id="fedXrayShieldGrad" x1="16" y1="4" x2="16" y2="28" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#10b981" />
            <stop offset="100%" stopColor="#06b6d4" />
          </linearGradient>
        </defs>

        {/* Clinical Protection Shield Contour */}
        <path
          d="M16 4.5L24.5 8.2V15C24.5 20.8 20.8 25.2 16 27.5C11.2 25.2 7.5 20.8 7.5 15V8.2L16 4.5Z"
          stroke="url(#fedXrayShieldGrad)"
          strokeWidth="2.2"
          strokeLinejoin="round"
        />

        {/* Internal Diagnostic Pulse & Precision Saliency Wave */}
        <path
          d="M11 15H13.2L14.8 11.8L17.2 18.2L18.8 15H21"
          stroke="#34d399"
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>

      {showText && (
        <span className="font-display font-extrabold text-[15px] sm:text-[16px] tracking-tight text-[var(--text-heading)] whitespace-nowrap">
          Fed<span className="text-emerald-500 font-semibold">-</span>XRay
        </span>
      )}
    </div>
  );
}
