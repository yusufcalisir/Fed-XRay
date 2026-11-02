"use client";

import React from "react";

export interface FedXRayLogoProps {
  size?: number;
  className?: string;
  showText?: boolean;
}

export default function FedXRayLogo({
  size = 28,
  className = "",
  showText = false,
}: FedXRayLogoProps) {
  return (
    <div className={`inline-flex items-center gap-2 select-none ${className}`}>
      {/* Precision Federated X-Ray Monogram */}
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
          <linearGradient id="fxRayBeam1" x1="6" y1="6" x2="26" y2="26" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#10b981" />
            <stop offset="100%" stopColor="#06b6d4" />
          </linearGradient>
          <linearGradient id="fxRayBeam2" x1="26" y1="6" x2="6" y2="26" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#38bdf8" />
            <stop offset="100%" stopColor="#10b981" />
          </linearGradient>
        </defs>

        {/* Diagonal 1: Primary Sensing Beam (Top-Left to Bottom-Right) */}
        <path
          d="M7.5 7.5L24.5 24.5"
          stroke="url(#fxRayBeam1)"
          strokeWidth="3.6"
          strokeLinecap="round"
        />

        {/* Diagonal 2: Federated Aggregation Tunnel with Enclave Focal Gap */}
        <path
          d="M24.5 7.5L19 13"
          stroke="url(#fxRayBeam2)"
          strokeWidth="3.6"
          strokeLinecap="round"
        />
        <path
          d="M13 19L7.5 24.5"
          stroke="url(#fxRayBeam2)"
          strokeWidth="3.6"
          strokeLinecap="round"
        />

        {/* Central Privacy Enclave & Focal Calibration Beacon */}
        <circle cx="16" cy="16" r="2.2" fill="#10b981" />
      </svg>

      {showText && (
        <span className="font-display font-extrabold text-[15px] sm:text-[16px] tracking-tight text-[var(--text-heading)] whitespace-nowrap">
          Fed<span className="text-emerald-500 font-semibold">-</span>XRay
        </span>
      )}
    </div>
  );
}
