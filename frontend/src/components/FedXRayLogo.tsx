"use client";

import React from "react";

export interface FedXRayLogoProps {
  size?: number;
  className?: string;
  showText?: boolean;
}

/**
 * Fed-XRay Precision Clinical & Federated Emblem.
 * Symbolizes:
 * 1. Decentralized Multi-Hospital Consortium (4 Cryptographic Institutional Nodes)
 * 2. Precision Radiological Penetration & Cross-Mesh (The Dynamic 'X' Beam)
 * 3. Thoracic Oncology & Anatomical Radiography (Bilateral Lung Field Silhouette)
 * 4. Zero-Knowledge Consensus Core (Central Radiance & Aperture Beacon)
 */
export default function FedXRayLogo({
  size = 24,
  className = "",
  showText = false,
}: FedXRayLogoProps) {
  return (
    <div className={`inline-flex items-center gap-2 select-none ${className}`}>
      {/* Precision Medical X-Ray & Federated Network Emblem */}
      <svg
        width={size}
        height={size}
        viewBox="0 0 32 32"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="shrink-0 drop-shadow-[0_1px_4px_rgba(16,185,129,0.25)]"
        aria-hidden="true"
      >
        <defs>
          <linearGradient id="fedxray-grad-primary" x1="4" y1="4" x2="28" y2="28" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#10b981" />
            <stop offset="50%" stopColor="#06b6d4" />
            <stop offset="100%" stopColor="#3b82f6" />
          </linearGradient>
          <linearGradient id="fedxray-grad-secondary" x1="28" y1="4" x2="4" y2="28" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#34d399" />
            <stop offset="50%" stopColor="#06b6d4" />
            <stop offset="100%" stopColor="#0ea5e9" />
          </linearGradient>
          <linearGradient id="fedxray-lung-left" x1="4.8" y1="5.5" x2="10.5" y2="26.5" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.75" />
          </linearGradient>
          <linearGradient id="fedxray-lung-right" x1="27.2" y1="5.5" x2="21.5" y2="26.5" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#34d399" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0.75" />
          </linearGradient>
        </defs>

        {/* Bilateral Thoracic Radiography Arcs (Chest X-Ray Pulmonary Contour) */}
        <path
          d="M10.5 5.5C6.8 8.8 4.8 12.2 4.8 16C4.8 19.8 6.8 23.2 10.5 26.5"
          stroke="url(#fedxray-lung-left)"
          strokeWidth="2.2"
          strokeLinecap="round"
        />
        <path
          d="M21.5 5.5C25.2 8.8 27.2 12.2 27.2 16C27.2 19.8 25.2 23.2 21.5 26.5"
          stroke="url(#fedxray-lung-right)"
          strokeWidth="2.2"
          strokeLinecap="round"
        />

        {/* Dynamic Diagonal X-Ray Beams (The Federated 'X') */}
        <path
          d="M7.5 7.5L12.5 12.5M19.5 19.5L24.5 24.5"
          stroke="url(#fedxray-grad-primary)"
          strokeWidth="2.8"
          strokeLinecap="round"
        />
        <path
          d="M24.5 7.5L19.5 12.5M12.5 19.5L7.5 24.5"
          stroke="url(#fedxray-grad-secondary)"
          strokeWidth="2.8"
          strokeLinecap="round"
        />

        {/* Central Consensus Aperture Ring & Radiance Sensor */}
        <circle
          cx="16"
          cy="16"
          r="4.6"
          stroke="url(#fedxray-grad-primary)"
          strokeWidth="1.8"
          fill="none"
        />
        <circle
          cx="16"
          cy="16"
          r="2.2"
          fill="#10b981"
        />

        {/* 4 Decentralized Institutional Hospital Nodes */}
        <circle cx="7.5" cy="7.5" r="2.4" fill="#10b981" />
        <circle cx="24.5" cy="7.5" r="2.4" fill="#34d399" />
        <circle cx="7.5" cy="24.5" r="2.4" fill="#06b6d4" />
        <circle cx="24.5" cy="24.5" r="2.4" fill="#3b82f6" />
      </svg>

      {showText && (
        <span className="font-display font-extrabold text-[15px] sm:text-[16px] tracking-tight text-[var(--text-heading)] whitespace-nowrap">
          Fed<span className="text-emerald-500 font-semibold">-</span>XRay
        </span>
      )}
    </div>
  );
}
