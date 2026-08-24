import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Deep medical navy palette
        navy: {
          950: "#040711",
          900: "#0a0f1e",
          800: "#0f1629",
          700: "#151d35",
          600: "#1b2540",
          500: "#243056",
        },
        // Accent brand teal for medical trust
        accent: {
          50: "#ecfdf5",
          100: "#d1fae5",
          200: "#a7f3d0",
          300: "#6ee7b7",
          400: "#34d399",
          500: "#10b981",
          600: "#059669",
          700: "#047857",
        },
        // Clinical status colors
        status: {
          healthy: "#10b981",
          warning: "#f59e0b",
          critical: "#ef4444",
          info: "#3b82f6",
          active: "#8b5cf6",
        },
        // Chart palette
        chart: {
          loss: "#f87171",
          accuracy: "#34d399",
          f1: "#60a5fa",
          precision: "#a78bfa",
          recall: "#fbbf24",
        },
      },
      fontFamily: {
        sans: ["'Inter'", "system-ui", "sans-serif"],
        display: ["'Outfit'", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
      borderRadius: {
        "2xl": "16px",
        "3xl": "20px",
        "4xl": "24px",
      },
      boxShadow: {
        card: "0 1px 3px rgba(0,0,0,0.3), 0 4px 20px rgba(0,0,0,0.15)",
        glow: "0 0 30px rgba(16,185,129,0.15)",
        "glow-accent": "0 0 40px rgba(16,185,129,0.2)",
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-in": "fadeIn 0.3s ease-out forwards",
        "slide-up": "slideUp 0.4s ease-out forwards",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      spacing: {
        "18": "4.5rem",
      },
    },
  },
  plugins: [],
};
export default config;
