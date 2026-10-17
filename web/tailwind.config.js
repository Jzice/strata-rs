import animate from "tailwindcss-animate";

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Noto Sans"', "Inter", "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      colors: {
        ink: { 900: "#0B0F19", 800: "#111827", 700: "#1F2937" },
        brand: { 400: "#8B5CF6", 500: "#6366F1", 600: "#4F46E5" },
        ok: "#10B981",
        danger: "#EF4444",
        warn: "#F59E0B",
        info: "#38BDF8",
      },
      backgroundImage: {
        "brand-gradient": "linear-gradient(120deg, #6366F1 0%, #8B5CF6 55%, #4F46E5 100%)",
        "panel-sheen": "linear-gradient(160deg, rgba(255,255,255,0.07) 0%, rgba(255,255,255,0) 45%)",
      },
      boxShadow: {
        glow: "0 0 42px -14px rgba(99,102,241,0.65)",
        panel: "0 24px 60px -32px rgba(0,0,0,0.85)",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-soft": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.35" },
        },
        shimmer: {
          "100%": { transform: "translateX(100%)" },
        },
      },
      animation: {
        "fade-up": "fade-up 260ms cubic-bezier(0.22,1,0.36,1) both",
        "pulse-soft": "pulse-soft 1.6s ease-in-out infinite",
        shimmer: "shimmer 1.8s infinite",
      },
    },
  },
  plugins: [animate],
};
