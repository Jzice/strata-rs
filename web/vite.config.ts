import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The app talks to the Rust server through the dev proxy; in production the
// server itself serves the built assets from the same origin.
const target = process.env.STRATA_URL ?? "http://127.0.0.1:8080";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    allowedHosts: true,
    port: 5173,
    proxy: {
      "/v1": { target, changeOrigin: true },
      "/api": { target, changeOrigin: true },
      "/api-monitor": { target, changeOrigin: true },
      "/metrics": { target, changeOrigin: true },
      "/mcp": { target, changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200,
  },
});
