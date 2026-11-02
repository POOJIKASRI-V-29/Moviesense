import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development, /api and /health are proxied to the FastAPI backend (no CORS needed).
// For a separately hosted build, set VITE_API_URL (see .env.example).
const backend = process.env.MOVIESENSE_BACKEND_URL || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": backend, "/health": backend },
  },
});
