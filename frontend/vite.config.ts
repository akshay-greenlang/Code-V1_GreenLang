import path from "path";
import { fileURLToPath } from "url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@greenlang/shell-ui": path.resolve(__dirname, "packages/shell-ui/src/index.ts")
    }
  },
  build: {
    outDir: "dist",
    sourcemap: true
  },
  server: {
    port: 5173
  }
});
