import path from "path";
import { fileURLToPath } from "url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@greenlang/shell-ui": path.resolve(__dirname, "packages/shell-ui/src/index.ts"),
      // Local path alias so the frontend consumes v1.2.0 SDK types from
      // the monorepo source directly without waiting on an npm publish.
      "@greenlang/factors-sdk": path.resolve(__dirname, "../greenlang/factors/sdk/ts/src/index.ts")
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
