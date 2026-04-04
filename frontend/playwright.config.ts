import path from "path";
import { fileURLToPath } from "url";
import { defineConfig, devices } from "@playwright/test";

const frontendDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(frontendDir, "..");

/**
 * When PLAYWRIGHT_PREBUILT=1 (e.g. CI after `npm run build`), only start uvicorn — avoids
 * rebuilding inside webServer and reduces flakes. Local default runs build + server.
 */
const prebuilt = process.env.PLAYWRIGHT_PREBUILT === "1";

const webServerCommand = prebuilt
  ? `python -m uvicorn cbam_pack.web.app:create_app --factory --host 127.0.0.1 --port 4179`
  : `npm run build && cd .. && python -m uvicorn cbam_pack.web.app:create_app --factory --host 127.0.0.1 --port 4179`;

const webServerCwd = prebuilt ? repoRoot : frontendDir;

const cbamSrc = path.join(repoRoot, "cbam-pack-mvp", "src");
const webServerEnv = {
  ...process.env,
  PYTHONPATH: [cbamSrc, repoRoot, process.env.PYTHONPATH ?? ""].filter(Boolean).join(path.delimiter),
  // Expose deterministic run fixtures for Playwright (no effect in normal deployments).
  GL_SHELL_E2E_FIXTURES: process.env.GL_SHELL_E2E_FIXTURES ?? "1"
};

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:4179",
    trace: "on-first-retry"
  },
  webServer: {
    command: webServerCommand,
    cwd: webServerCwd,
    env: webServerEnv,
    url: "http://127.0.0.1:4179/health",
    reuseExistingServer: !process.env.CI,
    // Cold Python imports (greenlang) can exceed 120s on Windows/CI agents.
    timeout: prebuilt ? 600_000 : 600_000
  },
  expect: {
    toHaveScreenshot: {
      maxDiffPixels: 120
    }
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }]
});
