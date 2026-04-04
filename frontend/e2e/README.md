# Playwright E2E — V2.2 shell

## Running locally

From `frontend/`:

```bash
npm ci
npm run test:e2e
```

With a prebuilt SPA (matches CI): build once, then set `PLAYWRIGHT_PREBUILT=1` so the webServer only starts uvicorn from the repo root.

## Visual regression baselines

`e2e/visual-shell.spec.ts` compares viewport screenshots against committed PNGs. **`v2-2-shell-quality` runs on `windows-latest`**, so baselines use Playwright’s default per-OS names (for example `*-chromium-win32.png`). Baselines live under `e2e/visual-shell.spec.ts-snapshots/`.

### Generate or update baselines

1. Install Python deps and Node deps as in `.github/workflows/v2-frontend-ux-ci.yml` (repo root: `pip install -e .` and `pip install -e "./cbam-pack-mvp[dev,web]"`).
2. `cd frontend && npm run build`
3. `cd frontend && PLAYWRIGHT_PREBUILT=1 npx playwright test e2e/visual-shell.spec.ts --update-snapshots`  
   (omit `CI=true` locally so tests execute on your OS; CI uses committed baselines only.)
4. Commit the updated `*.png` files.

Optional: run `.github/workflows/v2-shell-visual-snapshots.yml` via **Actions → Update V2 shell Playwright snapshots → Run workflow**, then download the artifact and commit the PNGs.

### Policy

- Regenerate on the **same OS as CI** when possible (Windows for current workflow), or expect snapshot renames when the runner OS changes.
- Keep `maxDiffPixels` in `playwright.config.ts` modest; investigate large diffs (theme or layout drift).
