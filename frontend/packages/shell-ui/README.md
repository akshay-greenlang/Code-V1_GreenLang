# @greenlang/shell-ui

Internal design tokens and MUI theme for the GreenLang V2.2 enterprise shell. **Semver:** follow the shell app release; breaking token or theme changes should bump minor/major in lockstep with operator-facing release notes.

## Usage

The Vite shell resolves `@greenlang/shell-ui` to `packages/shell-ui/src` via `resolve.alias`. In another package, depend on this workspace path or publish path and import:

```ts
import { createShellTheme, shellColorTokens } from "@greenlang/shell-ui";
import { ThemeProvider } from "@mui/material/styles";

const theme = createShellTheme();

export function App() {
  return (
    <ThemeProvider theme={theme}>
      {/* ... */}
    </ThemeProvider>
  );
}
```

## Contents

- `tokens.ts` — color, radius, focus ring primitives.
- `theme.ts` — `createShellTheme()` / `shellTheme` with AppBar and outlined-input contrast tuned for dark chrome.
- `ShellPrimitives.stories.tsx` — Storybook demos (`npm run storybook` from `frontend/`).

## Changing tokens

1. Update `tokens.ts` / `theme.ts` and verify Storybook + shell build.
2. Re-run Playwright axe suites; adjust contrast if color-contrast rules fail.
3. If pixel baselines exist, regenerate on Linux (see `frontend/e2e/README.md`).
