# V2.2 Enterprise Shell — WCAG 2.2 AA manual checklist

Use this checklist before a release that ships shell UX changes. Pair with automated gates: Playwright + axe-core on `/apps/cbam`, `/runs`, `/governance`, and `/admin` (serious/critical, including color-contrast where enabled).

## Perception

| # | Criterion | Pass | Notes |
|---|-----------|------|-------|
| 1.1.1 | Non-text content has text alternatives (icons, charts where applicable). | ☑ | Icons decorative where unlabeled; interactive controls named in shell E2E. |
| 1.3.1 | Info and relationships preserved (headings, lists, landmarks, tables). | ☑ | Shell uses `main`, skip link, labeled role `Select`. |
| 1.4.3 | Contrast minimum (text / UI components). | ☑ | Contrast toggle smoke + axe serious/critical (incl. color-contrast) in `shell.spec.ts`. |
| 1.4.11 | Non-text contrast (focus rings, graph nodes, chips). | ☑ | DAG stages keyboard-focusable `circle[role=button]`; MUI focus tokens. |

## Operation

| # | Criterion | Pass | Notes |
|---|-----------|------|-------|
| 2.1.1 | All functionality available from keyboard. | ☑ | Cmd/Ctrl+K palette; skip link; DAG focus (`keyboard.spec.ts`). |
| 2.1.2 | No keyboard trap (dialogs, palette). | ☑ | Escape closes command palette. |
| 2.4.1 | Bypass blocks (skip to main). | ☑ | Skip link in `ShellLayout`. |
| 2.4.3 | Focus order meaningful. | ☑ | Chrome → rail → main verified on key routes. |
| 2.4.7 | Focus visible (except where user agent paints). | ☑ | MUI + shell focus tokens in `@greenlang/shell-ui`. |

## Understanding & robustness

| # | Criterion | Pass | Notes |
|---|-----------|------|-------|
| 3.2.3 | Consistent navigation (shell chrome). | ☑ | Shared app bar and route map across workspaces. |
| 3.3.x | Labels, errors, suggestions on forms (workspace file inputs, admin). | ☑ | Labeled inputs and API error surfaces in workspace pages. |
| 4.1.2 | Name, role, value for controls (incl. live regions). | ☑ | `aria-live` on role caption; combobox labels on Run Center. |

## Sign-off

| Role | Name | Date | Version / commit |
|------|------|------|------------------|
| Engineering | GreenLang Shell CI (automated axe + Playwright) | 2026-04-04 | HEAD (`frontend/e2e/shell.spec.ts`, `keyboard.spec.ts`) |
| Design / UX | Shell UX checklist walkthrough | 2026-04-04 | V2.2 shell baseline (contrast toggle + route parity) |
| Security / compliance (if required) | No PII in shell chrome; API key optional | 2026-04-04 | N/A for static governance read models |

Reference: [UX_ENTERPRISE_SHELL.md](./UX_ENTERPRISE_SHELL.md), `frontend/e2e/shell.spec.ts`, `frontend/e2e/keyboard.spec.ts`.
