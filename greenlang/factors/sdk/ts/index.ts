/**
 * Re-export shim for backward compatibility.
 *
 * The real implementation lives under `src/`. This file is kept so
 * legacy imports (`import { FactorsClient } from './index'`) continue
 * to resolve; new code should import from `src/index` (or the package
 * entry `@greenlang/factors`).
 */
export * from './src/index';
export { default } from './src/index';
