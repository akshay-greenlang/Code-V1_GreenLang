/**
 * editionStore — tiny localStorage-backed store for the operator's pinned
 * `X-GL-Edition` header. Kept dependency-free (no Zustand / Context wrapper)
 * so it can be read by the API client without React imports.
 *
 * The Factors API treats `X-GL-Edition` as an optional override; if absent,
 * the server resolves the active edition for the caller's tenant.
 */

const STORAGE_KEY = "gl.factors.pinnedEdition";

type Listener = (edition: string | null) => void;
const listeners = new Set<Listener>();

export function getPinnedEdition(): string | null {
  if (typeof window === "undefined") return null;
  try {
    const v = window.localStorage.getItem(STORAGE_KEY);
    return v && v.trim().length > 0 ? v : null;
  } catch {
    return null;
  }
}

export function setPinnedEdition(edition: string | null): void {
  if (typeof window === "undefined") return;
  try {
    if (edition && edition.trim().length > 0) {
      window.localStorage.setItem(STORAGE_KEY, edition.trim());
    } else {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  } catch {
    /* ignore quota / disabled storage */
  }
  listeners.forEach((fn) => fn(getPinnedEdition()));
}

export function subscribePinnedEdition(fn: Listener): () => void {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}
