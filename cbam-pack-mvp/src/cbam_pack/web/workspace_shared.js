// Shared helpers for shell workspaces.
// Keep framework-free to match existing CBAM web approach.

export function qs(id) {
  return document.getElementById(id);
}

export function setText(id, value) {
  const el = qs(id);
  if (el) el.textContent = value;
}

export function show(id) {
  const el = qs(id);
  if (el) el.style.display = 'block';
}

export function hide(id) {
  const el = qs(id);
  if (el) el.style.display = 'none';
}

export async function jsonFetch(url, options) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch (e) {
    payload = { error: 'Invalid JSON from server', raw: text?.slice(0, 500) };
  }
  return { response, payload };
}

