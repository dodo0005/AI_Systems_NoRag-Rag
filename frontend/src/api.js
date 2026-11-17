export const API = "/api";

export async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || `Request failed (${response.status})`);
  }

  return data;
}

// Error messages from /api/ask end with "(trace <id>)".
export function traceIdIn(message) {
  const match = /trace ([0-9a-f]{12})/.exec(message || "");
  return match ? match[1] : null;
}

export function formatMs(value) {
  if (value === null || value === undefined) return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

export function formatUsd(value) {
  if (!value) return "$0";
  return value < 0.01 ? `$${value.toFixed(6)}` : `$${value.toFixed(4)}`;
}
