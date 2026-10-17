/** Small presentation helpers shared by the dashboard pages. */

export function bytes(n: number): string {
  if (!n) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const i = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / 1024 ** i).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

export function duration(ms: number): string {
  const s = Math.floor(ms / 1000);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

export function shortTime(ms: number): string {
  const d = new Date(ms);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(
    d.getSeconds(),
  ).padStart(2, "0")}`;
}

export function number(n: number): string {
  return new Intl.NumberFormat("zh-CN").format(n);
}

export function uid(prefix: string): string {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

/** Strips markdown noise to derive a conversation title from the first turn. */
export function titleFrom(text: string): string {
  const firstLine = text.trim().split("\n").find((l) => l.trim().length > 0) ?? "新会话";
  const clean = firstLine.replace(/[#>*`_]/g, "").trim();
  return clean.length > 42 ? `${clean.slice(0, 42)}…` : clean || "新会话";
}
