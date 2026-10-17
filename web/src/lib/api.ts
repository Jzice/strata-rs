import type { ChatMessage, Health, LoadResult, ModelEntry, Monitor, RequestRecord, Status } from "./types";
import type { Settings } from "./settings";

/** Resolves an API path against the configured base URL (empty = same origin). */
export function url(baseUrl: string, path: string): string {
  const base = baseUrl.trim().replace(/\/+$/, "");
  return `${base}${path}`;
}

function headers(settings: Settings, json: boolean): Record<string, string> {
  const h: Record<string, string> = {};
  if (json) h["Content-Type"] = "application/json";
  if (settings.apiKey.trim()) h.Authorization = `Bearer ${settings.apiKey.trim()}`;
  return h;
}

async function get<T>(settings: Settings, path: string): Promise<T> {
  const res = await fetch(url(settings.baseUrl, path), {
    headers: headers(settings, false),
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`${path} → ${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

export const api = {
  health: (s: Settings) => get<Health>(s, "/health"),
  status: (s: Settings) => get<Status>(s, "/v1/status"),
  models: (s: Settings) => get<{ object: string; data: ModelEntry[] }>(s, "/v1/models"),
  monitor: (s: Settings) => get<Monitor>(s, "/api-monitor"),
  requests: (s: Settings, limit = 50) =>
    get<{ requests: RequestRecord[] }>(s, `/api/requests?limit=${limit}`),

  async load(s: Settings, model: string, nCtx?: number): Promise<LoadResult> {
    const res = await fetch(url(s.baseUrl, "/v1/load"), {
      method: "POST",
      headers: headers(s, true),
      body: JSON.stringify({ model, n_ctx: nCtx }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body?.error?.message ?? `load failed: ${res.status}`);
    return body as LoadResult;
  },

  async unload(s: Settings): Promise<void> {
    const res = await fetch(url(s.baseUrl, "/v1/unload"), {
      method: "POST",
      headers: headers(s, true),
      body: "{}",
    });
    if (!res.ok) throw new Error(`unload failed: ${res.status}`);
  },
};

export interface StreamHandle {
  abort: () => void;
}

export interface StreamResult {
  text: string;
  finishReason: string | null;
  chunks: number;
  elapsedMs: number;
}

export interface StreamOptions {
  settings: Settings;
  messages: ChatMessage[];
  model?: string;
  signal: AbortSignal;
  onDelta: (chunk: string) => void;
}

/**
 * Streams `/v1/chat/completions` over SSE. The server emits OpenAI-shaped
 * chunks (`choices[0].delta.content`) and terminates with `data: [DONE]`.
 */
export async function streamChat(opts: StreamOptions): Promise<StreamResult> {
  const { settings, messages, signal, onDelta } = opts;
  const started = performance.now();
  const res = await fetch(url(settings.baseUrl, "/v1/chat/completions"), {
    method: "POST",
    headers: headers(settings, true),
    signal,
    body: JSON.stringify({
      model: opts.model || undefined,
      messages,
      stream: true,
      temperature: settings.temperature,
      top_p: settings.topP,
      top_k: settings.topK,
      max_tokens: settings.maxTokens,
      repetition_penalty: settings.repetitionPenalty,
    }),
  });

  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body?.error?.message ?? `chat failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  let chunks = 0;
  let finishReason: string | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by a blank line.
    let split = buffer.indexOf("\n\n");
    while (split !== -1) {
      const event = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      for (const line of event.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload || payload === "[DONE]") continue;
        try {
          const json = JSON.parse(payload);
          const delta = json?.choices?.[0]?.delta?.content;
          if (typeof delta === "string" && delta.length > 0) {
            text += delta;
            chunks += 1;
            onDelta(delta);
          }
          const reason = json?.choices?.[0]?.finish_reason;
          if (typeof reason === "string") finishReason = reason;
        } catch {
          // A partial frame: ignore and keep buffering.
        }
      }
      split = buffer.indexOf("\n\n");
    }
  }

  return { text, finishReason, chunks, elapsedMs: performance.now() - started };
}
