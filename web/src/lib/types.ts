/** Wire types mirroring the Rust server's JSON responses. */

export type Role = "system" | "user" | "assistant";

export interface ChatMessage {
  role: Role;
  content: string;
}

export interface Health {
  status: string;
  loaded: boolean;
  model: string | null;
  backend: string;
  uptime_ms: number;
  version: string;
}

export interface ModelStatus {
  name: string;
  path: string;
  family: string;
  architecture: string;
  layers: number;
  n_embd: number;
  experts: number;
  experts_used: number;
  context_length: number;
  vocabulary: number;
  quant: string;
  mapped_bytes: number;
  loaded_ms: number;
}

export interface Status {
  loaded: boolean;
  model: ModelStatus | null;
  backend: string;
  bind: string;
  port: number;
  uptime_ms: number;
  requests: number;
  errors: number;
  prompt_tokens: number;
  gen_tokens: number;
  version: string;
}

export interface ModelEntry {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  strata?: {
    family: string;
    architecture: string;
    context_length: number;
    backend: string;
  };
}

export interface RequestRecord {
  id: number;
  method: string;
  path: string;
  status: number;
  ms: number;
  prompt_tokens: number;
  gen_tokens: number;
  at_ms: number;
  streamed: boolean;
  model: string | null;
}

export interface Monitor {
  uptime_ms: number;
  backend: string;
  requests: number;
  errors: number;
  chat_completions: number;
  completions: number;
  messages: number;
  responses: number;
  streamed: number;
  prompt_tokens: number;
  gen_tokens: number;
  gen_seconds: number;
  gen_tokens_per_s: number;
  recent: RequestRecord[];
}

export interface LoadResult {
  status: string;
  model: string;
  path: string;
  family: string;
  context_length: number;
  backend: string;
}

/** One turn of a conversation as the UI stores it. */
export interface Turn {
  id: string;
  role: Role;
  content: string;
  /** Wall-clock ms the answer took, assistant turns only. */
  elapsedMs?: number;
  /** Tokens per second, assistant turns only. */
  tokensPerSecond?: number;
  finishReason?: string | null;
  at: number;
}

export interface Conversation {
  id: string;
  title: string;
  turns: Turn[];
  updatedAt: number;
  systemPrompt: string;
}
