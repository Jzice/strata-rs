import { useEffect, useRef, useState } from "react";
import { Cpu, Send, Square, SlidersHorizontal } from "lucide-react";
import { useConversations } from "../lib/conversations";
import { useSettings } from "../lib/settings";
import { useStatus } from "../lib/status";
import { streamChat } from "../lib/api";
import type { ChatMessage, Conversation, Turn } from "../lib/types";
import { titleFrom, uid } from "../lib/format";
import { MessageBubble } from "../components/MessageBubble";
import { Button } from "../components/ui/button";
import { Slider, Textarea } from "../components/ui/field";

function buildMessages(systemPrompt: string, turns: Turn[]): ChatMessage[] {
  const m: ChatMessage[] = [];
  if (systemPrompt.trim()) m.push({ role: "system", content: systemPrompt });
  for (const t of turns) if (t.content) m.push({ role: t.role, content: t.content });
  return m;
}

export function ChatPage() {
  const { active, appendTurn, updateTurn, create, rename } = useConversations();
  const { settings, update } = useSettings();
  const status = useStatus();

  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showParams, setShowParams] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  const turns = active?.turns ?? [];

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, busy]);

  const generate = async (convId: string, messages: ChatMessage[], assistantId?: string) => {
    setBusy(true);
    setError(null);
    const aId = assistantId ?? uid("t");
    if (!assistantId) appendTurn(convId, { id: aId, role: "assistant", content: "", at: Date.now() });
    else updateTurn(convId, aId, { content: "", elapsedMs: undefined, tokensPerSecond: undefined });

    const ac = new AbortController();
    abortRef.current = ac;
    let acc = "";

    try {
      const res = await streamChat({
        settings,
        messages,
        model: status.health?.model ?? undefined,
        signal: ac.signal,
        onDelta: (d) => {
          acc += d;
          updateTurn(convId, aId, { content: acc });
        },
      });
      const tps = res.elapsedMs > 0 ? res.chunks / (res.elapsedMs / 1000) : 0;
      updateTurn(convId, aId, {
        elapsedMs: res.elapsedMs,
        tokensPerSecond: tps,
        finishReason: res.finishReason,
      });
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        const msg = e instanceof Error ? e.message : "生成失败";
        updateTurn(convId, aId, { content: `${acc}\n\n⚠️ ${msg}` });
        setError(msg);
      } else {
        updateTurn(convId, aId, { content: acc });
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
      status.refresh();
    }
  };

  const send = () => {
    const content = input.trim();
    if (!content || busy) return;
    let convId = active?.id;
    let sys = active?.systemPrompt ?? "";
    let base: Turn[] = active?.turns ?? [];
    if (!convId) {
      const c: Conversation = create();
      convId = c.id;
      sys = c.systemPrompt;
    }
    const userTurn: Turn = { id: uid("t"), role: "user", content, at: Date.now() };
    appendTurn(convId, userTurn);
    if (base.length === 0) rename(convId, titleFrom(content));
    setInput("");
    const full = [...base, userTurn];
    generate(convId, buildMessages(sys, full));
  };

  const regenerate = () => {
    if (!active || busy) return;
    const last = turns[turns.length - 1];
    if (!last || last.role !== "assistant") return;
    const trimmed = turns.slice(0, -1);
    generate(active.id, buildMessages(active.systemPrompt, trimmed), last.id);
  };

  const stop = () => abortRef.current?.abort();

  return (
    <div className="flex h-[calc(100vh-3.5rem)] flex-col">
      {/* Message stream */}
      <div className="scrollbar-thin flex-1 space-y-5 overflow-y-auto px-4 py-6 sm:px-8">
        {turns.length === 0 ? (
          <div className="mx-auto mt-10 max-w-xl text-center">
            <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-gradient shadow-glow">
              <Cpu size={24} className="text-white" />
            </div>
            <h1 className="text-[22px] font-semibold tracking-tight gradient-text">本地大模型，即刻对话</h1>
            <p className="mt-2 text-[14px] leading-relaxed text-gray-400">
              Strata 在你的机器上以 {status.health?.backend ?? "CPU"} 后端运行
              {status.health?.model ? ` · 模型 ${status.health.model}` : " · 尚未加载模型"}。
              所有推理都在本地完成，数据不会离开本机。
            </p>
            <div className="mt-5 grid grid-cols-1 gap-2 text-left sm:grid-cols-2">
              {[
                "用 Rust 重写的高性能推理引擎",
                "支持 Qwen3.8-Flash-Next 与 Qwen3.6-35B-A3B",
                "OpenAI / Anthropic 兼容 API",
                "流式输出 · 思维链可折叠",
              ].map((f) => (
                <div key={f} className="glass rounded-xl px-3 py-2 text-[13px] text-gray-300">
                  {f}
                </div>
              ))}
            </div>
          </div>
        ) : (
          turns.map((t) => (
            <MessageBubble
              key={t.id}
              turn={t}
              streaming={busy && t.id === turns[turns.length - 1]?.id && t.role === "assistant"}
              onRegenerate={t.role === "assistant" && !busy ? regenerate : undefined}
            />
          ))
        )}
        <div ref={endRef} />
      </div>

      {/* Composer */}
      <div className="border-t border-white/10 bg-ink-900/70 px-4 py-3 backdrop-blur-xl sm:px-8">
        <div className="mx-auto max-w-3xl">
          {error ? (
            <p className="mb-2 rounded-lg border border-danger/30 bg-danger/10 px-3 py-1.5 text-[12.5px] text-danger">
              {error}
            </p>
          ) : null}
          <div className="glass relative flex items-end gap-2 rounded-2xl p-2">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              rows={1}
              placeholder="给 Strata 发消息…（Enter 发送，Shift+Enter 换行）"
              className="max-h-40 min-h-[2.5rem] flex-1 border-0 bg-transparent focus:ring-0"
            />
            <div className="flex items-center gap-1.5">
              <Button
                variant="ghost"
                size="icon"
                aria-label="采样参数"
                onClick={() => setShowParams((v) => !v)}
                className={showParams ? "text-brand-400" : ""}
              >
                <SlidersHorizontal size={18} />
              </Button>
              {busy ? (
                <Button variant="danger" size="icon" onClick={stop} aria-label="停止">
                  <Square size={16} />
                </Button>
              ) : (
                <Button variant="primary" size="icon" onClick={send} aria-label="发送" disabled={!input.trim()}>
                  <Send size={16} />
                </Button>
              )}
            </div>
          </div>

          {showParams ? (
            <div className="glass mt-2 grid grid-cols-1 gap-3 rounded-xl p-3 sm:grid-cols-3">
              {(
                [
                  ["temperature", "温度", 0, 2, 0.05],
                  ["topP", "Top-P", 0, 1, 0.05],
                  ["maxTokens", "最大 Token", 32, 4096, 32],
                ] as const
              ).map(([key, label, min, max, step]) => (
                <div key={key}>
                  <div className="mb-1 flex items-center justify-between text-[12px] text-gray-400">
                    <span>{label}</span>
                    <span className="tabular-nums text-gray-300">
                      {(settings as unknown as Record<string, number>)[key]}
                    </span>
                  </div>
                  <Slider
                    value={(settings as unknown as Record<string, number>)[key]}
                    min={min}
                    max={max}
                    step={step}
                    onChange={(v) => update({ [key]: v } as never)}
                  />
                </div>
              ))}
            </div>
          ) : null}
          <p className="mt-1.5 text-center text-[11px] text-gray-600">
            本地推理 · 上下文上限 {settings.contextLength.toLocaleString()} tokens
          </p>
        </div>
      </div>
    </div>
  );
}
