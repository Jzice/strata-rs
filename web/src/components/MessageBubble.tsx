import { useState } from "react";
import { Bot, ChevronDown, RefreshCw, User } from "lucide-react";
import type { Turn } from "../lib/types";
import { Markdown } from "./Markdown";
import { cn } from "../lib/cn";
import { Button } from "./ui/button";

/**
 * Splits a model answer into an optional `<think>…</think>` reasoning block
 * and the visible answer. Reasoning models emit this; plain ones don't.
 */
function splitReasoning(text: string): { reasoning: string; answer: string } {
  const m = text.match(/^([\s\S]*?)<think>([\s\S]*?)<\/think>([\s\S]*)$/);
  if (m) return { reasoning: m[2].trim(), answer: (m[1] + m[3]).trim() };
  const open = text.indexOf("<think>");
  if (open !== -1) {
    const head = text.slice(0, open);
    return { reasoning: text.slice(open + 7), answer: head };
  }
  return { reasoning: "", answer: text };
}

export function MessageBubble({
  turn,
  streaming,
  onRegenerate,
}: {
  turn: Turn;
  streaming?: boolean;
  onRegenerate?: () => void;
}) {
  const [showReasoning, setShowReasoning] = useState(false);
  const isUser = turn.role === "user";
  const { reasoning, answer } = splitReasoning(turn.content);
  const hasReasoning = reasoning.length > 0 && !isUser;

  if (isUser) {
    return (
      <div className="flex animate-fade-up justify-end gap-3">
        <div className="max-w-[78%] rounded-2xl rounded-tr-md bg-brand-gradient px-4 py-2.5 text-[14px] leading-relaxed text-white shadow-glow">
          <p className="whitespace-pre-wrap">{turn.content}</p>
        </div>
        <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-white/10 text-gray-200">
          <User size={16} />
        </span>
      </div>
    );
  }

  return (
    <div className="flex animate-fade-up gap-3">
      <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-brand-500/15 text-brand-400">
        <Bot size={16} />
      </span>
      <div className="min-w-0 max-w-[82%]">
        {hasReasoning ? (
          <div className="mb-2 overflow-hidden rounded-xl border border-white/[0.08] bg-white/[0.03]">
            <button
              type="button"
              onClick={() => setShowReasoning((v) => !v)}
              className="flex w-full cursor-pointer items-center gap-2 px-3 py-1.5 text-[12px] text-gray-400 transition-colors hover:bg-white/[0.05]"
            >
              <ChevronDown
                size={14}
                className={cn("transition-transform", showReasoning && "rotate-180")}
              />
              推理过程
            </button>
            {showReasoning ? (
              <div className="scrollbar-thin max-h-64 overflow-y-auto border-t border-white/[0.06] px-3 py-2 text-[13px] leading-relaxed text-gray-400">
                <Markdown content={reasoning} />
              </div>
            ) : null}
          </div>
        ) : null}

        <div
          className={cn(
            "rounded-2xl rounded-tl-md border border-white/[0.08] bg-ink-800/50 px-4 py-2.5",
            streaming && answer.length === 0 && "caret",
          )}
        >
          {answer.length === 0 && !streaming ? (
            <p className="text-[14px] text-gray-500">（空回复）</p>
          ) : (
            <Markdown content={streaming ? `${answer}` : answer} />
          )}
          {streaming && answer.length > 0 ? <span className="caret" /> : null}
        </div>

        <div className="mt-1.5 flex items-center gap-3 pl-1 text-[11px] text-gray-500">
          {turn.tokensPerSecond ? (
            <span className="tabular-nums text-brand-400">
              {turn.tokensPerSecond.toFixed(1)} tok/s
            </span>
          ) : null}
          {turn.elapsedMs ? <span>{((turn.elapsedMs / 1000) | 0).toFixed(1)}s</span> : null}
          {!streaming && onRegenerate ? (
            <button
              type="button"
              onClick={onRegenerate}
              className="inline-flex cursor-pointer items-center gap-1 transition-colors hover:text-gray-300"
            >
              <RefreshCw size={12} /> 重生成
            </button>
          ) : null}
        </div>
      </div>
    </div>
  );
}
