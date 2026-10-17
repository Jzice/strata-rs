import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { Conversation, Turn } from "./types";
import { uid } from "./format";

const KEY = "strata.web.conversations.v1";

interface ConversationsValue {
  list: Conversation[];
  activeId: string | null;
  active: Conversation | null;
  create: (systemPrompt?: string) => Conversation;
  remove: (id: string) => void;
  select: (id: string) => void;
  appendTurn: (convId: string, turn: Turn) => void;
  updateTurn: (convId: string, turnId: string, patch: Partial<Turn>) => void;
  rename: (convId: string, title: string) => void;
  clearAll: () => void;
}

const Ctx = createContext<ConversationsValue>({
  list: [],
  activeId: null,
  active: null,
  create: () => ({} as Conversation),
  remove: () => {},
  select: () => {},
  appendTurn: () => {},
  updateTurn: () => {},
  rename: () => {},
  clearAll: () => {},
});

function load(): Conversation[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Conversation[]) : [];
  } catch {
    return [];
  }
}

export function ConversationsProvider({ children }: { children: React.ReactNode }) {
  const [list, setList] = useState<Conversation[]>(load);
  const [activeId, setActiveId] = useState<string | null>(() => load()[0]?.id ?? null);

  useEffect(() => {
    localStorage.setItem(KEY, JSON.stringify(list));
  }, [list]);

  const value = useMemo<ConversationsValue>(() => {
    const update = (id: string, fn: (c: Conversation) => Conversation) =>
      setList((prev) => prev.map((c) => (c.id === id ? fn(c) : c)));

    return {
      list,
      activeId,
      active: list.find((c) => c.id === activeId) ?? null,
      select: (id) => setActiveId(id),
      create: (systemPrompt) => {
        const conv: Conversation = {
          id: uid("conv"),
          title: "新会话",
          turns: [],
          updatedAt: Date.now(),
          systemPrompt: systemPrompt ?? "",
        };
        setList((prev) => [conv, ...prev]);
        setActiveId(conv.id);
        return conv;
      },
      remove: (id) => {
        setList((prev) => prev.filter((c) => c.id !== id));
        setActiveId((cur) => (cur === id ? null : cur));
      },
      appendTurn: (convId, turn) =>
        update(convId, (c) => ({ ...c, turns: [...c.turns, turn], updatedAt: Date.now() })),
      updateTurn: (convId, turnId, patch) =>
        update(convId, (c) => ({
          ...c,
          turns: c.turns.map((t) => (t.id === turnId ? { ...t, ...patch } : t)),
          updatedAt: Date.now(),
        })),
      rename: (convId, title) => update(convId, (c) => ({ ...c, title })),
      clearAll: () => {
        setList([]);
        setActiveId(null);
      },
    };
  }, [list, activeId]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useConversations(): ConversationsValue {
  return useContext(Ctx);
}
