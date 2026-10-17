import { createContext, useContext, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import type { Health, Status } from "./types";
import { useSettings } from "./settings";

interface StatusValue {
  health: Health | null;
  status: Status | null;
  ok: boolean;
  loading: boolean;
  error: string | null;
  refresh: () => void;
  lastChecked: number;
}

const Ctx = createContext<StatusValue>({
  health: null,
  status: null,
  ok: false,
  loading: true,
  error: null,
  refresh: () => {},
  lastChecked: 0,
});

export function StatusProvider({ children }: { children: React.ReactNode }) {
  const { settings } = useSettings();
  const [health, setHealth] = useState<Health | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState(0);
  const timer = useRef<number | null>(null);

  const refresh = useMemo(
    () => async () => {
      try {
        const [h, s] = await Promise.all([api.health(settings), api.status(settings)]);
        setHealth(h);
        setStatus(s);
        setError(null);
      } catch (e) {
        // console.error: surface a connection failure without throwing.
        console.error("status poll failed", e);
        setError(e instanceof Error ? e.message : "无法连接服务");
      } finally {
        setLoading(false);
        setLastChecked(Date.now());
      }
    },
    [settings],
  );

  useEffect(() => {
    refresh();
    timer.current = window.setInterval(refresh, 5000);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [refresh]);

  const value = useMemo<StatusValue>(
    () => ({ health, status, ok: !!health, loading, error, refresh, lastChecked }),
    [health, status, loading, error, refresh, lastChecked],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useStatus(): StatusValue {
  return useContext(Ctx);
}
