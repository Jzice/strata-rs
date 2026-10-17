import { createContext, useContext, useEffect, useMemo, useState } from "react";

export interface Settings {
  /** Empty means "the origin the page was served from". */
  baseUrl: string;
  apiKey: string;
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
  contextLength: number;
  repetitionPenalty: number;
  systemPrompt: string;
}

export const DEFAULT_SETTINGS: Settings = {
  baseUrl: "",
  apiKey: "",
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  maxTokens: 512,
  contextLength: 32768,
  repetitionPenalty: 1.05,
  systemPrompt: "You are Strata, a helpful local assistant running entirely on this machine.",
};

const KEY = "strata.web.settings.v1";

interface SettingsContextValue {
  settings: Settings;
  update: (patch: Partial<Settings>) => void;
  reset: () => void;
}

const SettingsContext = createContext<SettingsContextValue>({
  settings: DEFAULT_SETTINGS,
  update: () => {},
  reset: () => {},
});

function load(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return DEFAULT_SETTINGS;
    return { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<Settings>) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = useState<Settings>(load);

  useEffect(() => {
    localStorage.setItem(KEY, JSON.stringify(settings));
  }, [settings]);

  const value = useMemo<SettingsContextValue>(
    () => ({
      settings,
      update: (patch) => setSettings((s) => ({ ...s, ...patch })),
      reset: () => setSettings(DEFAULT_SETTINGS),
    }),
    [settings],
  );

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings(): SettingsContextValue {
  return useContext(SettingsContext);
}
