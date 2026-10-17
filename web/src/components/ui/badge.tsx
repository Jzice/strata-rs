import { cn } from "../../lib/cn";

type Tone = "brand" | "ok" | "warn" | "danger" | "muted" | "info";

const TONES: Record<Tone, string> = {
  brand: "bg-brand-500/15 text-brand-400 border-brand-500/30",
  ok: "bg-ok/15 text-ok border-ok/30",
  warn: "bg-warn/15 text-warn border-warn/30",
  danger: "bg-danger/15 text-danger border-danger/30",
  muted: "bg-white/[0.06] text-gray-400 border-white/10",
  info: "bg-info/15 text-info border-info/30",
};

export function Badge({
  tone = "muted",
  className,
  children,
}: {
  tone?: Tone;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11px] font-medium",
        TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Dot({ tone = "muted" }: { tone?: Tone }) {
  const colors: Record<Tone, string> = {
    brand: "bg-brand-500",
    ok: "bg-ok",
    warn: "bg-warn",
    danger: "bg-danger",
    muted: "bg-gray-500",
    info: "bg-info",
  };
  return <span className={cn("h-1.5 w-1.5 rounded-full", colors[tone])} />;
}
