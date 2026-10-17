import { useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import {
  Bell,
  Boxes,
  MessageSquare,
  PanelLeftClose,
  PanelLeftOpen,
  Settings,
  Sparkles,
  Plus,
} from "lucide-react";
import { cn } from "../lib/cn";
import { Badge, Dot } from "./ui/badge";
import { Button } from "./ui/button";
import { useStatus } from "../lib/status";
import { useConversations } from "../lib/conversations";
import { shortTime } from "../lib/format";

const NAV = [
  { to: "/", label: "聊天", icon: MessageSquare, end: true },
  { to: "/models", label: "模型与状态", icon: Boxes },
  { to: "/monitor", label: "监控", icon: Bell },
  { to: "/settings", label: "设置", icon: Settings },
];

export function Shell() {
  const [open, setOpen] = useState(true);
  const { health, ok, error } = useStatus();
  const { list, activeId, select, create } = useConversations();
  const navigate = useNavigate();

  const newChat = () => {
    create();
    navigate("/");
  };

  return (
    <div className="flex h-full flex-col">
      {/* Fixed top navigation */}
      <header className="fixed inset-x-0 top-0 z-30 h-14 border-b border-white/10 bg-ink-900/80 backdrop-blur-xl">
        <div className="flex h-full items-center gap-3 px-3">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setOpen((v) => !v)}
            aria-label="切换侧边栏"
            className="shrink-0"
          >
            {open ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
          </Button>

          <div className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-gradient shadow-glow">
              <Sparkles size={15} className="text-white" />
            </span>
            <span className="text-[15px] font-semibold tracking-tight">
              Strata<span className="gradient-text">·</span>
              <span className="text-gray-200">控制台</span>
            </span>
          </div>

          <div className="ml-2 hidden min-w-0 items-center gap-2 sm:flex">
            <span className="truncate text-[13px] text-gray-400">
              {health?.model ? health.model : "未加载模型"}
            </span>
            <Badge tone={health?.backend ? "brand" : "muted"}>
              {health?.backend ?? "—"}
            </Badge>
          </div>

          <div className="ml-auto flex items-center gap-2">
            <span
              className={cn(
                "hidden items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] sm:inline-flex",
                ok
                  ? "border-ok/30 bg-ok/10 text-ok"
                  : "border-danger/30 bg-danger/10 text-danger",
              )}
            >
              <Dot tone={ok ? "ok" : "danger"} />
              {ok ? "已连接" : error ? "连接失败" : "连接中"}
            </span>
            <NavLink to="/settings">
              <Button variant="ghost" size="icon" aria-label="设置">
                <Settings size={18} />
              </Button>
            </NavLink>
          </div>
        </div>
      </header>

      <div className="flex flex-1 pt-14">
        {/* Collapsible sidebar */}
        <aside
          className={cn(
            "fixed bottom-0 left-0 top-14 z-20 flex w-64 flex-col border-r border-white/10 bg-ink-900/70 backdrop-blur-xl transition-all duration-300",
            open ? "translate-x-0" : "-translate-x-full",
          )}
        >
          <nav className="flex flex-col gap-1 p-3">
            <Button variant="primary" className="mb-2 w-full" onClick={newChat}>
              <Plus size={16} /> 新建对话
            </Button>
            {NAV.map((n) => (
              <NavLink key={n.to} to={n.to} end={n.end}>
                {({ isActive }) => (
                  <span
                    className={cn(
                      "flex items-center gap-3 rounded-xl px-3 py-2 text-[14px] transition-colors",
                      isActive
                        ? "bg-brand-500/15 text-white"
                        : "text-gray-400 hover:bg-white/5 hover:text-gray-200",
                    )}
                  >
                    <n.icon size={17} />
                    {n.label}
                  </span>
                )}
              </NavLink>
            ))}
          </nav>

          <div className="mx-3 my-1 border-t border-white/[0.08]" />

          {/* Recent conversations */}
          <div className="flex min-h-0 flex-1 flex-col px-3 py-2">
            <span className="px-1 pb-1.5 text-[11px] font-medium uppercase tracking-wider text-gray-500">
              最近会话
            </span>
            <div className="scrollbar-thin -mr-1 flex-1 space-y-0.5 overflow-y-auto">
              {list.length === 0 ? (
                <p className="px-1 text-[12.5px] text-gray-600">还没有对话，点击“新建对话”开始。</p>
              ) : (
                list.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => select(c.id)}
                    className={cn(
                      "block w-full truncate rounded-lg px-2.5 py-2 text-left text-[13px] transition-colors",
                      c.id === activeId
                        ? "bg-white/[0.07] text-gray-100"
                        : "text-gray-400 hover:bg-white/5 hover:text-gray-200",
                    )}
                  >
                    {c.title}
                    <span className="ml-2 text-[11px] text-gray-600">{shortTime(c.updatedAt)}</span>
                  </button>
                ))
              )}
            </div>
          </div>
        </aside>

        {/* Main content */}
        <main
          className={cn(
            "scrollbar-thin min-w-0 flex-1 overflow-y-auto transition-all duration-300",
            open ? "lg:pl-64" : "lg:pl-0",
          )}
        >
          <Outlet />
        </main>
      </div>
    </div>
  );
}
