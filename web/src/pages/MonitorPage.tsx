import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  ExternalLink,
  Gauge,
  MessageSquare,
  Zap,
} from "lucide-react";
import { api } from "../lib/api";
import { useSettings } from "../lib/settings";
import type { Monitor, RequestRecord } from "../lib/types";
import { number, shortTime } from "../lib/format";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export function MonitorPage() {
  const { settings } = useSettings();
  const [mon, setMon] = useState<Monitor | null>(null);
  const [reqs, setReqs] = useState<RequestRecord[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const [m, r] = await Promise.all([api.monitor(settings), api.requests(settings, 40)]);
        if (!alive) return;
        setMon(m);
        setReqs(r.requests);
        setErr(null);
      } catch (e) {
        if (alive) setErr(e instanceof Error ? e.message : "加载失败");
      }
    };
    tick();
    const id = window.setInterval(tick, 3000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [settings]);

  const chart = reqs.map((r, i) => ({
    name: `#${reqs.length - i}`,
    ms: Math.round(r.ms),
    ok: r.status < 400,
  }));

  return (
    <div className="mx-auto max-w-5xl space-y-5 px-4 py-6 sm:px-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-[22px] font-semibold tracking-tight">运行监控</h1>
          <p className="mt-1 text-[13px] text-gray-400">实时请求吞吐、延迟与错误情况。</p>
        </div>
        <Link to="/metrics" target="_blank">
          <Button variant="outline">
            <ExternalLink size={15} /> Prometheus 指标
          </Button>
        </Link>
      </div>

      {err ? (
        <div className="glass rounded-xl border border-danger/30 bg-danger/10 px-4 py-3 text-[13px] text-danger">
          {err}
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard icon={<Activity size={16} />} label="总请求" value={number(mon?.requests ?? 0)} tone="brand" />
        <StatCard icon={<MessageSquare size={16} />} label="对话补全" value={number(mon?.chat_completions ?? 0)} />
        <StatCard
          icon={<Zap size={16} />}
          label="生成速度"
          value={`${(mon?.gen_tokens_per_s ?? 0).toFixed(1)}`}
          unit="tok/s"
        />
        <StatCard
          icon={<AlertTriangle size={16} />}
          label="错误"
          value={number(mon?.errors ?? 0)}
          tone={mon && mon.errors > 0 ? "danger" : "ok"}
        />
      </div>

      <Card>
        <CardHeader
          title="最近请求延迟"
          subtitle="每次请求耗时（毫秒）"
          icon={<Gauge size={18} />}
        />
        <CardBody>
          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chart} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="name" tick={{ fill: "#6B7280", fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fill: "#6B7280", fontSize: 11 }} tickLine={false} axisLine={false} width={42} />
                <Tooltip
                  contentStyle={{
                    background: "#111827",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: 12,
                    fontSize: 12,
                  }}
                  labelStyle={{ color: "#9CA3AF" }}
                  formatter={(v: number) => [`${v} ms`, "耗时"]}
                />
                <Bar dataKey="ms" radius={[4, 4, 0, 0]}>
                  {chart.map((c, i) => (
                    <Cell key={i} fill={c.ok ? "#6366F1" : "#EF4444"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader title="请求日志" subtitle={`最近 ${reqs.length} 条`} icon={<BarChart3 size={18} />} />
        <CardBody className="p-0">
          <div className="scrollbar-thin max-h-[22rem] overflow-y-auto">
            <table className="w-full text-left text-[12.5px]">
              <thead className="sticky top-0 bg-ink-800/90 text-[11px] uppercase tracking-wider text-gray-500 backdrop-blur">
                <tr>
                  <Th>时间</Th>
                  <Th>方法</Th>
                  <Th>路径</Th>
                  <Th>状态</Th>
                  <Th>耗时</Th>
                  <Th>Token</Th>
                </tr>
              </thead>
              <tbody>
                {reqs.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-4 py-6 text-center text-gray-600">
                      暂无请求记录
                    </td>
                  </tr>
                ) : (
                  reqs.map((r, i) => (
                    <tr key={i} className="border-t border-white/[0.05] hover:bg-white/[0.03]">
                      <Td>{shortTime(Date.now() - (mon?.uptime_ms ?? 0) + r.at_ms)}</Td>
                      <Td className="font-medium text-gray-300">{r.method}</Td>
                      <Td className="max-w-[14rem] truncate text-gray-300">{r.path}</Td>
                      <Td>
                        <Badge tone={r.status < 400 ? "ok" : "danger"}>{r.status}</Badge>
                      </Td>
                      <Td className="tabular-nums text-gray-400">{r.ms.toFixed(0)}ms</Td>
                      <Td className="tabular-nums text-gray-400">
                        {r.prompt_tokens + r.gen_tokens || "—"}
                      </Td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </CardBody>
      </Card>
    </div>
  );
}

function StatCard({
  icon,
  label,
  value,
  unit,
  tone = "brand",
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  unit?: string;
  tone?: "brand" | "ok" | "danger";
}) {
  const ring = {
    brand: "bg-brand-500/15 text-brand-400",
    ok: "bg-ok/15 text-ok",
    danger: "bg-danger/15 text-danger",
  }[tone];
  return (
    <div className="glass flex items-center gap-3 rounded-2xl p-3.5">
      <span className={`flex h-10 w-10 items-center justify-center rounded-xl ${ring}`}>{icon}</span>
      <div>
        <div className="text-[11.5px] text-gray-500">{label}</div>
        <div className="text-[18px] font-semibold text-gray-100">
          {value}
          {unit ? <span className="ml-1 text-[12px] font-normal text-gray-500">{unit}</span> : null}
        </div>
      </div>
    </div>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return <th className="px-4 py-2.5 font-medium">{children}</th>;
}

function Td({ children, className }: { children: React.ReactNode; className?: string }) {
  return <td className={`px-4 py-2.5 ${className ?? ""}`}>{children}</td>;
}
