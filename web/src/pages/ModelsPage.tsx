import { useState } from "react";
import {
  Boxes,
  Cpu,
  HardDrive,
  Layers,
  Loader2,
  Power,
  PowerOff,
  Database,
} from "lucide-react";
import { useStatus } from "../lib/status";
import { api } from "../lib/api";
import { useSettings } from "../lib/settings";
import { bytes, duration, number } from "../lib/format";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Field, Input } from "../components/ui/field";

export function ModelsPage() {
  const { status, health, error, refresh } = useStatus();
  const { settings } = useSettings();
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const loaded = status?.model ?? null;

  const doLoad = async () => {
    const target = path.trim();
    if (!target) return;
    setBusy(true);
    setMsg(null);
    try {
      await api.load(settings, target);
      setPath("");
      setMsg(`已加载 ${target}`);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "加载失败");
    } finally {
      setBusy(false);
      refresh();
    }
  };

  const doUnload = async () => {
    setBusy(true);
    setMsg(null);
    try {
      await api.unload(settings);
      setMsg("已卸载模型");
    } catch (e) {
      setMsg(e instanceof Error ? e.message : "卸载失败");
    } finally {
      setBusy(false);
      refresh();
    }
  };

  return (
    <div className="mx-auto max-w-4xl space-y-5 px-4 py-6 sm:px-8">
      <div>
        <h1 className="text-[22px] font-semibold tracking-tight">模型与状态</h1>
        <p className="mt-1 text-[13px] text-gray-400">
          管理本地 GGUF 模型的加载与卸载，查看引擎运行信息。
        </p>
      </div>

      {loaded ? (
        <Card>
          <CardHeader
            title={loaded.name}
            subtitle={`${loaded.architecture} · ${loaded.family}`}
            icon={<Boxes size={18} />}
            action={
              <Badge tone="ok">
                <span className="h-1.5 w-1.5 rounded-full bg-ok" /> 已加载
              </Badge>
            }
          />
          <CardBody className="space-y-5">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat icon={<Layers size={15} />} label="层数" value={number(loaded.layers)} />
              <Stat icon={<Database size={15} />} label="词表" value={number(loaded.vocabulary)} />
              <Stat icon={<Cpu size={15} />} label="专家" value={`${loaded.experts_used}/${loaded.experts}`} />
              <Stat
                icon={<HardDrive size={15} />}
                label="量化"
                value={loaded.quant}
              />
            </div>

            {/* Memory / file mapping bar */}
            <div>
              <div className="mb-1.5 flex items-center justify-between text-[12.5px] text-gray-400">
                <span>GGUF 文件映射</span>
                <span className="tabular-nums text-gray-300">{bytes(loaded.mapped_bytes)}</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-white/[0.08]">
                <div
                  className="h-full rounded-full bg-brand-gradient"
                  style={{ width: `${Math.min(100, (loaded.mapped_bytes / (16 * 1024 ** 3)) * 100)}%` }}
                />
              </div>
              <p className="mt-1 text-[11px] text-gray-600">
                上下文长度 {loaded.context_length.toLocaleString()} · 加载耗时 {(loaded.loaded_ms / 1000).toFixed(2)}s
              </p>
            </div>

            <div className="flex items-center gap-2">
              <Button variant="danger" onClick={doUnload} disabled={busy}>
                {busy ? <Loader2 size={15} className="animate-spin" /> : <PowerOff size={15} />}
                卸载模型
              </Button>
              <Badge tone="muted">后端：{health?.backend ?? "—"}</Badge>
            </div>
          </CardBody>
        </Card>
      ) : (
        <Card>
          <CardHeader title="未加载模型" subtitle="选择一个 GGUF 文件路径开始对话" icon={<Boxes size={18} />} />
          <CardBody>
            <div className="flex items-center gap-3 rounded-xl border border-warn/30 bg-warn/10 px-3 py-2.5 text-[13px] text-warn">
              <Power size={16} /> {error ? `连接失败：${error}` : "当前没有模型驻留内存。"}
            </div>
          </CardBody>
        </Card>
      )}

      {/* Load a model by file path */}
      <Card>
        <CardHeader title="加载模型" subtitle="输入本机上的 GGUF 文件路径（含分片首文件）" icon={<HardDrive size={18} />} />
        <CardBody className="space-y-3">
          <Field label="模型路径" hint="例如 D:\\models\\qwen35moe-q4_k.gguf">
            <Input
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="D:\\models\\*.gguf"
              onKeyDown={(e) => e.key === "Enter" && doLoad()}
            />
          </Field>
          <Button variant="primary" onClick={doLoad} disabled={busy || !path.trim()}>
            {busy ? <Loader2 size={15} className="animate-spin" /> : <Power size={15} />}
            加载
          </Button>
          {msg ? <p className="text-[12.5px] text-gray-400">{msg}</p> : null}
        </CardBody>
      </Card>

      {/* System info */}
      <Card>
        <CardHeader title="引擎信息" icon={<Cpu size={18} />} />
        <CardBody>
          <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-[13px] sm:grid-cols-3">
            <Info k="版本" v={status?.version ?? "—"} />
            <Info k="后端" v={health?.backend ?? "—"} />
            <Info k="绑定" v={`${status?.bind ?? "—"}:${status?.port ?? "—"}`} />
            <Info k="运行时长" v={status ? duration(status.uptime_ms) : "—"} />
            <Info k="总请求" v={number(status?.requests ?? 0)} />
            <Info k="生成 Token" v={number(status?.gen_tokens ?? 0)} />
          </dl>
        </CardBody>
      </Card>
    </div>
  );
}

function Stat({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="glass rounded-xl px-3 py-2.5">
      <div className="mb-1 flex items-center gap-1.5 text-[11.5px] text-gray-500">
        {icon} {label}
      </div>
      <div className="truncate text-[15px] font-semibold text-gray-100">{value}</div>
    </div>
  );
}

function Info({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-center justify-between gap-2 border-b border-white/[0.06] py-1.5">
      <dt className="text-gray-500">{k}</dt>
      <dd className="truncate text-gray-200">{v}</dd>
    </div>
  );
}
