import { Cpu, KeyRound, RotateCcw, Server, SlidersHorizontal, Sparkles } from "lucide-react";
import { useSettings } from "../lib/settings";
import { useStatus } from "../lib/status";
import { Field, Input, Slider, Textarea } from "../components/ui/field";
import { Button } from "../components/ui/button";
import { Card, CardBody, CardHeader } from "../components/ui/card";
import { Badge } from "../components/ui/badge";

export function SettingsPage() {
  const { settings, update, reset } = useSettings();
  const { status, health, ok } = useStatus();

  return (
    <div className="mx-auto max-w-3xl space-y-5 px-4 py-6 sm:px-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-[22px] font-semibold tracking-tight">设置</h1>
          <p className="mt-1 text-[13px] text-gray-400">连接到 Strata 服务并调整默认采样参数。</p>
        </div>
        <Button variant="outline" onClick={reset}>
          <RotateCcw size={15} /> 恢复默认
        </Button>
      </div>

      <Card>
        <CardHeader title="服务连接" subtitle="留空表示使用当前页面所在地址" icon={<Server size={18} />} />
        <CardBody className="space-y-3">
          <Field label="服务地址" hint={ok ? "已连接" : "未连接"}>
            <Input
              value={settings.baseUrl}
              onChange={(e) => update({ baseUrl: e.target.value })}
              placeholder="http://127.0.0.1:8080"
            />
          </Field>
          <Field label="API Key" hint="与服务端 --api-key 一致">
            <Input
              type="password"
              value={settings.apiKey}
              onChange={(e) => update({ apiKey: e.target.value })}
              placeholder="（仅绑定非回环地址时需要）"
            />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <div className="glass rounded-xl px-3 py-2.5 text-[13px]">
              <div className="text-gray-500">绑定地址</div>
              <div className="text-gray-200">
                {status?.bind ?? "—"}:{status?.port ?? "—"}
              </div>
            </div>
            <div className="glass rounded-xl px-3 py-2.5 text-[13px]">
              <div className="text-gray-500">连接状态</div>
              <div className="text-gray-200">{health?.model ? "已加载模型" : ok ? "在线" : "离线"}</div>
            </div>
          </div>
        </CardBody>
      </Card>

      <Card>
        <CardHeader title="采样参数" subtitle="新对话的默认生成配置" icon={<SlidersHorizontal size={18} />} />
        <CardBody className="space-y-4">
          <Param label="温度" value={settings.temperature} min={0} max={2} step={0.05}
            onChange={(v) => update({ temperature: v })} hint="越高越发散" />
          <Param label="Top-P" value={settings.topP} min={0} max={1} step={0.05}
            onChange={(v) => update({ topP: v })} hint="核采样" />
          <Param label="Top-K" value={settings.topK} min={0} max={100} step={1}
            onChange={(v) => update({ topK: v })} hint="0 = 关闭" />
          <Param label="重复惩罚" value={settings.repetitionPenalty} min={1} max={1.5} step={0.01}
            onChange={(v) => update({ repetitionPenalty: v })} hint="抑制重复" />
          <Param label="最大 Token" value={settings.maxTokens} min={32} max={8192} step={32}
            onChange={(v) => update({ maxTokens: v })} hint="单次上限" />
          <Param label="上下文长度" value={settings.contextLength} min={2048} max={131072} step={2048}
            onChange={(v) => update({ contextLength: v })} hint="KV 窗口" />
        </CardBody>
      </Card>

      <Card>
        <CardHeader title="系统提示词" subtitle="作为每条对话的 system 角色" icon={<Sparkles size={18} />} />
        <CardBody>
          <Textarea
            rows={3}
            value={settings.systemPrompt}
            onChange={(e) => update({ systemPrompt: e.target.value })}
            placeholder="你是一个有用的本地助手…"
          />
        </CardBody>
      </Card>

      <Card>
        <CardHeader title="关于" icon={<Cpu size={18} />} />
        <CardBody>
          <div className="flex items-center gap-3">
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-gradient shadow-glow">
              <Sparkles size={18} className="text-white" />
            </span>
            <div>
              <div className="text-[15px] font-semibold">
                Strata<span className="gradient-text">·</span> 控制台
              </div>
              <div className="text-[12px] text-gray-400">
                本地 MoE 大模型推理引擎 · 版本 {status?.version ?? "—"}
              </div>
            </div>
            <Badge tone="brand" className="ml-auto">
              {health?.backend ?? "CPU"} 后端
            </Badge>
          </div>
          <p className="mt-3 text-[12.5px] leading-relaxed text-gray-500">
            本控制台由 React + TypeScript + Tailwind 构建，由 Rust 推理服务托管。所有生成都在你的机器上完成。
          </p>
        </CardBody>
      </Card>

      <div className="flex items-center gap-2 text-[12px] text-gray-600">
        <KeyRound size={13} /> 设置仅保存在本机浏览器中，不会上传。
      </div>
    </div>
  );
}

function Param({
  label,
  value,
  min,
  max,
  step,
  hint,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  hint?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-4">
      <div className="w-28 shrink-0">
        <div className="text-[13px] font-medium text-gray-300">{label}</div>
        {hint ? <div className="text-[11px] text-gray-600">{hint}</div> : null}
      </div>
      <div className="flex flex-1 items-center gap-3">
        <Slider value={value} min={min} max={max} step={step} onChange={onChange} />
        <span className="w-14 shrink-0 text-right text-[13px] tabular-nums text-gray-200">{value}</span>
      </div>
    </div>
  );
}
