import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { tokenize } from "../lib/highlight";
import { cn } from "../lib/cn";

function CodeBlock({ code, lang }: { code: string; lang: string }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };

  return (
    <div className="my-3 overflow-hidden rounded-xl border border-white/10 bg-ink-900/80">
      <div className="flex items-center justify-between border-b border-white/[0.08] px-3 py-1.5">
        <span className="font-mono text-[11px] uppercase tracking-wider text-gray-400">
          {lang || "code"}
        </span>
        <button
          type="button"
          onClick={copy}
          className="inline-flex cursor-pointer items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px]
            text-gray-400 transition-colors hover:bg-white/10 hover:text-white"
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? "已复制" : "复制"}
        </button>
      </div>
      <pre className="scrollbar-thin overflow-x-auto px-4 py-3 text-[13px] leading-relaxed">
        <code className="font-mono text-gray-200">
          {tokenize(code).map((t, i) => (
            <span key={i} className={t.cls}>
              {t.text}
            </span>
          ))}
        </code>
      </pre>
    </div>
  );
}

/** Inline emphasis: `code`, **bold**, *italic* and [links](url). */
function inline(text: string, keyPrefix: string): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  const re = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*]+\*)|(\[[^\]]+\]\([^)]+\))/g;
  let last = 0;
  let m: RegExpExecArray | null;

  while ((m = re.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    const [full, code, bold, italic, link] = m;
    const key = `${keyPrefix}-${m.index}`;
    if (code) {
      nodes.push(
        <code
          key={key}
          className="rounded-md bg-white/[0.09] px-1.5 py-0.5 font-mono text-[12.5px] text-brand-400"
        >
          {code.slice(1, -1)}
        </code>,
      );
    } else if (bold) {
      nodes.push(
        <strong key={key} className="font-semibold text-white">
          {bold.slice(2, -2)}
        </strong>,
      );
    } else if (italic) {
      nodes.push(
        <em key={key} className="italic text-gray-200">
          {italic.slice(1, -1)}
        </em>,
      );
    } else if (link) {
      const match = /\[([^\]]+)\]\(([^)]+)\)/.exec(link);
      nodes.push(
        <a
          key={key}
          href={match?.[2] ?? "#"}
          target="_blank"
          rel="noreferrer"
          className="cursor-pointer text-brand-400 underline decoration-brand-500/40 underline-offset-2 hover:text-brand-500"
        >
          {match?.[1] ?? link}
        </a>,
      );
    }
    last = m.index + full.length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

/**
 * A compact markdown renderer: fenced code, headings, lists, quotes, rules and
 * paragraphs. Enough for chat answers without pulling a full parser in.
 */
export function Markdown({ content }: { content: string }) {
  const lines = content.split("\n");
  const blocks: React.ReactNode[] = [];
  let i = 0;
  let key = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim().startsWith("```")) {
      const lang = line.trim().slice(3).trim();
      const body: string[] = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        body.push(lines[i]);
        i += 1;
      }
      i += 1;
      blocks.push(<CodeBlock key={`cb-${key++}`} code={body.join("\n")} lang={lang} />);
      continue;
    }

    if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
      blocks.push(<hr key={`hr-${key++}`} className="my-4 border-white/10" />);
      i += 1;
      continue;
    }

    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      const level = heading[1].length;
      const sizes = ["text-[19px]", "text-[17px]", "text-[15px]", "text-[14px]"];
      blocks.push(
        <p
          key={`h-${key++}`}
          className={cn("mt-4 mb-1.5 font-semibold tracking-tight text-white", sizes[level - 1])}
        >
          {inline(heading[2], `h${key}`)}
        </p>,
      );
      i += 1;
      continue;
    }

    if (/^\s*[-*+]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*+]\s+/, ""));
        i += 1;
      }
      blocks.push(
        <ul key={`ul-${key++}`} className="my-2 list-disc space-y-1 pl-5 text-[14px] text-gray-200">
          {items.map((t, n) => (
            <li key={n}>{inline(t, `li${key}-${n}`)}</li>
          ))}
        </ul>,
      );
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ""));
        i += 1;
      }
      blocks.push(
        <ol key={`ol-${key++}`} className="my-2 list-decimal space-y-1 pl-5 text-[14px] text-gray-200">
          {items.map((t, n) => (
            <li key={n}>{inline(t, `oli${key}-${n}`)}</li>
          ))}
        </ol>,
      );
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*>\s?/, ""));
        i += 1;
      }
      blocks.push(
        <blockquote
          key={`bq-${key++}`}
          className="my-3 rounded-r-lg border-l-2 border-brand-500/60 bg-white/[0.04] px-3 py-2 text-[14px] text-gray-300"
        >
          {inline(items.join(" "), `bq${key}`)}
        </blockquote>,
      );
      continue;
    }

    if (line.trim() === "") {
      i += 1;
      continue;
    }

    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !lines[i].trim().startsWith("```") &&
      !/^(#{1,4})\s+/.test(lines[i]) &&
      !/^\s*[-*+]\s+/.test(lines[i]) &&
      !/^\s*\d+\.\s+/.test(lines[i]) &&
      !/^\s*>\s?/.test(lines[i])
    ) {
      para.push(lines[i]);
      i += 1;
    }
    blocks.push(
      <p key={`p-${key++}`} className="my-1.5 text-[14px] leading-[1.75] text-gray-200">
        {inline(para.join(" "), `p${key}`)}
      </p>,
    );
  }

  return <div className="text-[14px]">{blocks}</div>;
}
