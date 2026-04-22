import { useMemo, useState } from "react";
import { AlertOctagon, AlertTriangle, Copy, ChevronDown, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { LogLevel, ParserLogEntry } from "@/types/ops";

/**
 * Collapsible log viewer with severity filter chips and jump-to-first-error.
 * Intentionally simple — no virtualization in the MVP; swap in react-window
 * once logs routinely exceed 5k lines.
 */
const LEVELS: LogLevel[] = ["DEBUG", "INFO", "WARN", "ERROR"];

const LEVEL_COLOR: Record<LogLevel, string> = {
  DEBUG: "text-muted-foreground",
  INFO: "text-foreground",
  WARN: "text-factor-preview-700",
  ERROR: "text-factor-deprecated-700",
};

interface Props {
  entries: ParserLogEntry[];
  defaultOpen?: boolean;
}

export function ParserLogViewer({ entries, defaultOpen = true }: Props) {
  const [open, setOpen] = useState(defaultOpen);
  const [enabled, setEnabled] = useState<Set<LogLevel>>(new Set(LEVELS));

  const filtered = useMemo(
    () => entries.filter((e) => enabled.has(e.level)),
    [entries, enabled]
  );
  const firstErrorIdx = useMemo(
    () => entries.findIndex((e) => e.level === "ERROR"),
    [entries]
  );

  const toggle = (lvl: LogLevel) =>
    setEnabled((prev) => {
      const next = new Set(prev);
      if (next.has(lvl)) next.delete(lvl);
      else next.add(lvl);
      return next;
    });

  const copy = () => {
    const text = filtered.map((e) => `L${e.line} ${e.level} ${e.message}`).join("\n");
    if (navigator.clipboard) void navigator.clipboard.writeText(text);
  };

  const jumpToError = () => {
    if (firstErrorIdx === -1) return;
    const el = document.getElementById(`log-line-${entries[firstErrorIdx]?.line}`);
    el?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  return (
    <section className="rounded-lg border border-border" aria-label="Parser log">
      <header className="flex items-center justify-between border-b border-border p-2">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex items-center gap-1 text-sm font-medium"
          aria-expanded={open}
        >
          {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          Parser log
          <span className="ml-1 text-xs text-muted-foreground">
            ({filtered.length} / {entries.length})
          </span>
        </button>
        <div className="flex items-center gap-1">
          {LEVELS.map((lvl) => (
            <button
              key={lvl}
              type="button"
              onClick={() => toggle(lvl)}
              aria-pressed={enabled.has(lvl)}
              className={cn(
                "rounded px-2 py-0.5 text-xs",
                enabled.has(lvl)
                  ? "bg-accent text-foreground"
                  : "bg-muted text-muted-foreground"
              )}
            >
              {lvl}
            </button>
          ))}
          <Button variant="ghost" size="sm" onClick={jumpToError} disabled={firstErrorIdx === -1}>
            <AlertOctagon className="h-3.5 w-3.5" /> jump to error
          </Button>
          <Button variant="ghost" size="sm" onClick={copy}>
            <Copy className="h-3.5 w-3.5" /> copy
          </Button>
        </div>
      </header>
      {open && (
        <pre className="max-h-[60vh] overflow-auto bg-background p-2 font-mono text-xs leading-snug">
          {filtered.map((e) => (
            <div
              key={e.line}
              id={`log-line-${e.line}`}
              className={cn("flex gap-3", LEVEL_COLOR[e.level])}
            >
              <span className="w-10 shrink-0 select-none text-right tabular-nums text-muted-foreground">
                L{e.line}
              </span>
              <span className="w-14 shrink-0">
                <Badge
                  variant={e.level === "ERROR" ? "danger" : e.level === "WARN" ? "warn" : "muted"}
                  className="text-2xs"
                >
                  {e.level === "WARN" && <AlertTriangle className="mr-0.5 h-2.5 w-2.5" />}
                  {e.level}
                </Badge>
              </span>
              <span className="whitespace-pre-wrap break-words">{e.message}</span>
            </div>
          ))}
        </pre>
      )}
    </section>
  );
}
