import { Badge } from "@/components/ui/badge";
import { truncateHash } from "@/lib/utils";
import type { FactorDiff } from "@/types/ops";

/**
 * Field-level diff for a single factor between two editions.
 * Same visual grammar as the Explorer's read-only DiffTable, but this version
 * allows inline "revert this field" actions (surfaced by the ops console only).
 */
interface Props {
  diff: FactorDiff;
  onRevertField?: (field: string) => void;
  editable?: boolean;
}

function renderValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

const TYPE_VARIANT: Record<
  "added" | "removed" | "changed",
  "success" | "danger" | "warn"
> = {
  added: "success",
  removed: "danger",
  changed: "warn",
};

export function DiffTable({ diff, onRevertField, editable = false }: Props) {
  return (
    <section className="space-y-3">
      <header className="flex flex-wrap items-center justify-between gap-2 text-sm">
        <div>
          <span className="font-mono">{diff.factor_id}</span>{" "}
          <Badge variant={diff.status === "changed" ? "warn" : "muted"}>{diff.status}</Badge>
        </div>
        <div className="text-xs text-muted-foreground">
          {diff.left_edition} → {diff.right_edition}
          {diff.left_content_hash && diff.right_content_hash && (
            <span className="ml-2 font-mono">
              {truncateHash(diff.left_content_hash)} → {truncateHash(diff.right_content_hash)}
            </span>
          )}
        </div>
      </header>
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead className="bg-muted/50 text-left">
            <tr>
              <th className="px-3 py-2 w-48">field</th>
              <th className="px-3 py-2">old</th>
              <th className="px-3 py-2">new</th>
              <th className="px-3 py-2 w-24">type</th>
              {editable && <th className="px-3 py-2 w-24" />}
            </tr>
          </thead>
          <tbody>
            {diff.changes.map((c) => (
              <tr key={c.field} className="border-t border-border align-top">
                <td className="px-3 py-2 font-mono text-xs">{c.field}</td>
                <td className="px-3 py-2 font-mono text-xs text-muted-foreground">
                  {renderValue(c.old_value)}
                </td>
                <td className="px-3 py-2 font-mono text-xs">{renderValue(c.new_value)}</td>
                <td className="px-3 py-2">
                  <Badge variant={TYPE_VARIANT[c.type]}>{c.type}</Badge>
                </td>
                {editable && onRevertField && (
                  <td className="px-3 py-2 text-right">
                    <button
                      type="button"
                      className="text-xs text-primary underline underline-offset-2"
                      onClick={() => onRevertField(c.field)}
                    >
                      revert
                    </button>
                  </td>
                )}
              </tr>
            ))}
            {diff.changes.length === 0 && (
              <tr>
                <td colSpan={editable ? 5 : 4} className="px-3 py-4 text-center text-xs text-muted-foreground">
                  No field-level changes.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
