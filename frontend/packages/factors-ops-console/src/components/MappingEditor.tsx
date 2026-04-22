import { useState } from "react";
import { Check, X, Hand, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import type { MappingRow } from "@/types/ops";

/**
 * Mapping workbench editor: for each raw activity-text row, shows the
 * suggestion-agent picks, and lets the user accept / reject / manual-pick.
 *
 * The SuggestionAgent is intentionally inline-rendered here (a second dropdown
 * panel would be over-engineered for MVP) — accept/reject write into
 * component-local state via onChange.
 */
interface Props {
  rows: MappingRow[];
  onAccept: (index: number, factorId: string) => void;
  onReject: (index: number) => void;
  onManualPick: (index: number, factorId: string) => void;
}

export function MappingEditor({ rows, onAccept, onReject, onManualPick }: Props) {
  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead className="bg-muted/50 text-left">
          <tr>
            <th className="px-3 py-2 w-1/3">Raw activity text</th>
            <th className="px-3 py-2">Suggested factor</th>
            <th className="px-3 py-2 w-40 text-right">Action</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <MappingEditorRow
              key={row.index}
              row={row}
              onAccept={onAccept}
              onReject={onReject}
              onManualPick={onManualPick}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MappingEditorRow({
  row,
  onAccept,
  onReject,
  onManualPick,
}: {
  row: MappingRow;
  onAccept: (index: number, factorId: string) => void;
  onReject: (index: number) => void;
  onManualPick: (index: number, factorId: string) => void;
}) {
  const [manualMode, setManualMode] = useState(false);
  const [manualValue, setManualValue] = useState("");
  const top = row.suggested[0];

  return (
    <tr className="border-t border-border align-top">
      <td className="px-3 py-2 font-mono text-xs">{row.raw_text}</td>
      <td className="px-3 py-2">
        {manualMode ? (
          <div className="flex items-center gap-1">
            <Search className="h-3.5 w-3.5 text-muted-foreground" />
            <Input
              value={manualValue}
              onChange={(e) => setManualValue(e.target.value)}
              placeholder="FACTOR-ID..."
              className="h-7 font-mono text-xs"
            />
            <Button
              size="sm"
              variant="secondary"
              onClick={() => {
                if (manualValue.trim()) onManualPick(row.index, manualValue.trim());
                setManualMode(false);
              }}
            >
              confirm mapping
            </Button>
          </div>
        ) : top ? (
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs">{top.factor_id}</span>
              <Badge variant="muted">conf {(top.confidence * 100).toFixed(0)}%</Badge>
            </div>
            <span className="text-xs text-muted-foreground">{top.reason}</span>
          </div>
        ) : (
          <span className="text-xs text-muted-foreground">no suggestion</span>
        )}
      </td>
      <td className="px-3 py-2">
        <div className="flex items-center justify-end gap-1">
          {row.state === "accepted" ? (
            <Badge variant="success">accepted</Badge>
          ) : row.state === "rejected" ? (
            <Badge variant="muted">rejected</Badge>
          ) : (
            <>
              <Button
                size="sm"
                variant="default"
                disabled={!top}
                onClick={() => top && onAccept(row.index, top.factor_id)}
              >
                <Check className="h-3.5 w-3.5" /> accept
              </Button>
              <Button size="sm" variant="outline" onClick={() => onReject(row.index)}>
                <X className="h-3.5 w-3.5" /> reject
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setManualMode((v) => !v)}
                title="Manual pick"
              >
                <Hand className="h-3.5 w-3.5" />
              </Button>
            </>
          )}
        </div>
      </td>
    </tr>
  );
}
