import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { ChevronDown, GitCommit } from "lucide-react";
import { getEditions, getEditionCurrent } from "@/lib/api";
import { queryKeys } from "@/lib/query";
import { truncateHash } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface EditionPinProps {
  value?: string; // currently pinned edition id (from URL ?edition=...)
  onChange: (editionId: string | null) => void;
  clickable?: boolean;
}

/**
 * Shows the current edition pin; click to open dropdown of editions;
 * persists via the `onChange` callback (the route layer writes it to the URL
 * ?edition= param so it's sharable + SEO-canonical-stable).
 */
export function EditionPin({
  value,
  onChange,
  clickable = true,
}: EditionPinProps) {
  const [open, setOpen] = useState(false);

  const { data: current } = useQuery({
    queryKey: queryKeys.editionCurrent(),
    queryFn: getEditionCurrent,
  });

  const { data: editions } = useQuery({
    queryKey: queryKeys.editions(12),
    queryFn: () => getEditions(12),
    enabled: open,
  });

  const pinnedId = value ?? current?.edition_id;
  const pinned = editions?.find((e) => e.edition_id === pinnedId) ?? current;

  if (!pinned) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-muted px-2.5 py-1 text-xs text-muted-foreground">
        <GitCommit className="h-3 w-3" />
        loading edition…
      </span>
    );
  }

  return (
    <div className="relative inline-block">
      <Button
        variant="outline"
        size="sm"
        disabled={!clickable}
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <GitCommit className="h-3.5 w-3.5" />
        <span className="font-medium">Edition {pinned.edition_id}</span>
        <span className="text-xs text-muted-foreground">
          {truncateHash(pinned.manifest_fingerprint, 6, 4)}
        </span>
        {clickable ? <ChevronDown className="h-3 w-3 opacity-50" /> : null}
      </Button>

      {open ? (
        <ul
          role="listbox"
          className="absolute right-0 z-20 mt-1 max-h-80 w-80 overflow-y-auto rounded-md border border-border bg-card shadow-lg"
        >
          {(editions ?? []).map((e) => (
            <li key={e.edition_id}>
              <button
                role="option"
                aria-selected={e.edition_id === pinnedId}
                className="flex w-full flex-col items-start gap-0.5 border-b border-border px-3 py-2 text-left text-sm hover:bg-accent"
                onClick={() => {
                  onChange(e.is_current ? null : e.edition_id);
                  setOpen(false);
                }}
              >
                <span className="font-medium">
                  {e.edition_id}
                  {e.is_current ? (
                    <span className="ml-2 rounded bg-factor-certified-100 px-1 py-px text-[10px] text-factor-certified-700">
                      CURRENT
                    </span>
                  ) : null}
                </span>
                <span className="text-xs text-muted-foreground">
                  released {e.released_at.slice(0, 10)} •{" "}
                  {truncateHash(e.manifest_fingerprint, 6, 4)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
