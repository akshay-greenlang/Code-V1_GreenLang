import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ExternalLink, GitCommit } from "lucide-react";
import { getEditions } from "@/lib/api";
import { queryKeys } from "@/lib/query";
import { Badge } from "@/components/ui/badge";
import { formatDate, truncateHash } from "@/lib/utils";

export const Route = createFileRoute("/editions")({
  component: EditionsPage,
});

function EditionsPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.editions(24),
    queryFn: () => getEditions(24),
  });

  if (error) {
    return (
      <div
        role="alert"
        className="rounded-md border border-factor-deprecated-500 bg-factor-deprecated-50 p-4 text-sm text-factor-deprecated-700"
      >
        Editions unavailable.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-2xl font-semibold">Editions</h1>
        <p className="text-sm text-muted-foreground">
          Every release is a signed, immutable manifest. Pin your calculations
          to a specific edition; deprecated factors always name their
          replacement.
        </p>
      </header>

      {isLoading ? (
        <ol className="space-y-3" aria-busy="true">
          {Array.from({ length: 5 }).map((_, i) => (
            <li
              key={i}
              className="h-20 animate-pulse rounded-lg border border-border bg-muted/40"
            />
          ))}
        </ol>
      ) : (
        <ol className="relative space-y-4 border-l-2 border-border pl-4">
          {(data ?? []).map((e) => (
            <li key={e.edition_id} className="relative">
              <span
                className={`absolute -left-[21px] top-1 inline-flex h-4 w-4 items-center justify-center rounded-full border-2 ${
                  e.is_current
                    ? "border-factor-certified-500 bg-factor-certified-500"
                    : "border-border bg-background"
                }`}
                aria-hidden="true"
              />
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-sm font-semibold">
                  {e.edition_id}
                </span>
                {e.is_current ? <Badge variant="success">CURRENT</Badge> : null}
                <span className="text-xs text-muted-foreground">
                  released {formatDate(e.released_at)}
                </span>
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                <span className="inline-flex items-center gap-1 font-mono">
                  <GitCommit className="h-3 w-3" />
                  {truncateHash(e.manifest_fingerprint, 12, 6)}
                </span>
                {e.release_notes_url ? (
                  <a
                    href={e.release_notes_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 underline underline-offset-2 hover:text-foreground"
                  >
                    Release notes <ExternalLink className="h-3 w-3" />
                  </a>
                ) : null}
              </div>
              {e.summary ? (
                <p className="mt-1 text-sm">{e.summary}</p>
              ) : null}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
