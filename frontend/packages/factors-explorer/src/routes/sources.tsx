import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { SourceTile } from "@/components/SourceTile";
import { getSources } from "@/lib/api";
import { queryKeys } from "@/lib/query";

export const Route = createFileRoute("/sources")({
  component: SourcesPage,
});

function SourcesPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.sources(),
    queryFn: getSources,
  });

  return (
    <div className="space-y-4">
      <header>
        <h1 className="text-2xl font-semibold">Sources</h1>
        <p className="text-sm text-muted-foreground">
          The authoritative publishers and datasets the factor catalogue is
          built from.
        </p>
      </header>

      {isLoading ? (
        <SkeletonGrid />
      ) : error ? (
        <div
          role="alert"
          className="rounded-md border border-factor-deprecated-500 bg-factor-deprecated-50 p-4 text-sm text-factor-deprecated-700"
        >
          Failed to load sources.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {(data ?? []).map((s) => (
            <SourceTile key={s.source_id} source={s} />
          ))}
        </div>
      )}
    </div>
  );
}

function SkeletonGrid() {
  return (
    <div
      className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
      aria-busy="true"
    >
      {Array.from({ length: 8 }).map((_, i) => (
        <div
          key={i}
          className="h-36 animate-pulse rounded-lg border border-border bg-muted/40"
        />
      ))}
    </div>
  );
}
