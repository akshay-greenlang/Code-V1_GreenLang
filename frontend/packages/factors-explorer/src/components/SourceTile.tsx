import { Link } from "@tanstack/react-router";
import { Calendar, Database } from "lucide-react";
import type { Source } from "@/types/factors";
import { Card, CardContent } from "@/components/ui/card";
import { LicenseBadge } from "@/components/LicenseBadge";
import { formatDate } from "@/lib/utils";

interface SourceTileProps {
  source: Source;
}

/** Publisher, jurisdiction, license class, last ingested, cadence. */
export function SourceTile({ source }: SourceTileProps) {
  return (
    <Card className="transition-shadow hover:shadow-md">
      <Link
        to="/sources/$sourceId"
        params={{ sourceId: source.source_id }}
        className="block focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        aria-label={`Open source ${source.name}`}
      >
        <CardContent className="space-y-2 p-4">
          <div className="flex items-center gap-2">
            <Database
              className="h-4 w-4 text-muted-foreground"
              aria-hidden="true"
            />
            <h3 className="truncate font-semibold">{source.name}</h3>
          </div>
          <p className="text-xs text-muted-foreground">
            {source.publisher}
            {source.jurisdiction ? ` • ${source.jurisdiction}` : ""}
          </p>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <LicenseBadge licenseClass={source.license_class} compact />
            <span className="rounded-full bg-muted px-2 py-0.5 font-mono">
              v{source.current_version}
            </span>
            <span className="rounded-full bg-muted px-2 py-0.5">
              {source.cadence}
            </span>
          </div>
          <div className="flex items-center justify-between pt-1 text-xs text-muted-foreground">
            <span className="tabular-nums">
              {source.factor_count.toLocaleString()} factors
            </span>
            <span className="inline-flex items-center gap-1">
              <Calendar className="h-3 w-3" />
              {formatDate(source.last_updated)}
            </span>
          </div>
        </CardContent>
      </Link>
    </Card>
  );
}
