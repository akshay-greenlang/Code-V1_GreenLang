import { AlertTriangle, Check, CornerDownRight, Info } from "lucide-react";
import type {
  AlternateCandidate,
  FallbackRank,
  ResolvedFactorExplain,
  StepLabel,
} from "@/types/factors";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface ExplainTraceProps {
  explain: ResolvedFactorExplain;
}

const STEP_ORDER: Array<{ rank: FallbackRank; label: StepLabel; title: string }> = [
  { rank: 1, label: "customer_override", title: "Customer override" },
  { rank: 2, label: "supplier_specific", title: "Supplier-specific" },
  { rank: 3, label: "facility_specific", title: "Facility-specific" },
  { rank: 4, label: "region_specific", title: "Region-specific" },
  {
    rank: 5,
    label: "country_or_sector_average",
    title: "Country / sector average",
  },
  { rank: 6, label: "global_average", title: "Global average" },
  { rank: 7, label: "default_assumption", title: "Default assumption" },
];

/**
 * Consumes ResolvedFactor.explain() payload:
 *  - 7-step cascade (step_label + fallback_rank 1..7 visualized as a progression)
 *  - why_chosen sentence
 *  - list of alternates with "why not chosen" reasons
 *  - unit conversion trace if present
 *  - deprecation banner if deprecation_status present
 */
export function ExplainTrace({ explain }: ExplainTraceProps) {
  const chosenRank = explain.derivation.fallback_rank;
  const isDeprecated =
    explain.derivation.deprecation_status === "deprecated" ||
    explain.derivation.deprecation_status === "superseded";

  return (
    <Card data-testid="explain-trace">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <CornerDownRight className="h-4 w-4" />
          Explain trace
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {isDeprecated ? (
          <div
            role="alert"
            data-testid="deprecation-banner"
            className="flex items-start gap-2 rounded-md border border-factor-deprecated-500 bg-factor-deprecated-50 p-3 text-sm text-factor-deprecated-700"
          >
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <div>
              <p className="font-semibold">
                This factor is {explain.derivation.deprecation_status}.
              </p>
              {explain.derivation.deprecation_replacement ? (
                <p>
                  Replacement:{" "}
                  <code className="font-mono">
                    {explain.derivation.deprecation_replacement}
                  </code>
                </p>
              ) : null}
            </div>
          </div>
        ) : null}

        <ol
          className="space-y-1"
          data-testid="cascade-steps"
          aria-label="7-step resolution cascade"
        >
          {STEP_ORDER.map((step) => {
            const status =
              step.rank === chosenRank
                ? "chosen"
                : step.rank < chosenRank
                  ? "skipped"
                  : "considered";
            return (
              <li
                key={step.rank}
                data-rank={step.rank}
                data-status={status}
                className={cn(
                  "flex items-center justify-between gap-3 rounded-md border px-3 py-1.5 text-sm",
                  status === "chosen"
                    ? "border-factor-certified-500 bg-factor-certified-50 text-factor-certified-700"
                    : status === "skipped"
                      ? "border-border bg-muted text-muted-foreground"
                      : "border-border text-foreground"
                )}
              >
                <div className="flex items-center gap-2">
                  <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-white text-xs font-semibold ring-1 ring-inset ring-border tabular-nums">
                    {step.rank}
                  </span>
                  <span className="font-medium">{step.title}</span>
                  <span className="font-mono text-xs text-muted-foreground">
                    {step.label}
                  </span>
                </div>
                {status === "chosen" ? (
                  <Badge variant="success" className="gap-1">
                    <Check className="h-3 w-3" />
                    CHOSEN
                  </Badge>
                ) : status === "skipped" ? (
                  <span className="text-xs uppercase tracking-wide">
                    skipped
                  </span>
                ) : (
                  <span className="text-xs uppercase tracking-wide text-muted-foreground">
                    considered
                  </span>
                )}
              </li>
            );
          })}
        </ol>

        <div className="rounded-md border border-border bg-muted/30 p-3">
          <p className="text-sm">
            <span className="font-semibold">Why chosen: </span>
            <span>{explain.derivation.why_chosen}</span>
          </p>
          {explain.derivation.assumptions.length > 0 ? (
            <ul className="mt-2 list-inside list-disc text-xs text-muted-foreground">
              {explain.derivation.assumptions.map((a, i) => (
                <li key={i}>{a}</li>
              ))}
            </ul>
          ) : null}
        </div>

        {explain.unit_conversion ? (
          <div className="rounded-md border border-border p-3 text-sm">
            <p className="font-semibold">Unit conversion</p>
            <p className="text-xs text-muted-foreground">
              → {explain.unit_conversion.target_unit}
              {explain.unit_conversion.factor !== null &&
              explain.unit_conversion.factor !== undefined
                ? ` via factor ${explain.unit_conversion.factor}`
                : ""}
            </p>
            {explain.unit_conversion.path.length > 0 ? (
              <p className="mt-1 font-mono text-xs">
                {explain.unit_conversion.path.join(" → ")}
              </p>
            ) : null}
            {explain.unit_conversion.note ? (
              <p className="mt-1 text-xs italic text-muted-foreground">
                {explain.unit_conversion.note}
              </p>
            ) : null}
          </div>
        ) : null}

        {explain.alternates.length > 0 ? (
          <AlternatesList alternates={explain.alternates} />
        ) : (
          <p className="text-xs text-muted-foreground">
            <Info className="mr-1 inline h-3 w-3" />
            No other candidates were considered.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function AlternatesList({
  alternates,
}: {
  alternates: AlternateCandidate[];
}) {
  return (
    <div data-testid="alternates-list">
      <p className="mb-1.5 text-sm font-semibold">
        Alternates ({alternates.length})
      </p>
      <ul className="space-y-1">
        {alternates.map((a) => (
          <li
            key={a.factor_id}
            className="flex items-start justify-between gap-3 rounded border border-border px-3 py-1.5 text-sm"
          >
            <div className="min-w-0">
              <code className="font-mono text-xs">{a.factor_id}</code>
              <p className="text-xs text-muted-foreground">{a.why_not_chosen}</p>
            </div>
            <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
              tie {a.tie_break_score.toFixed(2)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
