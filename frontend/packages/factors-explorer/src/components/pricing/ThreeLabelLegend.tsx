import { CheckCircle2, Eye, Plug } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

/**
 * Inline legend explaining the three factor labels (Certified, Preview,
 * Connector-only) and how each tier is allowed to surface them.
 *
 * This complements ``ThreeLabelDashboard`` (which shows live counts) by
 * focusing on **policy**: which redistribution class each label maps to
 * and which tier can call it. Source of truth for the policy is
 * ``greenlang.factors.tier_enforcement.TierVisibility.from_tier``.
 */
export function ThreeLabelLegend() {
  return (
    <div
      className="grid grid-cols-1 gap-3 md:grid-cols-3"
      data-testid="three-label-legend"
    >
      <LegendCard
        label="Certified"
        Icon={CheckCircle2}
        accent="border-l-factor-certified-500"
        iconClass="text-factor-certified-700"
        description="Audit-grade. Resolved against a primary source, full provenance, signed receipts. Available on every tier."
        tiers={[
          { name: "Community", tone: "ok" },
          { name: "Pro", tone: "ok" },
          { name: "Platform", tone: "ok" },
          { name: "Enterprise", tone: "ok" },
        ]}
        testId="legend-certified"
      />
      <LegendCard
        label="Preview"
        Icon={Eye}
        accent="border-l-factor-preview-500"
        iconClass="text-factor-preview-700"
        description="Pre-publication or under-review factors. Numerically usable but the source has not yet hit Certified status."
        tiers={[
          { name: "Community", tone: "blocked" },
          { name: "Pro", tone: "ok" },
          { name: "Platform", tone: "ok" },
          { name: "Enterprise", tone: "ok" },
        ]}
        testId="legend-preview"
      />
      <LegendCard
        label="Connector-only"
        Icon={Plug}
        accent="border-l-factor-connector-500"
        iconClass="text-factor-connector-700"
        description="Resolvable in your tenant via a private connector / customer override; never returned over the public API for redistribution."
        tiers={[
          { name: "Community", tone: "blocked" },
          { name: "Pro", tone: "blocked" },
          { name: "Platform", tone: "blocked" },
          { name: "Enterprise", tone: "ok" },
        ]}
        testId="legend-connector-only"
      />
    </div>
  );
}

function LegendCard({
  label,
  Icon,
  accent,
  iconClass,
  description,
  tiers,
  testId,
}: {
  label: string;
  Icon: React.ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
  accent: string;
  iconClass: string;
  description: string;
  tiers: { name: string; tone: "ok" | "blocked" }[];
  testId: string;
}) {
  return (
    <Card className={cn("border-l-4", accent)} data-testid={testId}>
      <CardContent className="space-y-3 p-4">
        <div className="flex items-center gap-2">
          <Icon
            className={cn("h-5 w-5", iconClass)}
            aria-hidden={true}
          />
          <h3 className="font-semibold">{label}</h3>
        </div>
        <p className="text-sm text-muted-foreground">{description}</p>
        <div className="flex flex-wrap gap-1.5">
          {tiers.map((t) => (
            <span
              key={t.name}
              className={cn(
                "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium",
                t.tone === "ok"
                  ? "border-emerald-500 bg-emerald-50 text-emerald-700"
                  : "border-border bg-muted text-muted-foreground line-through"
              )}
            >
              {t.name}
            </span>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
