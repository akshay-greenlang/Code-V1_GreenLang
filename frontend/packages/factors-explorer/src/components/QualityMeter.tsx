import type { CompositeFqs } from "@/types/factors";
import { cn, fqsBand } from "@/lib/utils";

interface QualityMeterProps {
  fqs: CompositeFqs;
  compact?: boolean;
}

/**
 * Factor-Quality-Score meter. 0-100 arc gauge plus 5 component bars
 * (temporal, geographic, technology, verification, completeness) with
 * CTO-alias tooltips. Color-banded by FQS rating.
 */
export function QualityMeter({ fqs, compact = false }: QualityMeterProps) {
  const band = fqsBand(fqs.overall);
  const arcColor = bandColor(band);

  if (compact) {
    return (
      <div
        className="inline-flex items-center gap-2"
        aria-label={`FQS ${fqs.rating} (${fqs.overall.toFixed(0)})`}
      >
        <RatingPill rating={fqs.rating} band={band} />
        <span className="text-xs text-muted-foreground tabular-nums">
          {fqs.overall.toFixed(0)}/100
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-3" data-testid="quality-meter">
      <div className="flex items-center gap-4">
        <ArcGauge value={fqs.overall} color={arcColor} />
        <div className="space-y-1">
          <RatingPill rating={fqs.rating} band={band} />
          <p className="text-xs text-muted-foreground">
            Factor Quality Score (composite)
          </p>
          {fqs.uncertainty_95ci !== null && fqs.uncertainty_95ci !== undefined ? (
            <p className="text-xs text-muted-foreground">
              ±{(fqs.uncertainty_95ci * 100).toFixed(1)}% 95% CI
            </p>
          ) : null}
        </div>
      </div>

      <ul className="grid grid-cols-1 gap-1.5">
        <ComponentBar
          label="Temporal"
          alias="TiR · time representativeness"
          value={fqs.temporal_representativeness}
        />
        <ComponentBar
          label="Geographic"
          alias="GR · geography representativeness"
          value={fqs.geographic_representativeness}
        />
        <ComponentBar
          label="Technology"
          alias="TeR · technology representativeness"
          value={fqs.technology_representativeness}
        />
        <ComponentBar
          label="Verification"
          alias="Ver · third-party verification"
          value={fqs.verification}
        />
        <ComponentBar
          label="Completeness"
          alias="C · data completeness"
          value={fqs.completeness}
        />
      </ul>
    </div>
  );
}

function RatingPill({
  rating,
  band,
}: {
  rating: "A" | "B" | "C" | "D" | "E";
  band: "excellent" | "good" | "fair" | "poor";
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-sm font-semibold",
        bandPalette(band)
      )}
      data-band={band}
    >
      FQS {rating}
    </span>
  );
}

function ComponentBar({
  label,
  alias,
  value,
}: {
  label: string;
  alias: string;
  value: number;
}) {
  const band = fqsBand(value);
  return (
    <li className="grid grid-cols-[110px_1fr_40px] items-center gap-2 text-xs">
      <span className="truncate text-muted-foreground" title={alias}>
        {label}
      </span>
      <div
        className="h-2 w-full overflow-hidden rounded-full bg-slate-100"
        role="progressbar"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`${label} score`}
      >
        <div
          className={cn("h-full rounded-full", bandBarColor(band))}
          style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
        />
      </div>
      <span className="text-right tabular-nums text-muted-foreground">
        {value.toFixed(0)}
      </span>
    </li>
  );
}

function ArcGauge({ value, color }: { value: number; color: string }) {
  // simple semi-circle arc using svg
  const pct = Math.max(0, Math.min(100, value));
  const angle = (pct / 100) * 180;
  const r = 32;
  const cx = 40;
  const cy = 40;
  const radians = ((180 - angle) * Math.PI) / 180;
  const x = cx + r * Math.cos(radians);
  const y = cy - r * Math.sin(radians);
  const largeArcFlag = angle > 180 ? 1 : 0;
  const d = `M ${cx - r} ${cy} A ${r} ${r} 0 ${largeArcFlag} 1 ${x} ${y}`;
  return (
    <svg width="80" height="48" viewBox="0 0 80 48" aria-hidden="true">
      <path
        d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        stroke="#e2e8f0"
        strokeWidth={6}
        fill="none"
        strokeLinecap="round"
      />
      <path
        d={d}
        stroke={color}
        strokeWidth={6}
        fill="none"
        strokeLinecap="round"
      />
      <text
        x={cx}
        y={cy - 4}
        textAnchor="middle"
        className="fill-foreground"
        style={{ font: "bold 14px ui-sans-serif" }}
      >
        {pct.toFixed(0)}
      </text>
    </svg>
  );
}

function bandColor(band: "excellent" | "good" | "fair" | "poor"): string {
  return {
    excellent: "#059669",
    good: "#84cc16",
    fair: "#f59e0b",
    poor: "#e11d48",
  }[band];
}

function bandPalette(band: "excellent" | "good" | "fair" | "poor"): string {
  switch (band) {
    case "excellent":
      return "bg-emerald-50 text-emerald-700 border-emerald-500";
    case "good":
      return "bg-lime-50 text-lime-700 border-lime-500";
    case "fair":
      return "bg-amber-50 text-amber-700 border-amber-500";
    case "poor":
      return "bg-rose-50 text-rose-700 border-rose-500";
  }
}

function bandBarColor(band: "excellent" | "good" | "fair" | "poor"): string {
  switch (band) {
    case "excellent":
      return "bg-emerald-500";
    case "good":
      return "bg-lime-500";
    case "fair":
      return "bg-amber-500";
    case "poor":
      return "bg-rose-500";
  }
}
