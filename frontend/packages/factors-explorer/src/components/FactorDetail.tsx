import { Link } from "@tanstack/react-router";
import { ExternalLink, Tag } from "lucide-react";
import type { FactorDetailPayload } from "@/types/factors";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { LicenseBadge } from "@/components/LicenseBadge";
import { QualityMeter } from "@/components/QualityMeter";
import { ExplainTrace } from "@/components/ExplainTrace";
import { SignedReceipt } from "@/components/SignedReceipt";
import { EditionPin } from "@/components/EditionPin";
import { formatDate } from "@/lib/utils";

interface FactorDetailProps {
  payload: FactorDetailPayload;
  editionPin?: string;
  onEditionChange: (editionId: string | null) => void;
}

/**
 * Full factor record view:
 *  - top: chosen_factor_id + method_profile + edition
 *  - emissions: gas breakdown
 *  - quality: FQS + 5 components
 *  - provenance: source + version + citation + license
 *  - explain: <ExplainTrace/> (always visible — non-negotiable #2)
 *  - signed receipt with Copy button
 */
export function FactorDetail({
  payload,
  editionPin,
  onEditionChange,
}: FactorDetailProps) {
  const { factor, explain, edition_id, signed_receipt } = payload;

  return (
    <div className="space-y-6">
      {/* Top block */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="font-mono text-xs text-muted-foreground">
                {factor.factor_id}
              </p>
              <CardTitle className="mt-1">
                {factor.fuel_type || factor.factor_id}
                {factor.jurisdiction ? (
                  <span className="text-muted-foreground">
                    {" "}
                    — {factor.jurisdiction}
                  </span>
                ) : null}
              </CardTitle>
              <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                <Badge variant="outline">scope {factor.scope}</Badge>
                <Badge variant="outline">{factor.factor_family}</Badge>
                <Badge variant="outline" className="font-mono">
                  {explain.chosen.method_profile}
                </Badge>
              </div>
            </div>
            <EditionPin value={editionPin} onChange={onEditionChange} />
          </div>
        </CardHeader>
        <CardContent className="space-y-3 pt-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="success">{factor.factor_status}</Badge>
            <LicenseBadge
              licenseClass={factor.license_info.class}
              redistributionAllowed={factor.license_info.redistribution_allowed}
              commercialUseAllowed={factor.license_info.commercial_use_allowed}
              attributionRequired={factor.license_info.attribution_required}
            />
            <QualityMeter fqs={factor.quality} compact />
            <span className="text-xs text-muted-foreground">
              valid {formatDate(factor.valid_from)}
              {factor.valid_to ? ` → ${formatDate(factor.valid_to)}` : ""}
            </span>
            <span className="text-xs text-muted-foreground">
              edition <span className="font-mono">{edition_id}</span>
            </span>
          </div>

          {factor.sector_tags.length > 0 || factor.activity_tags.length > 0 ? (
            <div className="flex flex-wrap items-center gap-1.5 text-xs">
              <Tag className="h-3 w-3 text-muted-foreground" />
              {[...factor.sector_tags, ...factor.activity_tags].map((t) => (
                <span
                  key={t}
                  className="rounded bg-muted px-1.5 py-0.5 font-mono text-[11px]"
                >
                  {t}
                </span>
              ))}
            </div>
          ) : null}
        </CardContent>
      </Card>

      {/* Emissions + Quality grid */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Emissions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div>
              <p className="text-3xl font-semibold tabular-nums">
                {factor.co2e_per_unit.toFixed(4)}{" "}
                <span className="text-base text-muted-foreground">
                  kgCO₂e / {factor.unit}
                </span>
              </p>
              <p className="text-xs text-muted-foreground">
                GWP basis: {factor.gwp_100yr.gwp_basis}
              </p>
            </div>
            <ul className="mt-3 space-y-1 text-sm">
              <GasRow label="CO₂" value={explain.emissions.co2_kg} unit="kg" />
              <GasRow label="CH₄" value={explain.emissions.ch4_kg} unit="kg" />
              <GasRow label="N₂O" value={explain.emissions.n2o_kg} unit="kg" />
              {explain.emissions.hfcs_kg ? (
                <GasRow
                  label="HFCs"
                  value={explain.emissions.hfcs_kg}
                  unit="kg"
                />
              ) : null}
              {explain.emissions.biogenic_co2_kg ? (
                <GasRow
                  label="Biogenic CO₂"
                  value={explain.emissions.biogenic_co2_kg}
                  unit="kg"
                />
              ) : null}
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Quality</CardTitle>
          </CardHeader>
          <CardContent>
            <QualityMeter fqs={factor.quality} />
          </CardContent>
        </Card>
      </div>

      {/* Provenance */}
      <Card>
        <CardHeader>
          <CardTitle>Provenance</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-2 text-sm md:grid-cols-2">
          <div>
            <p className="font-medium">{factor.provenance.source_org}</p>
            <p className="text-muted-foreground">
              {factor.provenance.source_publication} (
              {factor.provenance.source_year}) v{factor.provenance.source_version}
            </p>
            {factor.source_id ? (
              <Link
                to="/sources/$sourceId"
                params={{ sourceId: factor.source_id }}
                className="mt-1 inline-flex items-center gap-1 text-xs text-primary underline-offset-2 hover:underline"
              >
                View source record
                <ExternalLink className="h-3 w-3" />
              </Link>
            ) : null}
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Methodology</p>
            <p className="font-mono text-xs">{factor.provenance.methodology}</p>
            {factor.provenance.citation ? (
              <p className="mt-1 text-xs italic text-muted-foreground">
                {factor.provenance.citation}
              </p>
            ) : null}
          </div>
        </CardContent>
      </Card>

      {/* Explain trace — always visible per non-negotiable #2 */}
      <ExplainTrace explain={explain} />

      {/* Uncertainty */}
      {explain.uncertainty.distribution !== "unknown" ||
      explain.uncertainty.ci_95_percent ? (
        <Card>
          <CardHeader>
            <CardTitle>Uncertainty</CardTitle>
          </CardHeader>
          <CardContent className="text-sm">
            <p>
              Distribution:{" "}
              <span className="font-mono">{explain.uncertainty.distribution}</span>
              {explain.uncertainty.ci_95_percent !== null &&
              explain.uncertainty.ci_95_percent !== undefined ? (
                <>
                  {" "}
                  • 95% CI ±
                  {(explain.uncertainty.ci_95_percent * 100).toFixed(1)}%
                </>
              ) : null}
            </p>
            {explain.uncertainty.low !== null &&
            explain.uncertainty.low !== undefined &&
            explain.uncertainty.high !== null &&
            explain.uncertainty.high !== undefined ? (
              <p className="text-muted-foreground">
                range {explain.uncertainty.low} – {explain.uncertainty.high}
              </p>
            ) : null}
            {explain.uncertainty.note ? (
              <p className="mt-1 text-xs italic text-muted-foreground">
                {explain.uncertainty.note}
              </p>
            ) : null}
          </CardContent>
        </Card>
      ) : null}

      {/* Signed receipt */}
      <SignedReceipt receipt={signed_receipt} />

      {/* Alternates CTA */}
      <div className="flex justify-end">
        <Button asChild variant="outline" size="sm">
          <Link
            to="/factors/$factorId"
            params={{ factorId: factor.factor_id }}
            search={(prev) => ({ ...prev, alternates: 20 })}
          >
            Show all alternates
          </Link>
        </Button>
      </div>
    </div>
  );
}

function GasRow({
  label,
  value,
  unit,
}: {
  label: string;
  value: number;
  unit: string;
}) {
  return (
    <li className="flex items-center justify-between border-b border-border pb-1 last:border-0 last:pb-0">
      <span className="text-muted-foreground">{label}</span>
      <span className="tabular-nums">
        {value.toFixed(4)} {unit}
      </span>
    </li>
  );
}
