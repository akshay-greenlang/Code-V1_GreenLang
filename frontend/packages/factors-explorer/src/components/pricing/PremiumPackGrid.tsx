import { useState } from "react";
import {
  Building2,
  CircleDollarSign,
  Factory,
  Leaf,
  Plus,
  ShieldCheck,
  Truck,
  Zap,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { PremiumPackSku } from "@/lib/billing";

/**
 * One Premium Data Pack as it appears on the Pricing Page.
 *
 * The 7 entries below mirror the SKUs in
 * ``greenlang.factors.billing.skus.PremiumPack`` (we surface 7 to keep
 * the grid readable; LAND_REMOVALS rolls into the Agrifood card so the
 * public layout stays at 7 tiles).
 */
export interface PremiumPackSpec {
  sku: PremiumPackSku;
  name: string;
  useCase: string;
  sampleFactors: string[];
  /** Lucide icon component, sized 5x5 in the card head. */
  Icon: React.ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
  /** "from $X / mo on Pro" — display only. */
  fromMonthlyUsd: number;
  /** Rights chip — "internal-only" or "redistributable" or "license-required". */
  rightsLabel: string;
  rightsTone: "neutral" | "warn";
}

export const PREMIUM_PACKS: PremiumPackSpec[] = [
  {
    sku: "electricity_premium",
    name: "Electricity Premium",
    useCase:
      "High-fidelity grid + market-based + half-hourly residual mixes",
    sampleFactors: [
      "ENTSO-E hourly grid mix (EU)",
      "EPA eGRID 2024 sub-region",
      "AIB Residual Mix (EU)",
    ],
    Icon: Zap,
    fromMonthlyUsd: 99,
    rightsLabel: "Internal-only",
    rightsTone: "neutral",
  },
  {
    sku: "freight_premium",
    name: "Freight Premium",
    useCase: "ISO 14083 land/air/sea, port-to-port + last-mile cascades",
    sampleFactors: [
      "GLEC v3 road TTW + WTW",
      "ICCT ocean container intensities",
      "EEA air mode emission factors",
    ],
    Icon: Truck,
    fromMonthlyUsd: 199,
    rightsLabel: "Internal-only",
    rightsTone: "neutral",
  },
  {
    sku: "product_carbon_premium",
    name: "Product Carbon / LCI Premium",
    useCase:
      "Cradle-to-gate product LCAs against ecoinvent + Sphera reference flows",
    sampleFactors: [
      "ecoinvent 3.10 cut-off (resolution overlay)",
      "Sphera Industrial Data 2024",
      "GaBi extension databases",
    ],
    Icon: Factory,
    fromMonthlyUsd: 499,
    rightsLabel: "License-required (BYOL)",
    rightsTone: "warn",
  },
  {
    sku: "epd_premium",
    name: "Construction EPD Premium",
    useCase:
      "Region-specific EPDs for steel, cement, glass, insulation (EN 15804)",
    sampleFactors: [
      "EC3 EPD ingest (~50k EPDs)",
      "ICE database v3.0 building materials",
      "EU PEF construction subset",
    ],
    Icon: Building2,
    fromMonthlyUsd: 199,
    rightsLabel: "Internal-only",
    rightsTone: "neutral",
  },
  {
    sku: "agrifood_premium",
    name: "Agrifood & Land Premium",
    useCase:
      "Agribalyse, FAO/INRAE livestock, GFW land-use change, removals",
    sampleFactors: [
      "Agribalyse 3.1 cradle-to-farm-gate",
      "FAOSTAT enteric CH4 by livestock category",
      "Global Forest Watch dLUC overlays",
    ],
    Icon: Leaf,
    fromMonthlyUsd: 199,
    rightsLabel: "Internal-only",
    rightsTone: "neutral",
  },
  {
    sku: "finance_premium",
    name: "Finance Proxy Premium (PCAF)",
    useCase:
      "EEIO + sector intensity proxies for Scope 3 Cat 15 financed emissions",
    sampleFactors: [
      "PCAF asset-class data quality scores 1–5",
      "Exiobase 3.8 EEIO intensities",
      "GHG Protocol sector intensity priors",
    ],
    Icon: CircleDollarSign,
    fromMonthlyUsd: 299,
    rightsLabel: "License-required (PCAF)",
    rightsTone: "warn",
  },
  {
    sku: "cbam_premium",
    name: "CBAM / EU Policy Premium",
    useCase:
      "EU CBAM default values, embedded emissions per CN8, country-of-origin overrides",
    sampleFactors: [
      "EU CBAM Annex IV default values",
      "JRC Tier-2 country defaults",
      "CN8 to direct/indirect mapping",
    ],
    Icon: ShieldCheck,
    fromMonthlyUsd: 299,
    rightsLabel: "Internal-only",
    rightsTone: "neutral",
  },
];

interface PremiumPackGridProps {
  /**
   * Currently-selected packs (managed by the parent so the grid
   * can be a controlled component). When the user clicks "Add to plan"
   * the parent toggles the SKU in or out of the set.
   */
  selected: Set<PremiumPackSku>;
  onToggle: (sku: PremiumPackSku) => void;
}

/**
 * Responsive grid of Premium Pack cards. Mobile-first: 1 col on
 * phones, 2 cols at md, 3 at lg, 4 at xl so the seventh card doesn't
 * orphan on the bottom row at common breakpoints.
 */
export function PremiumPackGrid({ selected, onToggle }: PremiumPackGridProps) {
  return (
    <div
      className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
      data-testid="premium-pack-grid"
    >
      {PREMIUM_PACKS.map((pack) => (
        <PremiumPackCard
          key={pack.sku}
          pack={pack}
          isSelected={selected.has(pack.sku)}
          onToggle={() => onToggle(pack.sku)}
        />
      ))}
    </div>
  );
}

function PremiumPackCard({
  pack,
  isSelected,
  onToggle,
}: {
  pack: PremiumPackSpec;
  isSelected: boolean;
  onToggle: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const visibleSamples = expanded
    ? pack.sampleFactors
    : pack.sampleFactors.slice(0, 2);
  const Icon = pack.Icon;
  return (
    <Card
      className={cn(
        "flex flex-col transition-shadow",
        isSelected && "ring-2 ring-primary shadow-md"
      )}
      data-testid={`premium-pack-${pack.sku}`}
    >
      <CardHeader className="space-y-2 pb-3">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "inline-flex h-8 w-8 items-center justify-center rounded-md",
                "bg-primary/10 text-primary"
              )}
              aria-hidden="true"
            >
              <Icon className="h-4 w-4" aria-hidden={true} />
            </span>
            <CardTitle className="text-base leading-tight">
              {pack.name}
            </CardTitle>
          </div>
        </div>
        <CardDescription className="text-xs">{pack.useCase}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-3 pb-4 pt-0">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold tabular-nums">
            from ${pack.fromMonthlyUsd}/mo
          </span>
          <Badge
            variant={pack.rightsTone === "warn" ? "warning" : "secondary"}
            className="text-[10px]"
          >
            {pack.rightsLabel}
          </Badge>
        </div>

        <div className="space-y-1">
          <p className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
            Sample factors
          </p>
          <ul className="space-y-0.5 text-xs text-muted-foreground">
            {visibleSamples.map((s) => (
              <li key={s} className="font-mono">
                {s}
              </li>
            ))}
          </ul>
          {pack.sampleFactors.length > 2 ? (
            <button
              type="button"
              onClick={() => setExpanded((x) => !x)}
              className="text-[10px] font-medium text-primary hover:underline"
              aria-expanded={expanded}
            >
              {expanded
                ? "Show fewer"
                : `Show ${pack.sampleFactors.length - 2} more`}
            </button>
          ) : null}
        </div>

        <Button
          variant={isSelected ? "default" : "outline"}
          size="sm"
          onClick={onToggle}
          className="mt-auto w-full"
          aria-pressed={isSelected}
          data-testid={`premium-pack-cta-${pack.sku}`}
        >
          <Plus
            className={cn(
              "h-3.5 w-3.5 transition-transform",
              isSelected && "rotate-45"
            )}
          />
          {isSelected ? "Added to plan" : "Add to plan"}
        </Button>
      </CardContent>
    </Card>
  );
}

/** Convenience accessor for the test file. */
export const PREMIUM_PACK_SKUS: PremiumPackSku[] = PREMIUM_PACKS.map(
  (p) => p.sku
);
