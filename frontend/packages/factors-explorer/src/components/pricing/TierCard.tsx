import { useState } from "react";
import {
  ArrowRight,
  Check,
  Loader2,
  Lock,
  Sparkles,
  Zap,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  startCheckout,
  type CheckoutSkuName,
  type PremiumPackSku,
} from "@/lib/billing";

/**
 * The four tiers we surface on the public pricing page.
 *
 * String values intentionally line up with
 * ``greenlang.commercial.tiers.Tier`` and the SKU catalog in
 * ``greenlang.factors.billing.skus`` so a backend webhook handler can
 * round-trip the analytics event back to the right ``TierConfig``.
 */
export type TierKey = "community" | "pro" | "platform" | "enterprise";

export type TierBilling =
  | { kind: "free" }
  | {
      kind: "usage";
      monthlyPriceUsd: number;
      annualPriceUsd: number;
      headlineUnit: string; // "/mo, billed monthly"
    }
  | {
      kind: "annual";
      annualPriceUsdFloor: number;
      headlineUnit: string; // "/yr, annual contract"
    }
  | { kind: "contact" };

export type CtaSpec =
  | {
      kind: "external";
      href: string;
      label: string;
    }
  | {
      kind: "checkout";
      skuName: CheckoutSkuName;
      label: string;
    }
  | {
      kind: "modal";
      label: string;
      onClick: () => void;
    };

export interface TierCardProps {
  tier: TierKey;
  name: string;
  tagline: string;
  targetBuyer: string;
  billing: TierBilling;
  /** Bullet list of capabilities & rate-limit highlights. */
  features: string[];
  /** Premium packs included free at this tier (display-only). */
  includedPacks: string[];
  /** API rate limits to surface (e.g. "100k API calls/mo"). */
  rateLimits: string[];
  cta: CtaSpec;
  /** Premium packs the user has tagged as add-ons (passed to checkout). */
  addOnPacks?: PremiumPackSku[];
  /** Highlight (recommended) styling. */
  highlighted?: boolean;
  /** Override fetch (only used by tests). */
  fetchImpl?: typeof fetch;
  /** Override redirect (only used by tests). */
  onRedirect?: (url: string) => void;
}

/**
 * One pricing tier presented as a vertically-stacked Card.
 *
 * The visual hierarchy follows the existing factors-explorer style
 * (border-l accent, muted body text, primary CTA button) so the page
 * sits cleanly next to ``index.tsx`` and ``editions.tsx``.
 */
export function TierCard(props: TierCardProps) {
  const {
    tier,
    name,
    tagline,
    targetBuyer,
    billing,
    features,
    includedPacks,
    rateLimits,
    cta,
    addOnPacks,
    highlighted = false,
    fetchImpl,
    onRedirect,
  } = props;
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const accent = ACCENT[tier];

  const handleCta = async () => {
    setError(null);
    if (cta.kind === "external") {
      if (typeof window !== "undefined") {
        window.location.assign(cta.href);
      }
      return;
    }
    if (cta.kind === "modal") {
      cta.onClick();
      return;
    }
    // checkout
    setPending(true);
    try {
      await startCheckout(
        {
          skuName: cta.skuName,
          premiumPacks: addOnPacks,
        },
        { fetchImpl, redirect: onRedirect }
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Checkout failed";
      setError(msg);
    } finally {
      setPending(false);
    }
  };

  return (
    <Card
      className={cn(
        "flex flex-col border-l-4 transition-shadow",
        accent,
        highlighted &&
          "shadow-lg ring-2 ring-primary/40 scale-[1.01] md:scale-[1.02]"
      )}
      data-testid={`tier-card-${tier}`}
    >
      <CardHeader className="space-y-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xl">{name}</CardTitle>
          {highlighted ? (
            <Badge variant="default" className="gap-1">
              <Sparkles className="h-3 w-3" />
              Recommended
            </Badge>
          ) : null}
        </div>
        <CardDescription>{tagline}</CardDescription>
        <p className="text-xs uppercase tracking-wide text-muted-foreground">
          For {targetBuyer}
        </p>
      </CardHeader>

      <CardContent className="flex flex-1 flex-col gap-4">
        <BillingRow billing={billing} />

        {rateLimits.length > 0 ? (
          <div className="space-y-1">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Limits
            </p>
            <ul className="space-y-1 text-sm">
              {rateLimits.map((r) => (
                <li key={r} className="flex items-start gap-2">
                  <Zap
                    className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-primary"
                    aria-hidden="true"
                  />
                  <span>{r}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        <div className="space-y-1">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Includes
          </p>
          <ul className="space-y-1 text-sm">
            {features.map((f) => (
              <li key={f} className="flex items-start gap-2">
                <Check
                  className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-emerald-600"
                  aria-hidden="true"
                />
                <span>{f}</span>
              </li>
            ))}
          </ul>
        </div>

        {includedPacks.length > 0 ? (
          <div className="space-y-1">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Premium packs bundled
            </p>
            <div className="flex flex-wrap gap-1.5">
              {includedPacks.map((p) => (
                <Badge key={p} variant="secondary">
                  {p}
                </Badge>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-1">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Premium packs
            </p>
            <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Lock className="h-3 w-3" aria-hidden="true" />
              {tier === "community"
                ? "Not available — upgrade to add packs"
                : "Buy any pack as an add-on"}
            </p>
          </div>
        )}

        <div className="mt-auto space-y-2 pt-2">
          <Button
            className="w-full"
            variant={highlighted ? "default" : "outline"}
            onClick={handleCta}
            disabled={pending}
            data-testid={`tier-cta-${tier}`}
          >
            {pending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Opening Stripe…
              </>
            ) : (
              <>
                {cta.label}
                <ArrowRight className="h-4 w-4" />
              </>
            )}
          </Button>
          {error ? (
            <p
              role="alert"
              className="text-xs text-factor-deprecated-700"
              data-testid={`tier-error-${tier}`}
            >
              {error}
            </p>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}

function BillingRow({ billing }: { billing: TierBilling }) {
  if (billing.kind === "free") {
    return (
      <div>
        <p className="text-3xl font-bold tabular-nums">Free</p>
        <p className="text-xs text-muted-foreground">
          Forever. No card, no commitment.
        </p>
      </div>
    );
  }
  if (billing.kind === "usage") {
    return (
      <div>
        <p className="text-3xl font-bold tabular-nums">
          ${billing.monthlyPriceUsd.toLocaleString()}
          <span className="ml-1 text-sm font-normal text-muted-foreground">
            {billing.headlineUnit}
          </span>
        </p>
        <p className="text-xs text-muted-foreground">
          or ${billing.annualPriceUsd.toLocaleString()} / year (~17% off) •
          usage-based meters apply
        </p>
      </div>
    );
  }
  if (billing.kind === "annual") {
    return (
      <div>
        <p className="text-3xl font-bold tabular-nums">
          From ${(billing.annualPriceUsdFloor / 1000).toLocaleString()}k
          <span className="ml-1 text-sm font-normal text-muted-foreground">
            {billing.headlineUnit}
          </span>
        </p>
        <p className="text-xs text-muted-foreground">
          Annual contract • premium packs + multi-tenant included
        </p>
      </div>
    );
  }
  // contact
  return (
    <div>
      <p className="text-3xl font-bold">Custom</p>
      <p className="text-xs text-muted-foreground">
        High-ACV contracts • SSO/SCIM • VPC / private deployment
      </p>
    </div>
  );
}

const ACCENT: Record<TierKey, string> = {
  community: "border-l-emerald-500",
  pro: "border-l-sky-500",
  platform: "border-l-indigo-500",
  enterprise: "border-l-amber-500",
};
