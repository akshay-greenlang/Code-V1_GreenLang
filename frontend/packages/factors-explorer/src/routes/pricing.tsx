import { useEffect, useMemo, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Mail, X } from "lucide-react";
import {
  TierCard,
  type TierBilling,
  type TierKey,
  type CtaSpec,
} from "@/components/pricing/TierCard";
import {
  PremiumPackGrid,
  PREMIUM_PACKS,
} from "@/components/pricing/PremiumPackGrid";
import { ThreeLabelLegend } from "@/components/pricing/ThreeLabelLegend";
import { FAQSection } from "@/components/pricing/FAQSection";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import type { PremiumPackSku } from "@/lib/billing";

/**
 * Public Pricing Page for GreenLang Factors FY27.
 *
 * Layout:
 *   1. Hero (headline, sub, "no per-factor pricing" reassurance)
 *   2. Tier grid — 4 cards (Community / Pro / Platform / Enterprise)
 *   3. Premium pack grid — 7 add-on cards
 *   4. Three-label legend
 *   5. FAQ
 *
 * Tier and pack copy comes from this file but the SKU strings,
 * meter quotas, and entitlements come from the source of truth in
 * ``greenlang/factors/billing/skus.py`` and
 * ``greenlang/factors/tier_enforcement.py``. Anything quantitative is
 * either present-tense in the SKU catalog or flagged as "from $X" so
 * the UI never drifts ahead of the backend price book.
 */
export const Route = createFileRoute("/pricing")({
  component: PricingPage,
});

/**
 * Exported for unit testing — the test file mounts this component
 * directly under an in-memory router rather than depending on the
 * generated route tree.
 */
export function PricingPage() {
  const [selectedPacks, setSelectedPacks] = useState<Set<PremiumPackSku>>(
    new Set()
  );
  const [contactOpen, setContactOpen] = useState(false);

  const togglePack = (sku: PremiumPackSku) =>
    setSelectedPacks((prev) => {
      const next = new Set(prev);
      if (next.has(sku)) next.delete(sku);
      else next.add(sku);
      return next;
    });

  const addOnPacks: PremiumPackSku[] = useMemo(
    () => Array.from(selectedPacks),
    [selectedPacks]
  );

  const tiers = useMemo(() => buildTiers(() => setContactOpen(true)), []);

  return (
    <div className="space-y-12">
      {/* Hero */}
      <section className="space-y-3">
        <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">
          Pricing that scales with your usage, not your catalog.
        </h1>
        <p className="max-w-3xl text-muted-foreground">
          GreenLang Factors is open-core. Every tier — including the free
          one — sees the full Certified catalog. You pay for{" "}
          <strong>API calls</strong>, <strong>premium data packs</strong>,
          private factor registries, multi-tenant seats, OEM rights, and SLA
          — never per factor.
        </p>
        <div className="flex flex-wrap gap-2 pt-1">
          <Badge variant="secondary">Stripe-billed</Badge>
          <Badge variant="secondary">Cancel anytime on Pro</Badge>
          <Badge variant="secondary">Annual contracts on Platform &amp; Enterprise</Badge>
        </div>
      </section>

      {/* Tier grid */}
      <section
        aria-labelledby="tiers-heading"
        className="space-y-4"
        data-testid="tier-grid"
      >
        <h2 id="tiers-heading" className="sr-only">
          Subscription tiers
        </h2>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
          {tiers.map((t) => (
            <TierCard
              key={t.tier}
              {...t}
              addOnPacks={addOnPacks}
              highlighted={t.tier === "pro"}
            />
          ))}
        </div>
        {selectedPacks.size > 0 ? (
          <p className="text-xs text-muted-foreground">
            <strong>{selectedPacks.size}</strong> premium pack
            {selectedPacks.size === 1 ? "" : "s"} will be bundled into your
            Stripe Checkout when you click any Pro / Platform CTA above.
          </p>
        ) : null}
      </section>

      {/* Premium packs */}
      <section aria-labelledby="packs-heading" className="space-y-4">
        <div>
          <h2 id="packs-heading" className="text-2xl font-semibold">
            Premium data packs
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Each pack adds an additional family of certified factors to every
            edition. Buy as add-ons on Pro &amp; Platform or roll them into
            your Enterprise contract.
          </p>
        </div>
        <PremiumPackGrid selected={selectedPacks} onToggle={togglePack} />
      </section>

      {/* Three-label legend */}
      <section aria-labelledby="labels-heading" className="space-y-4">
        <div>
          <h2 id="labels-heading" className="text-2xl font-semibold">
            What each tier can resolve
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            The three-label model controls which redistribution classes are
            visible to each tier. Source of truth:{" "}
            <code className="rounded bg-muted px-1 font-mono text-xs">
              tier_enforcement.TierVisibility
            </code>
            .
          </p>
        </div>
        <ThreeLabelLegend />
      </section>

      {/* FAQ */}
      <section aria-labelledby="faq-heading" className="space-y-4">
        <h2 id="faq-heading" className="text-2xl font-semibold">
          Frequently asked questions
        </h2>
        <FAQSection />
      </section>

      {/* Final CTA banner */}
      <section className="rounded-lg border border-border bg-muted/30 p-6">
        <div className="flex flex-col items-start justify-between gap-3 md:flex-row md:items-center">
          <div>
            <h2 className="text-lg font-semibold">
              Need volume pricing, VPC, SSO, or a private factor registry?
            </h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Talk to sales — we&apos;ll size a contract from your call
              volume, target SLA, and which premium packs your team needs.
            </p>
          </div>
          <Button onClick={() => setContactOpen(true)}>
            <Mail className="h-4 w-4" />
            Talk to sales
          </Button>
        </div>
      </section>

      {contactOpen ? (
        <ContactSalesModal onClose={() => setContactOpen(false)} />
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tier definitions
// ---------------------------------------------------------------------------

interface TierDef {
  tier: TierKey;
  name: string;
  tagline: string;
  targetBuyer: string;
  billing: TierBilling;
  features: string[];
  includedPacks: string[];
  rateLimits: string[];
  cta: CtaSpec;
}

function buildTiers(openContact: () => void): TierDef[] {
  return [
    {
      tier: "community",
      name: "Community",
      tagline: "Open-core SDK + a public sandbox API.",
      targetBuyer: "individual developers, students, OSS projects",
      billing: { kind: "free" },
      rateLimits: [
        "1,000 API calls / month",
        "1,000 batch rows / month",
        "100 calls / day soft throttle",
      ],
      features: [
        "Factor schema, SDK, CLI, docs",
        "Public Certified factor pack",
        "Sandbox API (corporate Scope 1 & 2 + electricity)",
        "Example mappings + 5-min Quickstart",
        "Community Slack + GitHub issues",
      ],
      includedPacks: [],
      cta: {
        kind: "external",
        href: "/sign-up?tier=community",
        label: "Get free API key",
      },
    },
    {
      tier: "pro",
      name: "Developer Pro",
      tagline: "Production API + version pinning + premium pack add-ons.",
      targetBuyer: "product teams shipping climate features",
      billing: {
        kind: "usage",
        monthlyPriceUsd: 299,
        annualPriceUsd: 2988,
        headlineUnit: "/mo",
      },
      rateLimits: [
        "100,000 API calls / month included",
        "100,000 batch rows / month included",
        "1 GB private registry storage",
        "Overage: $0.002 / call, $0.0005 / row",
      ],
      features: [
        "Production API + edition pinning",
        "Hosted explain logs + signed receipts",
        "50 private overrides per project",
        "Email + chat support",
        "99.5% SLA",
      ],
      includedPacks: [],
      cta: {
        kind: "checkout",
        skuName: "pro_monthly",
        label: "Start Pro on Stripe",
      },
    },
    {
      tier: "platform",
      name: "Platform / Consulting",
      tagline:
        "Multi-tenant workspaces, white-label option, audit exports.",
      targetBuyer:
        "consultancies, climate platforms, mid-market software vendors",
      billing: {
        kind: "annual",
        annualPriceUsdFloor: 50000,
        headlineUnit: "/yr",
      },
      rateLimits: [
        "5M API calls / month included",
        "5M batch rows / month included",
        "100 sub-tenant seats",
        "3 OEM white-label sites bundled",
        "Audit-bundle export enabled",
      ],
      features: [
        "Multi-client workspaces + override vaults",
        "Premium packs bundled (Electricity, Freight, CBAM)",
        "Audit-grade exports for client engagements",
        "Partner support + onboarding",
        "99.9% SLA",
      ],
      includedPacks: ["Electricity", "Freight", "CBAM / EU Policy"],
      cta: {
        kind: "checkout",
        skuName: "platform_annual",
        label: "Start Platform on Stripe",
      },
    },
    {
      tier: "enterprise",
      name: "Enterprise",
      tagline:
        "VPC / private deployment, SSO/SCIM, signed releases, named SLA.",
      targetBuyer: "regulated enterprises, large climate programs",
      billing: { kind: "contact" },
      rateLimits: [
        "10M+ API calls / month (negotiable)",
        "Unlimited private registry",
        "Unlimited sub-tenants",
        "Customer-specific factors + signed releases",
      ],
      features: [
        "SSO / SCIM, role mapping, audit log feed",
        "VPC peering or fully private deployment",
        "Approval workflows for factor changes",
        "Named SLA (99.95%) + procurement / DPA support",
        "Premium packs negotiated into ACV",
      ],
      includedPacks: [],
      cta: {
        kind: "modal",
        label: "Talk to sales",
        onClick: openContact,
      },
    },
  ];
}

// ---------------------------------------------------------------------------
// Contact-sales modal
// ---------------------------------------------------------------------------

interface ContactFormState {
  email: string;
  companySize: string;
  useCase: string;
  submitted: boolean;
  error: string | null;
}

function ContactSalesModal({ onClose }: { onClose: () => void }) {
  const [state, setState] = useState<ContactFormState>({
    email: "",
    companySize: "1-50",
    useCase: "",
    submitted: false,
    error: null,
  });

  // ESC closes
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!state.email.includes("@")) {
      setState((s) => ({ ...s, error: "A valid email is required." }));
      return;
    }
    if (state.useCase.trim().length < 10) {
      setState((s) => ({
        ...s,
        error: "Tell us a little about your use case (10+ chars).",
      }));
      return;
    }
    // The real submission will be wired to the same /v1/billing/contact
    // endpoint as the rest of the lifecycle when the backend lands.
    // For now we optimistically mark the form as submitted.
    setState((s) => ({ ...s, submitted: true, error: null }));
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="contact-modal-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
      data-testid="contact-sales-modal"
    >
      <Card
        className="w-full max-w-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <CardContent className="space-y-4 p-6">
          <div className="flex items-start justify-between">
            <div>
              <h2
                id="contact-modal-title"
                className="text-xl font-semibold"
              >
                Talk to sales
              </h2>
              <p className="mt-1 text-sm text-muted-foreground">
                We&apos;ll size your contract from API call volume, SLA needs,
                and the premium packs you want bundled.
              </p>
            </div>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close"
              className="rounded-md p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          {state.submitted ? (
            <div
              role="status"
              className="rounded-md border border-emerald-500 bg-emerald-50 p-4 text-sm text-emerald-700"
            >
              Thanks — we&apos;ll be in touch within 1 business day at{" "}
              <strong>{state.email}</strong>.
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-3">
              <div className="space-y-1">
                <label
                  htmlFor="contact-email"
                  className="text-sm font-medium"
                >
                  Work email
                </label>
                <Input
                  id="contact-email"
                  type="email"
                  required
                  value={state.email}
                  onChange={(e) =>
                    setState((s) => ({ ...s, email: e.target.value }))
                  }
                  placeholder="you@company.com"
                />
              </div>

              <div className="space-y-1">
                <label
                  htmlFor="contact-size"
                  className="text-sm font-medium"
                >
                  Company size
                </label>
                <select
                  id="contact-size"
                  className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm"
                  value={state.companySize}
                  onChange={(e) =>
                    setState((s) => ({ ...s, companySize: e.target.value }))
                  }
                >
                  <option>1-50</option>
                  <option>51-250</option>
                  <option>251-1,000</option>
                  <option>1,001-5,000</option>
                  <option>5,000+</option>
                </select>
              </div>

              <div className="space-y-1">
                <label
                  htmlFor="contact-use-case"
                  className="text-sm font-medium"
                >
                  Use case
                </label>
                <textarea
                  id="contact-use-case"
                  required
                  minLength={10}
                  rows={4}
                  value={state.useCase}
                  onChange={(e) =>
                    setState((s) => ({ ...s, useCase: e.target.value }))
                  }
                  className="flex w-full rounded-md border border-input bg-background p-3 text-sm shadow-sm"
                  placeholder="Which premium packs, how many API calls/month, target SLA, deployment model…"
                />
              </div>

              {state.error ? (
                <p
                  role="alert"
                  className="text-sm text-factor-deprecated-700"
                >
                  {state.error}
                </p>
              ) : null}

              <div className="flex items-center justify-between gap-2 pt-2">
                <p className="text-xs text-muted-foreground">
                  We respond within 1 business day.
                </p>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={onClose}
                  >
                    Cancel
                  </Button>
                  <Button type="submit">Send</Button>
                </div>
              </div>
            </form>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// ``PREMIUM_PACKS`` is re-exported via the index of this module so it
// stays trivially importable from the test file without pulling the
// route component itself.
export { PREMIUM_PACKS };
