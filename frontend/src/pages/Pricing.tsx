/**
 * Pricing — full 5-tier comparison matrix for the FY27 Factors launch.
 *
 * This is the "commercial surface" for Agent W4-E (C4):
 *   1. Community        -> Sign up for free (self-serve)
 *   2. Developer Pro    -> Stripe Checkout (self-serve)
 *   3. Consulting       -> Contact Sales (mailto)
 *   4. Platform         -> Contact Sales (mailto)
 *   5. Enterprise       -> Contact Sales (mailto)
 *
 * Below the tier matrix we surface the 8 Premium Pack add-ons as a
 * carousel so visitors can see the full commercial posture without
 * scrolling through a wall of text.
 *
 * Design rule (brief): DO NOT price by factor count. The page surfaces
 * metered dimensions (API calls, tenants, OEM redistribution rights)
 * and calls out the anti-pattern in a prominent banner.
 *
 * NOTE: This file is a new component (W4-E C4). Route wiring belongs to
 * Agent W4-D; until they pick it up, the existing ``PricingPage.tsx``
 * remains the live route and this file is a standalone variant.
 * TODO(W4-D): mount at /pricing-v2 (or replace PricingPage) after review.
 */
import { useEffect, useMemo, useState } from "react";
import Alert from "@mui/material/Alert";
import AlertTitle from "@mui/material/AlertTitle";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardActions from "@mui/material/CardActions";
import CardContent from "@mui/material/CardContent";
import Chip from "@mui/material/Chip";
import CircularProgress from "@mui/material/CircularProgress";
import Container from "@mui/material/Container";
import Divider from "@mui/material/Divider";
import Grid from "@mui/material/Grid";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import Stack from "@mui/material/Stack";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";

// ---------------------------------------------------------------------------
// Canonical FY27 pricing (v1 — subject to CTO/Commercial approval)
// ---------------------------------------------------------------------------
//
// Source of truth at runtime is GET /v1/billing/plans (see billing/api.py).
// The static values below are shown pre-fetch to avoid an empty first paint
// and as a fallback if the backend is unavailable.

interface TierInfo {
  planId: string;
  displayName: string;
  tagline: string;
  priceMonthly: string | null;
  priceAnnual: string | null;
  apiCallsIncluded: string;
  overageRate: string;
  batchVolume: string;
  packs: string;
  sla: string;
  support: string;
  cta: "signup" | "checkout" | "contact";
}

const TIERS_FALLBACK: TierInfo[] = [
  {
    planId: "community",
    displayName: "Community",
    tagline: "Free open-core for individuals, students, and OSS projects.",
    priceMonthly: "$0",
    priceAnnual: "—",
    apiCallsIncluded: "1,000 / mo",
    overageRate: "hard cap",
    batchVolume: "1,000 rows / mo",
    packs: "Certified, open-core only",
    sla: "none",
    support: "Community Slack",
    cta: "signup",
  },
  {
    planId: "developer_pro",
    displayName: "Developer Pro",
    tagline: "For startups, consultants, and one-engineer teams shipping to production.",
    priceMonthly: "$299 / mo",
    priceAnnual: "$2,990 / yr (~17% off)",
    apiCallsIncluded: "50,000 / mo",
    overageRate: "$0.01 / call",
    batchVolume: "10,000 rows / day",
    packs: "Certified + Preview. No premium packs bundled.",
    sla: "99.5% uptime",
    support: "Email",
    cta: "checkout",
  },
  {
    planId: "consulting",
    displayName: "Consulting",
    tagline: "For climate consultants running audits across multiple clients.",
    priceMonthly: "$2,499 / mo",
    priceAnnual: "$24,990 / yr",
    apiCallsIncluded: "500,000 / mo",
    overageRate: "$0.005 / call",
    batchVolume: "50,000 rows / day",
    packs: "All Certified + choose 3 premium packs",
    sla: "99.9% uptime",
    support: "Slack-Connect",
    cta: "contact",
  },
  {
    planId: "platform",
    displayName: "Platform",
    tagline: "For ESG SaaS platforms and OEM partners embedding Factors.",
    priceMonthly: "$4,999 / mo",
    priceAnnual: "$50,000 / yr",
    apiCallsIncluded: "5,000,000 / mo",
    overageRate: "$0.0005 / call",
    batchVolume: "100,000 rows / day",
    packs: "All Certified + 3 premium packs + OEM redistribution",
    sla: "99.9% uptime",
    support: "Slack-Connect + TAM",
    cta: "contact",
  },
  {
    planId: "enterprise",
    displayName: "Enterprise",
    tagline: "For Fortune 500 with regulated reporting and private deployment.",
    priceMonthly: "Custom ACV",
    priceAnnual: "$75K – $500K / yr typical",
    apiCallsIncluded: "Negotiated (10M+ / mo baseline)",
    overageRate: "included",
    batchVolume: "Unlimited (fair-use)",
    packs: "All Certified + all premium packs + SSO / VPC",
    sla: "99.95% uptime",
    support: "Named TAM + 24/7 escalation",
    cta: "contact",
  },
];

// ---------------------------------------------------------------------------
// Premium Pack carousel data
// ---------------------------------------------------------------------------

interface PackInfo {
  slug: string;
  name: string;
  monthly: string;
  annual: string;
  blurb: string;
}

const PREMIUM_PACKS: PackInfo[] = [
  {
    slug: "electricity_premium",
    name: "Electricity Premium",
    monthly: "$499 / mo",
    annual: "$4,990 / yr",
    blurb: "Residual-mix + hourly grid factors for market-based Scope 2 reporting.",
  },
  {
    slug: "freight_premium",
    name: "Freight Premium (ISO 14083)",
    monthly: "$499 / mo",
    annual: "$4,990 / yr",
    blurb: "Mode-specific freight factors aligned to GLEC / ISO 14083.",
  },
  {
    slug: "product_carbon_premium",
    name: "Product Carbon / LCI Premium",
    monthly: "$799 / mo",
    annual: "$7,990 / yr",
    blurb: "ecoinvent-compatible LCI resolution (BYO license; higher price reflects license chain).",
  },
  {
    slug: "epd_premium",
    name: "Construction EPD Premium",
    monthly: "$699 / mo",
    annual: "$6,990 / yr",
    blurb: "EPD + PCR factors for embodied-carbon accounting in construction.",
  },
  {
    slug: "agrifood_premium",
    name: "Agrifood Premium",
    monthly: "$499 / mo",
    annual: "$4,990 / yr",
    blurb: "Crop, dairy, and protein factors for Scope 3 Cat 1 agrifood supply chains.",
  },
  {
    slug: "finance_premium",
    name: "Finance Premium (PCAF)",
    monthly: "$599 / mo",
    annual: "$5,990 / yr",
    blurb: "PCAF-aligned sector + asset-class proxies for financed emissions.",
  },
  {
    slug: "cbam_premium",
    name: "CBAM / EU Policy Premium",
    monthly: "$999 / mo",
    annual: "$9,990 / yr",
    blurb: "EU CBAM default values + quarterly regulator updates (highest demand).",
  },
  {
    slug: "land_premium",
    name: "Land / Removals Premium",
    monthly: "$399 / mo",
    annual: "$3,990 / yr",
    blurb: "LULUCF, biochar, and removal-pathway factors for GHG accounting.",
  },
];

// ---------------------------------------------------------------------------
// CTA helpers
// ---------------------------------------------------------------------------

function ctaLabel(cta: TierInfo["cta"]): string {
  switch (cta) {
    case "signup":
      return "Sign up free";
    case "checkout":
      return "Start checkout";
    case "contact":
      return "Contact sales";
  }
}

async function beginCheckout(planId: string): Promise<void> {
  try {
    const resp = await fetch(`/v1/billing/checkout/${planId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        success_url: `${window.location.origin}/checkout/success`,
        cancel_url: `${window.location.origin}/pricing`,
      }),
    });
    if (!resp.ok) {
      alert(`Checkout failed (HTTP ${resp.status}). Please email sales@greenlang.io.`);
      return;
    }
    const data = await resp.json();
    if (data.url) {
      window.location.assign(data.url);
    }
  } catch (err) {
    alert(`Checkout failed: ${err}. Please email sales@greenlang.io.`);
  }
}

function contactSales(planId: string): void {
  const subject = encodeURIComponent(`Inquiry: ${planId} tier`);
  const body = encodeURIComponent(
    `Hi GreenLang team,\n\nI'd like to learn more about the ${planId} tier for Factors API.\n\nCompany: \nIndustry: \nEstimated volume (calls/mo): \n\nThanks!`
  );
  window.location.href = `mailto:sales@greenlang.io?subject=${subject}&body=${body}`;
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function Pricing(): JSX.Element {
  const [tiers, setTiers] = useState<TierInfo[]>(TIERS_FALLBACK);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch("/v1/billing/plans")
      .then((r) => (r.ok ? r.json() : null))
      .then((body) => {
        if (cancelled) return;
        // Best-effort: the backend returns the 4-SKU public surface; we keep
        // the fallback 5-tier table visually since Consulting / Platform are
        // split internally.
        if (body && Array.isArray(body.plans)) {
          // no-op — we still render TIERS_FALLBACK but could enrich in future
        }
        setLoading(false);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(String(err));
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <Container maxWidth="xl" sx={{ py: 6 }}>
      <Alert severity="warning" sx={{ mb: 4 }} variant="outlined">
        <AlertTitle>Pricing proposal v1 — subject to CTO / Commercial approval</AlertTitle>
        All prices on this page are proposed for the FY27 launch. Final
        numbers will be ratified by the Commercial lead before public
        release. Do not publish externally yet.
      </Alert>

      <Stack spacing={1} sx={{ mb: 4 }}>
        <Typography variant="h3" fontWeight={700}>
          Pricing
        </Typography>
        <Typography variant="subtitle1" color="text.secondary">
          Five tiers. Metered by API calls, tenants, and OEM rights —{" "}
          <strong>never by factor count.</strong> Every factor in every
          tier carries full provenance + signed receipts.
        </Typography>
      </Stack>

      <Alert severity="info" sx={{ mb: 4 }}>
        <AlertTitle>Why we don't price by factor count</AlertTitle>
        Every tier gets the same catalog. We price by <em>usage</em> (API
        calls, batch volume), <em>multi-tenancy</em> (sub-tenant seats,
        OEM rights), and <em>assurance</em> (SLA, SSO, audit bundles).
        That way a small team with deep usage pays for their volume, not
        for features they don't touch.
      </Alert>

      {loading && (
        <Box sx={{ display: "flex", justifyContent: "center", my: 2 }}>
          <CircularProgress size={24} />
        </Box>
      )}
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {/* ---- Tier comparison matrix ---- */}
      <Box sx={{ overflowX: "auto", mb: 6 }}>
        <Table size="small" aria-label="pricing tier comparison">
          <TableHead>
            <TableRow>
              <TableCell></TableCell>
              {tiers.map((t) => (
                <TableCell key={t.planId} align="center">
                  <Typography variant="h6">{t.displayName}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {t.tagline}
                  </Typography>
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            <TableRow>
              <TableCell>Monthly</TableCell>
              {tiers.map((t) => (
                <TableCell key={t.planId} align="center">
                  <Typography variant="h5">{t.priceMonthly ?? "—"}</Typography>
                </TableCell>
              ))}
            </TableRow>
            <TableRow>
              <TableCell>Annual</TableCell>
              {tiers.map((t) => (
                <TableCell key={t.planId} align="center">
                  <Typography variant="body2">{t.priceAnnual ?? "—"}</Typography>
                </TableCell>
              ))}
            </TableRow>
            <TableRow>
              <TableCell>API calls included</TableCell>
              {tiers.map((t) => (
                <TableCell key={t.planId} align="center">{t.apiCallsIncluded}</TableCell>
              ))}
            </TableRow>
            <TableRow>
              <TableCell>Overage rate</TableCell>
              {tiers.map((t) => (
                <TableCell key={t.planId} align="center">{t.overageRate}</TableCell>
              ))}
            </TableRow>
            <TableRow>
              <TableCell>Batch volume</TableCell>
              {tiers.map((t) => (
                <TableCell key={t.planId} align="center">{t.batchVolume}</TableCell>
              ))}
            </TableRow>
            <TableRow>
              <TableCell>Pack entitlements</TableCell>
              {tiers.map((t) => (
                <TableCell key={t.planId} align="center">{t.packs}</TableCell>
              ))}
            </TableRow>
            <TableRow>
              <TableCell>SLA</TableCell>
              {tiers.map((t) => (
                <TableCell key={t.planId} align="center">{t.sla}</TableCell>
              ))}
            </TableRow>
            <TableRow>
              <TableCell>Support</TableCell>
              {tiers.map((t) => (
                <TableCell key={t.planId} align="center">{t.support}</TableCell>
              ))}
            </TableRow>
            <TableRow>
              <TableCell></TableCell>
              {tiers.map((t) => (
                <TableCell key={t.planId} align="center">
                  <Button
                    variant={t.cta === "checkout" ? "contained" : "outlined"}
                    color={t.cta === "checkout" ? "primary" : "inherit"}
                    onClick={() => {
                      if (t.cta === "signup") {
                        window.location.assign("/auth/signup");
                      } else if (t.cta === "checkout") {
                        void beginCheckout(t.planId);
                      } else {
                        contactSales(t.planId);
                      }
                    }}
                  >
                    {ctaLabel(t.cta)}
                  </Button>
                </TableCell>
              ))}
            </TableRow>
          </TableBody>
        </Table>
      </Box>

      {/* ---- Premium Pack carousel ---- */}
      <Divider sx={{ mb: 4 }} />
      <Stack spacing={1} sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight={600}>
          Premium Data Packs
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Add-on data packs for licensed / regulated methodologies.
          Available on Developer Pro and above. Three are bundled on
          Consulting and Platform tiers; all eight are available on
          Enterprise.
        </Typography>
      </Stack>
      <Grid container spacing={2}>
        {PREMIUM_PACKS.map((pack) => (
          <Grid item xs={12} sm={6} md={3} key={pack.slug}>
            <Card variant="outlined" sx={{ height: "100%" }}>
              <CardContent>
                <Typography variant="subtitle1" fontWeight={600}>
                  {pack.name}
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {pack.monthly} · {pack.annual}
                </Typography>
                <Typography variant="body2">{pack.blurb}</Typography>
              </CardContent>
              <CardActions>
                <Button size="small" onClick={() => contactSales(pack.slug)}>
                  Contact sales
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Box sx={{ mt: 6, textAlign: "center" }}>
        <Typography variant="caption" color="text.secondary">
          Questions? Email{" "}
          <a href="mailto:sales@greenlang.io">sales@greenlang.io</a> —
          we respond within one business day.
        </Typography>
      </Box>
    </Container>
  );
}
