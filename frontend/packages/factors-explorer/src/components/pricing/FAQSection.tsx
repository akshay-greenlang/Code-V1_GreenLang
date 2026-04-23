import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Common pricing questions surfaced under the tier grid.
 *
 * Implemented as a native ``<details>`` per question for zero-JS
 * accessibility — the parent component just adds an animated chevron.
 */
export interface FAQItem {
  q: string;
  a: React.ReactNode;
}

const FAQ_ITEMS: FAQItem[] = [
  {
    q: "What counts as one API call?",
    a: (
      <>
        A single HTTP request to a metered endpoint —{" "}
        <code className="rounded bg-muted px-1 font-mono text-xs">
          /factors/resolve
        </code>
        ,{" "}
        <code className="rounded bg-muted px-1 font-mono text-xs">
          /factors/search
        </code>
        ,{" "}
        <code className="rounded bg-muted px-1 font-mono text-xs">
          /factors/&#123;id&#125;/explain
        </code>
        , and the{" "}
        <code className="rounded bg-muted px-1 font-mono text-xs">/match</code>{" "}
        family. Bulk endpoints (
        <code className="rounded bg-muted px-1 font-mono text-xs">
          /match_bulk
        </code>
        ,{" "}
        <code className="rounded bg-muted px-1 font-mono text-xs">/export</code>
        ) instead consume from your <strong>Batch rows</strong> meter, one row
        per record returned. Pagination, edition pinning, and signed-receipt
        verification are <strong>never</strong> billed.
      </>
    ),
  },
  {
    q: "Is there a free trial?",
    a: (
      <>
        Yes — the <strong>Community</strong> tier is free forever and gives you
        1,000 API calls / month against the open-core method profiles
        (corporate Scope 1 & 2, electricity). For Pro you can book a 14-day
        guided trial through the contact form; we provision a real Stripe
        subscription with a $0 trial price so you do not have to migrate
        anything when you decide to keep it.
      </>
    ),
  },
  {
    q: "Can I downgrade or cancel?",
    a: (
      <>
        Yes. Pro is month-to-month and you can downgrade at any time from the
        billing portal — entitlements drop to the lower tier at the end of the
        current period and your data, tenants, and signed receipts stay
        intact. Consulting / Platform / Enterprise are annual contracts;
        downgrade requests follow the renewal clause in your order form.
      </>
    ),
  },
  {
    q: "Do you offer OEM / white-label terms?",
    a: (
      <>
        OEM redistribution is included in <strong>Platform</strong> (3 sites
        bundled, $500/site after that) and is a negotiated add-on on{" "}
        <strong>Enterprise</strong>. White-label requires a signed OEM rider
        plus per-pack confirmation that the underlying source allows
        redistribution — for example, the Product Carbon / LCI and PCAF
        Finance packs require that the customer brings their own ecoinvent /
        Sphera / PCAF license. Community and Pro tiers cannot redistribute or
        white-label any factor.
      </>
    ),
  },
  {
    q: "How is pricing related to the number of factors?",
    a: (
      <>
        It isn&apos;t. Every tier — including Community — sees the full
        Certified catalog. We price on{" "}
        <strong>API calls, batch rows, premium pack entitlements, private
        registry storage, multi-tenant seats, OEM rights, and SLA tier</strong>
        . That keeps the catalog incentive aligned: more Certified factors
        helps every customer, not just the ones on the biggest plan.
      </>
    ),
  },
  {
    q: "What's a Premium Data Pack vs. an Edition?",
    a: (
      <>
        An <strong>Edition</strong> is a versioned snapshot of the entire
        catalog (signed manifest, fingerprint, immutable) — pin to one for
        reproducibility. A <strong>Premium Pack</strong> is a paid entitlement
        that unlocks an additional <em>family</em> of certified factors inside
        every edition (Electricity, Freight, EPDs, etc.). You can mix one
        Edition with any combination of packs you have entitlement for.
      </>
    ),
  },
];

interface FAQSectionProps {
  /** Override item list (e.g. for tests). */
  items?: FAQItem[];
}

export function FAQSection({ items = FAQ_ITEMS }: FAQSectionProps) {
  return (
    <div className="space-y-2" data-testid="faq-section">
      {items.map((item, idx) => (
        <FAQRow key={idx} item={item} />
      ))}
    </div>
  );
}

function FAQRow({ item }: { item: FAQItem }) {
  const [open, setOpen] = useState(false);
  return (
    <div
      className="rounded-lg border border-border bg-card transition-colors"
      data-testid="faq-row"
    >
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-4 px-4 py-3 text-left"
      >
        <span className="font-medium">{item.q}</span>
        <ChevronDown
          className={cn(
            "h-4 w-4 flex-shrink-0 text-muted-foreground transition-transform",
            open && "rotate-180"
          )}
          aria-hidden="true"
        />
      </button>
      {open ? (
        <div className="px-4 pb-4 text-sm text-muted-foreground">
          {item.a}
        </div>
      ) : null}
    </div>
  );
}
