import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { z } from "zod";
import { FactorDetail } from "@/components/FactorDetail";
import { getFactor } from "@/lib/api";
import { queryKeys } from "@/lib/query";

const searchSchema = z.object({
  edition: z.string().optional(),
  alternates: z.coerce.number().int().min(0).max(20).optional(),
});

export const Route = createFileRoute("/factors/$factorId")({
  validateSearch: (raw) => searchSchema.parse(raw),
  component: FactorDetailRoute,
});

function FactorDetailRoute() {
  const { factorId } = Route.useParams();
  const { edition } = Route.useSearch();
  const navigate = useNavigate({ from: "/factors/$factorId" });

  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.factor(factorId, edition),
    queryFn: () => getFactor(factorId, edition),
  });

  if (isLoading) {
    return (
      <div
        role="status"
        aria-busy="true"
        className="space-y-4"
      >
        <div className="h-32 animate-pulse rounded-lg border border-border bg-muted/40" />
        <div className="h-64 animate-pulse rounded-lg border border-border bg-muted/40" />
        <div className="h-48 animate-pulse rounded-lg border border-border bg-muted/40" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div
        role="alert"
        className="rounded-md border border-factor-deprecated-500 bg-factor-deprecated-50 p-4 text-sm text-factor-deprecated-700"
      >
        Factor <code className="font-mono">{factorId}</code> could not be
        loaded.
      </div>
    );
  }

  return (
    <>
      <FactorHead payload={data} />
      <FactorDetail
        payload={data}
        editionPin={edition}
        onEditionChange={(editionId) =>
          navigate({
            search: (prev) => ({
              ...prev,
              edition: editionId ?? undefined,
            }),
          })
        }
      />
    </>
  );
}

/**
 * Inject factor-specific SEO metadata into <head>. For preview / connector_only
 * / deprecated factors, this renders <meta name="robots" content="noindex,follow">.
 */
function FactorHead({
  payload,
}: {
  payload: Awaited<ReturnType<typeof getFactor>>;
}) {
  const { factor } = payload;
  const noindex = factor.factor_status !== "certified";

  if (typeof document !== "undefined") {
    document.title = `${factor.fuel_type} — ${factor.jurisdiction} — scope ${factor.scope} emission factor | GreenLang Factors`;

    upsertMeta(
      "description",
      `${factor.co2e_per_unit.toFixed(4)} kgCO₂e / ${factor.unit} — ${factor.provenance.source_org} ${factor.provenance.source_year} v${factor.provenance.source_version}. FQS ${factor.quality.rating}.`
    );
    upsertMeta("robots", noindex ? "noindex,follow" : "index,follow");
    upsertLink(
      "canonical",
      `https://factors.greenlang.io/factors/${factor.factor_id}`
    );
  }

  return null;
}

function upsertMeta(name: string, content: string) {
  let el = document.querySelector<HTMLMetaElement>(`meta[name="${name}"]`);
  if (!el) {
    el = document.createElement("meta");
    el.name = name;
    document.head.appendChild(el);
  }
  el.content = content;
}

function upsertLink(rel: string, href: string) {
  let el = document.querySelector<HTMLLinkElement>(`link[rel="${rel}"]`);
  if (!el) {
    el = document.createElement("link");
    el.rel = rel;
    document.head.appendChild(el);
  }
  el.href = href;
}
