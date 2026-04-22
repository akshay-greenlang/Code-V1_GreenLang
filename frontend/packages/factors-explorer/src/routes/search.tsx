import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { z } from "zod";
import { SearchBar } from "@/components/SearchBar";
import { FactorCard } from "@/components/FactorCard";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { searchFactors } from "@/lib/api";
import { queryKeys } from "@/lib/query";
import {
  FactorStatusSchema,
  LicenseClassSchema,
  MethodProfileSchema,
} from "@/types/factors";

/**
 * URL-as-state faceted search page.
 * Filters: family, jurisdiction, method_profile, source, validity date,
 * quality tier. Pagination via offset + limit.
 */
const searchParamsSchema = z.object({
  q: z.string().default(""),
  family: z.string().optional(),
  jurisdiction: z.string().optional(),
  method_profile: MethodProfileSchema.optional(),
  source_id: z.string().optional(),
  factor_status: FactorStatusSchema.optional(),
  license_class: LicenseClassSchema.optional(),
  dqs_min: z.coerce.number().min(0).max(100).optional(),
  valid_on_date: z.string().optional(),
  sort_by: z
    .enum(["relevance", "dqs_score", "co2e_total", "source_year", "factor_id"])
    .default("relevance"),
  sort_order: z.enum(["asc", "desc"]).default("desc"),
  offset: z.coerce.number().int().min(0).default(0),
  limit: z.coerce.number().int().min(1).max(200).default(20),
});

type SearchParams = z.infer<typeof searchParamsSchema>;

export const Route = createFileRoute("/search")({
  validateSearch: (raw) => searchParamsSchema.parse(raw),
  component: SearchPage,
});

function SearchPage() {
  const params = Route.useSearch();
  const navigate = useNavigate({ from: "/search" });

  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.search(params),
    queryFn: () =>
      searchFactors({
        query: params.q,
        family: params.family ?? null,
        jurisdiction: params.jurisdiction ?? null,
        method_profile: params.method_profile ?? null,
        source_id: params.source_id ?? null,
        factor_status: params.factor_status ?? null,
        license_class: params.license_class ?? null,
        dqs_min: params.dqs_min ?? null,
        valid_on_date: params.valid_on_date ?? null,
        sort_by: params.sort_by,
        sort_order: params.sort_order,
        offset: params.offset,
        limit: params.limit,
      }),
  });

  const updateSearch = (patch: Partial<SearchParams>) => {
    navigate({ search: (prev) => ({ ...prev, ...patch, offset: 0 }) });
  };

  const total = data?.total ?? 0;
  const pageStart = params.offset + 1;
  const pageEnd = Math.min(params.offset + params.limit, total);

  return (
    <div className="grid grid-cols-1 gap-6 md:grid-cols-[240px_1fr]">
      {/* Sidebar facets */}
      <aside className="space-y-4 md:sticky md:top-20 md:self-start">
        <FacetSection title="Family">
          {(data?.facets?.families ?? []).map((f) => (
            <FacetCheckbox
              key={f.value}
              checked={params.family === f.value}
              label={f.value}
              count={f.count}
              onToggle={() =>
                updateSearch({
                  family: params.family === f.value ? undefined : f.value,
                })
              }
            />
          ))}
        </FacetSection>

        <FacetSection title="Jurisdiction">
          {(data?.facets?.jurisdictions ?? []).map((j) => (
            <FacetCheckbox
              key={j.value}
              checked={params.jurisdiction === j.value}
              label={j.value}
              count={j.count}
              onToggle={() =>
                updateSearch({
                  jurisdiction:
                    params.jurisdiction === j.value ? undefined : j.value,
                })
              }
            />
          ))}
        </FacetSection>

        <FacetSection title="Method profile">
          {(data?.facets?.method_profiles ?? []).map((m) => (
            <FacetCheckbox
              key={m.value}
              checked={params.method_profile === m.value}
              label={m.value}
              count={m.count}
              onToggle={() =>
                updateSearch({
                  method_profile:
                    params.method_profile === m.value ? undefined : m.value,
                })
              }
            />
          ))}
        </FacetSection>

        <FacetSection title="Source">
          {(data?.facets?.sources ?? []).map((s) => (
            <FacetCheckbox
              key={s.value}
              checked={params.source_id === s.value}
              label={s.label || s.value}
              count={s.count}
              onToggle={() =>
                updateSearch({
                  source_id:
                    params.source_id === s.value ? undefined : s.value,
                })
              }
            />
          ))}
        </FacetSection>

        <FacetSection title="Valid on date">
          <input
            type="date"
            value={params.valid_on_date ?? ""}
            onChange={(e) =>
              updateSearch({ valid_on_date: e.target.value || undefined })
            }
            className="w-full rounded-md border border-input bg-background px-2 py-1 text-sm"
          />
        </FacetSection>

        <FacetSection title="Quality (FQS min)">
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={params.dqs_min ?? 0}
            onChange={(e) =>
              updateSearch({ dqs_min: Number(e.target.value) || undefined })
            }
            className="w-full"
            aria-label="Minimum FQS"
          />
          <p className="mt-1 text-xs text-muted-foreground">
            ≥ {params.dqs_min ?? 0}
          </p>
        </FacetSection>
      </aside>

      {/* Results */}
      <section className="space-y-4">
        <SearchBar
          value={params.q}
          onChange={(q) => updateSearch({ q })}
          onSubmit={(q) => updateSearch({ q })}
        />

        <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-muted-foreground">
          <div>
            {isLoading ? (
              "Searching…"
            ) : error ? (
              <span className="text-factor-deprecated-700">
                Error: unable to load results.
              </span>
            ) : total === 0 ? (
              "No results."
            ) : (
              <>
                <span className="tabular-nums">{total.toLocaleString()}</span>{" "}
                results • showing{" "}
                <span className="tabular-nums">{pageStart}</span>–
                <span className="tabular-nums">{pageEnd}</span>
              </>
            )}
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs">Sort by</label>
            <select
              value={params.sort_by}
              onChange={(e) =>
                updateSearch({
                  sort_by: e.target.value as SearchParams["sort_by"],
                })
              }
              className="rounded-md border border-input bg-background px-2 py-1 text-xs"
            >
              <option value="relevance">relevance</option>
              <option value="dqs_score">DQS</option>
              <option value="co2e_total">CO₂e total</option>
              <option value="source_year">year</option>
              <option value="factor_id">id</option>
            </select>
            <select
              value={params.sort_order}
              onChange={(e) =>
                updateSearch({
                  sort_order: e.target.value as "asc" | "desc",
                })
              }
              className="rounded-md border border-input bg-background px-2 py-1 text-xs"
            >
              <option value="desc">desc</option>
              <option value="asc">asc</option>
            </select>
          </div>
        </div>

        {isLoading ? (
          <SkeletonList />
        ) : (
          <ul className="space-y-2">
            {(data?.items ?? []).map((f) => (
              <li key={f.factor_id}>
                <FactorCard factor={f} />
              </li>
            ))}
          </ul>
        )}

        <div className="flex items-center justify-between pt-2">
          <Button
            variant="outline"
            size="sm"
            disabled={params.offset === 0}
            onClick={() =>
              navigate({
                search: (prev) => ({
                  ...prev,
                  offset: Math.max(0, prev.offset - prev.limit),
                }),
              })
            }
          >
            Previous
          </Button>
          <span className="text-xs text-muted-foreground">
            Page{" "}
            {Math.floor(params.offset / params.limit) + 1} of{" "}
            {Math.max(1, Math.ceil(total / params.limit))}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={pageEnd >= total}
            onClick={() =>
              navigate({
                search: (prev) => ({
                  ...prev,
                  offset: prev.offset + prev.limit,
                }),
              })
            }
          >
            Next
          </Button>
        </div>
      </section>
    </div>
  );
}

function FacetSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <CardContent className="space-y-1.5 p-3">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {title}
        </h3>
        <div className="space-y-0.5">{children}</div>
      </CardContent>
    </Card>
  );
}

function FacetCheckbox({
  checked,
  label,
  count,
  onToggle,
}: {
  checked: boolean;
  label: string;
  count: number;
  onToggle: () => void;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-2 rounded px-1 py-0.5 text-sm hover:bg-accent">
      <span className="flex items-center gap-1.5 truncate">
        <input
          type="checkbox"
          checked={checked}
          onChange={onToggle}
          className="accent-primary"
        />
        <span className="truncate">{label}</span>
      </span>
      <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
        {count}
      </span>
    </label>
  );
}

function SkeletonList() {
  return (
    <ul className="space-y-2" aria-busy="true">
      {Array.from({ length: 6 }).map((_, i) => (
        <li
          key={i}
          className="h-24 animate-pulse rounded-lg border border-border bg-muted/40"
        />
      ))}
    </ul>
  );
}
