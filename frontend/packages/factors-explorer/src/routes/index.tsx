import { useState } from "react";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Check, Copy } from "lucide-react";
import { ThreeLabelDashboard } from "@/components/ThreeLabelDashboard";
import { SourceTile } from "@/components/SourceTile";
import { SearchBar } from "@/components/SearchBar";
import { EditionPin } from "@/components/EditionPin";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { getSources } from "@/lib/api";
import { queryKeys } from "@/lib/query";
import { copyToClipboard } from "@/lib/utils";

export const Route = createFileRoute("/")({
  component: LandingPage,
});

type Lang = "python" | "typescript" | "curl";

function LandingPage() {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [lang, setLang] = useState<Lang>("python");

  const { data: sources } = useQuery({
    queryKey: queryKeys.sources(),
    queryFn: getSources,
  });

  const topSources = (sources ?? []).slice(0, 8);

  return (
    <div className="space-y-10">
      <section className="space-y-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">
              Emission factors you can ship.
            </h1>
            <p className="mt-2 max-w-2xl text-muted-foreground">
              Search 48,000+ factors across 7 method packs, with full
              provenance, signed receipts, and an audit-grade 7-step resolution
              cascade. Every number carries source, version, validity, license
              class, and FQS.
            </p>
          </div>
          <EditionPin onChange={() => undefined} />
        </div>

        <SearchBar
          value={query}
          onChange={setQuery}
          onSubmit={(q) =>
            navigate({ to: "/search", search: { q, offset: 0, limit: 20 } })
          }
        />
      </section>

      <section>
        <h2 className="mb-3 text-lg font-semibold">Coverage</h2>
        <ThreeLabelDashboard />
      </section>

      <section>
        <h2 className="mb-3 text-lg font-semibold">5-minute quickstart</h2>
        <Quickstart lang={lang} onLangChange={setLang} />
      </section>

      <section>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Sources</h2>
          <Button asChild variant="link" size="sm">
            <Link to="/sources">
              Browse all sources <ArrowRight className="h-3 w-3" />
            </Link>
          </Button>
        </div>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-4">
          {topSources.map((s) => (
            <SourceTile key={s.source_id} source={s} />
          ))}
        </div>
      </section>

      <section className="rounded-lg border border-border bg-muted/30 p-6">
        <h2 className="text-lg font-semibold">Build with GreenLang</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Full API reference, SDK clients, and signed-receipt verification guide
          on the developer portal.
        </p>
        <Button className="mt-3" asChild>
          <a
            href="https://developer.greenlang.io/docs/factors"
            target="_blank"
            rel="noopener noreferrer"
          >
            Open developer docs <ArrowRight className="h-4 w-4" />
          </a>
        </Button>
      </section>
    </div>
  );
}

const SNIPPETS: Record<Lang, string> = {
  python: `from greenlang import Factors

f = Factors.resolve(
    activity="natural_gas",
    jurisdiction="GB",
    method="corporate_scope1",
)
print(f.co2e_total_kg, f.why_chosen)`,
  typescript: `import { FactorsClient } from "@greenlang/sdk";

const client = new FactorsClient({ apiKey: process.env.GL_API_KEY });
const f = await client.resolve({
  activity: "natural_gas",
  jurisdiction: "GB",
  method_profile: "corporate_scope1",
});
console.log(f.co2e_total_kg, f.why_chosen);`,
  curl: `curl -s https://api.greenlang.io/v1/factors/resolve/explain \\
  -H "Authorization: Bearer $GL_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "activity": "natural_gas",
    "jurisdiction": "GB",
    "method_profile": "corporate_scope1"
  }'`,
};

function Quickstart({
  lang,
  onLangChange,
}: {
  lang: Lang;
  onLangChange: (l: Lang) => void;
}) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    const ok = await copyToClipboard(SNIPPETS[lang]);
    if (ok) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }
  };
  return (
    <Card>
      <CardContent className="p-0">
        <div className="flex items-center justify-between border-b border-border px-3 py-2">
          <div role="tablist" aria-label="Language" className="flex gap-1">
            {(["python", "typescript", "curl"] as const).map((l) => (
              <button
                key={l}
                role="tab"
                aria-selected={lang === l}
                onClick={() => onLangChange(l)}
                className={`rounded px-2 py-1 text-xs font-medium ${
                  lang === l
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-accent"
                }`}
              >
                {l}
              </button>
            ))}
          </div>
          <Button variant="ghost" size="sm" onClick={handleCopy}>
            {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
            {copied ? "Copied" : "Copy"}
          </Button>
        </div>
        <pre className="overflow-x-auto bg-background p-4 text-xs leading-relaxed">
          <code className="font-mono">{SNIPPETS[lang]}</code>
        </pre>
      </CardContent>
    </Card>
  );
}
