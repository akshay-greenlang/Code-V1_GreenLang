import { createFileRoute } from "@tanstack/react-router";
import { ShieldCheck } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export const Route = createFileRoute("/(coming-soon)/verify")({
  component: VerifyPage,
});

function VerifyPage() {
  return (
    <div className="mx-auto max-w-2xl space-y-4">
      <header>
        <h1 className="flex items-center gap-2 text-2xl font-semibold">
          <ShieldCheck className="h-6 w-6 text-factor-certified-700" />
          Verify a signed receipt
        </h1>
        <p className="text-sm text-muted-foreground">
          Every factor payload returned by the API carries a{" "}
          <code className="font-mono">_signed_receipt</code>. This page will
          let you paste a receipt and verify it against the public key at{" "}
          <code className="font-mono">factors.greenlang.io/.well-known/jwks.json</code>.
        </p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Coming soon</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <p>
            Today, verification is available in code. Use the SDKs or verify
            manually:
          </p>
          <pre className="overflow-x-auto rounded-md border border-border bg-muted p-3 text-xs leading-relaxed">
            <code className="font-mono">{EXAMPLE}</code>
          </pre>
          <p>
            An interactive verifier will ship with the Factors v1 GA release.
            In the meantime, see the developer docs for the full signing algorithm
            (Ed25519, SHA-256 content digest, JWKS rotation every 90 days).
          </p>
          <p>
            <a
              className="text-primary underline underline-offset-2"
              href="https://developer.greenlang.io/docs/factors/verify"
              target="_blank"
              rel="noopener noreferrer"
            >
              Read the verification guide →
            </a>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

const EXAMPLE = `from greenlang.factors.verify import verify_receipt

receipt = resp.json()["_signed_receipt"]
ok = verify_receipt(
    payload=resp.json(),
    receipt=receipt,
    jwks_url="https://factors.greenlang.io/.well-known/jwks.json",
)
assert ok, "factor receipt is invalid"`;
