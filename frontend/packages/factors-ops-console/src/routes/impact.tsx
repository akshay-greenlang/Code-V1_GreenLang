import { createFileRoute } from "@tanstack/react-router";
import { AuthGuard } from "@/components/AuthGuard";
import { ImpactSimulator } from "@/components/ImpactSimulator";
import { runImpactSimulation } from "@/lib/api";
import type { Identity } from "@/lib/auth";

export const Route = createFileRoute("/impact")({
  component: ImpactPage,
});

function ImpactPage() {
  return (
    <AuthGuard requiredAction="impact.run">
      {(identity) => <ImpactPanel identity={identity} />}
    </AuthGuard>
  );
}

function ImpactPanel({ identity }: { identity: Identity }) {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-semibold">Impact Simulator</h1>
        <p className="text-sm text-muted-foreground">
          What breaks if I replace factor pack X? Preview affected computations + affected tenants.
        </p>
      </header>
      <ImpactSimulator
        onRun={(payload) =>
          runImpactSimulation(identity, {
            factor_id: payload.factor_id,
            mode: payload.mode,
            hypothetical_value: payload.hypothetical_value,
            tenant_scope: payload.tenant_scope ?? null,
            reason: payload.reason,
          })
        }
      />
    </div>
  );
}
