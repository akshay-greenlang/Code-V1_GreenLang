import type {
  AgentLifecycleRecord,
  AppKey,
  PackTierRecord,
  PolicyBundleRecord,
  RunRecord,
  RunResponse
} from "./types";

export async function listRuns(): Promise<RunRecord[]> {
  const response = await fetch("/api/v1/runs");
  if (!response.ok) throw new Error(`Failed to fetch runs: ${response.status}`);
  const payload = (await response.json()) as { runs?: RunRecord[] };
  return payload.runs ?? [];
}

export async function runDemo(app: AppKey): Promise<RunResponse> {
  const response = await fetch(`/api/v1/apps/${app}/demo-run`, { method: "POST" });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Demo failed for ${app}`);
  }
  return (await response.json()) as RunResponse;
}

export async function runApp(app: AppKey, primaryFile?: File, secondaryFile?: File): Promise<RunResponse> {
  if (!primaryFile) {
    return runDemo(app);
  }

  const body = new FormData();
  if (app === "cbam") {
    body.append("config_file", primaryFile);
    if (secondaryFile) {
      body.append("imports_file", secondaryFile);
    } else {
      return runDemo(app);
    }
  } else {
    body.append("input_file", primaryFile);
  }

  const response = await fetch(`/api/v1/apps/${app}/run`, { method: "POST", body });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Run failed for ${app}`);
  }
  return (await response.json()) as RunResponse;
}

export function runBundleUrl(runId: string): string {
  return `/api/v1/runs/${runId}/bundle`;
}

export function runArtifactUrl(runId: string, artifactPath: string): string {
  return `/api/v1/runs/${runId}/artifacts/${encodeURIComponent(artifactPath)}`;
}

export async function fetchArtifactText(runId: string, artifactPath: string): Promise<string> {
  const response = await fetch(runArtifactUrl(runId, artifactPath));
  if (!response.ok) {
    throw new Error(`Failed to fetch artifact ${artifactPath}: ${response.status}`);
  }
  return response.text();
}

export async function listPackTiers(): Promise<PackTierRecord[]> {
  const response = await fetch("/api/v1/governance/pack-tiers");
  if (!response.ok) throw new Error(`Failed to fetch pack tiers: ${response.status}`);
  const payload = (await response.json()) as { packs?: PackTierRecord[] };
  return payload.packs ?? [];
}

export async function listAgents(): Promise<AgentLifecycleRecord[]> {
  const response = await fetch("/api/v1/governance/agents");
  if (!response.ok) throw new Error(`Failed to fetch agents: ${response.status}`);
  const payload = (await response.json()) as { agents?: AgentLifecycleRecord[] };
  return payload.agents ?? [];
}

export async function listPolicyBundles(): Promise<PolicyBundleRecord[]> {
  const response = await fetch("/api/v1/governance/policy-bundles");
  if (!response.ok) throw new Error(`Failed to fetch policy bundles: ${response.status}`);
  const payload = (await response.json()) as { bundles?: PolicyBundleRecord[] };
  return payload.bundles ?? [];
}
