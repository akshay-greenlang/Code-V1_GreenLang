import type {
  AgentLifecycleRecord,
  AppKey,
  PackTierRecord,
  PolicyBundleRecord,
  RunRecord,
  RunResponse
} from "./types";

export interface ListRunsQuery {
  app_id?: string;
  status?: string;
  since_ts?: number;
  until_ts?: number;
  q?: string;
}

export async function listRuns(query?: ListRunsQuery): Promise<RunRecord[]> {
  const params = new URLSearchParams();
  if (query?.app_id) params.set("app_id", query.app_id);
  if (query?.status) params.set("status", query.status);
  if (query?.since_ts != null) params.set("since_ts", String(query.since_ts));
  if (query?.until_ts != null) params.set("until_ts", String(query.until_ts));
  if (query?.q) params.set("q", query.q);
  const qs = params.toString();
  const response = await fetch(qs ? `/api/v1/runs?${qs}` : "/api/v1/runs");
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

export interface ReleaseTrainCycleSummary {
  cycle?: string;
  all_passed?: boolean;
  executed_at_utc?: string;
}

export interface ReleaseTrainPayload {
  available: boolean;
  evidence?: unknown;
  cycle_summary?: ReleaseTrainCycleSummary[];
}

export async function fetchReleaseTrainEvidence(): Promise<ReleaseTrainPayload> {
  const response = await fetch("/api/v1/admin/release-train");
  if (!response.ok) throw new Error(`Failed to fetch release train: ${response.status}`);
  return (await response.json()) as ReleaseTrainPayload;
}

export interface ConnectorAdminRow {
  connector_id: string;
  app_id: string;
  owner_team: string;
  support_channel: string;
  read_timeout_ms?: number;
  circuit_open_s?: number;
  operational_status?: string;
  incident_summary?: string | null;
  slo_target_availability_pct?: number | null;
  runbook_url?: string | null;
}

export interface ShellChromeContext {
  compliance_rail: {
    managed_pack_count: number;
    policy_bundle_count: number;
    deprecated_agent_count: number;
  };
  connector_incidents: Array<{
    connector_id: string;
    app_id: string;
    severity: string;
    message: string;
  }>;
  connector_probe_meta?: {
    probe_count?: number;
    last_refresh_utc?: string | null;
  };
}

export interface ConnectorRegistryPayload {
  registry_version: string;
  connectors: ConnectorAdminRow[];
}

export async function fetchConnectorRegistry(): Promise<ConnectorRegistryPayload> {
  const response = await fetch("/api/v1/admin/connectors");
  if (!response.ok) throw new Error(`Failed to fetch connectors: ${response.status}`);
  return (await response.json()) as ConnectorRegistryPayload;
}

export interface ConnectorHealthProbe {
  connector_id: string;
  app_id: string;
  ok: boolean;
  latency_ms: number;
  checked_at_utc: string;
  registry_status?: string;
}

export interface ConnectorHealthPayload {
  updated_at_utc: string;
  probes: ConnectorHealthProbe[];
}

export async function fetchConnectorHealth(): Promise<ConnectorHealthPayload> {
  const response = await fetch("/api/v1/admin/connectors/health");
  if (!response.ok) throw new Error(`Failed to fetch connector health: ${response.status}`);
  return (await response.json()) as ConnectorHealthPayload;
}

export async function fetchHealth(): Promise<{ status: string; version: string }> {
  const response = await fetch("/health");
  if (!response.ok) throw new Error(`Health check failed: ${response.status}`);
  return (await response.json()) as { status: string; version: string };
}

export async function fetchShellChromeContext(): Promise<ShellChromeContext> {
  const response = await fetch("/api/v1/shell/chrome-context");
  if (!response.ok) throw new Error(`Shell chrome context failed: ${response.status}`);
  return (await response.json()) as ShellChromeContext;
}
