export type AppKey = "cbam" | "csrd" | "vcci" | "eudr" | "ghg" | "iso14064" | "sb253" | "taxonomy";

/** Server-derived lifecycle; UI may also use loading/retrying locally. */
export type RunLifecycleState = "completed" | "failed" | "blocked" | "partial_success";

export interface RunErrorEnvelope {
  title: string;
  message: string;
  details?: string[];
}

export interface RunRecord {
  run_id: string;
  app_id?: string;
  status: string;
  success?: boolean;
  execution_mode?: string;
  created_at_ts?: number;
  artifacts?: string[];
  can_export?: boolean;
  run_state?: RunLifecycleState | string;
  lifecycle_phase?: string;
  status_chip?: string;
  error_envelope?: RunErrorEnvelope | null;
}

export interface RunResponse {
  run_id: string;
  status: string;
  app_id?: string;
  artifacts: string[];
  can_export?: boolean;
  errors?: string[];
  warnings?: string[];
  success?: boolean;
  run_state?: RunLifecycleState | string;
  lifecycle_phase?: string;
  status_chip?: string;
  error_envelope?: RunErrorEnvelope | null;
}

export interface PackTierRecord {
  pack_slug: string;
  app_id: string;
  tier: string;
  owner_team: string;
  promotion_status: string;
}

export interface AgentLifecycleRecord {
  agent_id: string;
  owner_team: string;
  state: string;
  current_version: string;
  replacement_agent_id?: string | null;
}

export interface PolicyBundleRecord {
  bundle: string;
  bytes: number;
}
