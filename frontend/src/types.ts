export type AppKey = "cbam" | "csrd" | "vcci" | "eudr" | "ghg" | "iso14064";

export interface RunRecord {
  run_id: string;
  app_id?: string;
  status: string;
  success?: boolean;
  execution_mode?: string;
  created_at_ts?: number;
  artifacts?: string[];
  can_export?: boolean;
}

export interface RunResponse {
  run_id: string;
  status: string;
  app_id?: string;
  artifacts: string[];
  can_export?: boolean;
  errors?: string[];
  warnings?: string[];
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
