/**
 * TypeScript type definitions for GreenLang SDK
 */

export enum WorkflowStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

export enum AgentCategory {
  CARBON = 'carbon',
  SUSTAINABILITY = 'sustainability',
  CLIMATE = 'climate',
  ESG = 'esg',
  ENERGY = 'energy',
  RESEARCH = 'research',
  ANALYSIS = 'analysis',
}

export enum CitationType {
  WEB = 'web',
  ACADEMIC = 'academic',
  DATABASE = 'database',
  API = 'api',
  INTERNAL = 'internal',
}

export interface Workflow {
  id: string;
  name: string;
  description?: string;
  category?: string;
  agents: Array<{ agent_id: string; config: Record<string, any> }>;
  config: Record<string, any>;
  is_public: boolean;
  created_at: string;
  updated_at: string;
  version: string;
}

export interface WorkflowDefinition {
  name: string;
  description?: string;
  category?: string;
  agents: Array<{ agent_id: string; config: Record<string, any> }>;
  config?: Record<string, any>;
  is_public?: boolean;
}

export interface Agent {
  id: string;
  name: string;
  description: string;
  category: AgentCategory;
  capabilities: string[];
  config_schema: Record<string, any>;
  is_public: boolean;
  version: string;
  created_at: string;
}

export interface ExecutionResult {
  id: string;
  workflow_id?: string;
  agent_id?: string;
  status: WorkflowStatus;
  input_data: Record<string, any>;
  output_data?: Record<string, any>;
  error?: string;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  citations: Citation[];
  metadata: Record<string, any>;
}

export interface Citation {
  id: string;
  execution_id: string;
  source_type: CitationType;
  source_url?: string;
  source_title?: string;
  source_author?: string;
  published_date?: string;
  excerpt?: string;
  relevance_score: number;
  metadata: Record<string, any>;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export interface ListWorkflowsOptions {
  limit?: number;
  offset?: number;
  category?: string;
}

export interface ListAgentsOptions {
  category?: string;
  limit?: number;
  offset?: number;
}

export interface ListExecutionsOptions {
  workflow_id?: string;
  status?: string;
  limit?: number;
  offset?: number;
}

export interface StreamChunk {
  type: 'progress' | 'agent_result' | 'citation' | 'complete' | 'error';
  percentage?: number;
  message?: string;
  agent_name?: string;
  output?: Record<string, any>;
  citation?: Citation;
  result?: Record<string, any>;
  error?: string;
}
