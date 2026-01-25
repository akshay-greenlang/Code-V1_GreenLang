/**
 * GreenLang Agent Factory - Marketplace Types
 *
 * Type definitions for the Agent Marketplace and Discovery system.
 */

// ============================================================================
// Agent Marketplace Types
// ============================================================================

export type AgentCategory =
  | 'emissions'
  | 'compliance'
  | 'reporting'
  | 'industry'
  | 'analytics'
  | 'integration';

export type AgentPricingTier = 'free' | 'starter' | 'professional' | 'enterprise';

export type AgentDeploymentStatus =
  | 'available'
  | 'deployed'
  | 'deploying'
  | 'updating'
  | 'deprecated';

export type AgentComplexity = 'simple' | 'moderate' | 'complex';

export type SortOption =
  | 'popularity'
  | 'rating'
  | 'newest'
  | 'regulatory_deadline'
  | 'name_asc'
  | 'name_desc';

// ============================================================================
// Marketplace Agent Types
// ============================================================================

export interface MarketplaceAgent {
  id: string;
  name: string;
  slug: string;
  shortDescription: string;
  description: string;
  category: AgentCategory;
  subcategory?: string;
  icon: string;
  iconColor: string;
  version: string;
  author: AgentAuthor;
  pricing: AgentPricing;
  metrics: AgentMarketplaceMetrics;
  features: AgentFeature[];
  useCases: AgentUseCase[];
  integrations: string[];
  regulatoryFrameworks: string[];
  deploymentStatus: AgentDeploymentStatus;
  complexity: AgentComplexity;
  estimatedSetupTime: string;
  documentation: AgentDocumentation;
  releaseNotes: string;
  tags: string[];
  isVerified: boolean;
  isFeatured: boolean;
  isNew: boolean;
  createdAt: string;
  updatedAt: string;
  lastDeployedAt?: string;
}

export interface AgentAuthor {
  id: string;
  name: string;
  organization: string;
  avatar?: string;
  isVerified: boolean;
}

export interface AgentPricing {
  tier: AgentPricingTier;
  basePrice: number;
  currency: string;
  billingPeriod: 'monthly' | 'yearly' | 'usage';
  usageBasedPricing?: {
    unit: string;
    pricePerUnit: number;
    includedUnits: number;
  };
  tiers: PricingTierDetail[];
  trialDays?: number;
  customPricingAvailable: boolean;
}

export interface PricingTierDetail {
  name: AgentPricingTier;
  price: number;
  features: string[];
  limits: {
    apiCalls: number;
    dataRetentionDays: number;
    supportLevel: 'community' | 'standard' | 'priority' | 'dedicated';
    maxUsers?: number;
  };
  recommended?: boolean;
}

export interface AgentMarketplaceMetrics {
  totalDeployments: number;
  activeDeployments: number;
  averageRating: number;
  totalReviews: number;
  accuracy: number;
  avgResponseTime: number;
  uptime: number;
  lastMonthDeployments: number;
  satisfactionScore: number;
}

export interface AgentFeature {
  id: string;
  name: string;
  description: string;
  category: string;
  isIncludedInFree: boolean;
  requiredTier: AgentPricingTier;
}

export interface AgentUseCase {
  id: string;
  title: string;
  description: string;
  industry: string;
  benefits: string[];
  estimatedROI?: string;
}

export interface AgentDocumentation {
  quickStartUrl: string;
  apiReferenceUrl: string;
  tutorialsUrl: string;
  changelogUrl: string;
  supportUrl: string;
}

// ============================================================================
// Agent Version Types
// ============================================================================

export interface AgentVersion {
  id: string;
  agentId: string;
  version: string;
  releaseDate: string;
  releaseNotes: string;
  isLatest: boolean;
  isStable: boolean;
  isBeta: boolean;
  breakingChanges: boolean;
  changes: VersionChange[];
  downloadCount: number;
  minPlatformVersion: string;
}

export interface VersionChange {
  type: 'feature' | 'improvement' | 'bugfix' | 'security' | 'deprecation';
  description: string;
}

// ============================================================================
// Review Types
// ============================================================================

export interface AgentReview {
  id: string;
  agentId: string;
  userId: string;
  userName: string;
  userAvatar?: string;
  userOrganization?: string;
  rating: number;
  title: string;
  content: string;
  pros: string[];
  cons: string[];
  useCaseDescription?: string;
  verifiedPurchase: boolean;
  helpfulCount: number;
  response?: ReviewResponse;
  createdAt: string;
  updatedAt: string;
}

export interface ReviewResponse {
  authorId: string;
  authorName: string;
  content: string;
  createdAt: string;
}

export interface ReviewSummary {
  totalReviews: number;
  averageRating: number;
  ratingDistribution: {
    1: number;
    2: number;
    3: number;
    4: number;
    5: number;
  };
  topPros: string[];
  topCons: string[];
}

// ============================================================================
// Deployment Types
// ============================================================================

export interface AgentDeployment {
  id: string;
  agentId: string;
  tenantId: string;
  version: string;
  status: AgentDeploymentStatus;
  configuration: AgentDeploymentConfig;
  metrics: DeploymentMetrics;
  createdAt: string;
  updatedAt: string;
}

export interface AgentDeploymentConfig {
  endpoint: string;
  apiKey: string;
  maxConcurrency: number;
  timeout: number;
  retryAttempts: number;
  enableCache: boolean;
  webhookUrl?: string;
  customSettings: Record<string, unknown>;
}

export interface DeploymentMetrics {
  requestsToday: number;
  requestsThisMonth: number;
  avgResponseTime: number;
  errorRate: number;
  uptime: number;
  lastRequestAt?: string;
}

// ============================================================================
// Workflow Types
// ============================================================================

export interface Workflow {
  id: string;
  name: string;
  description: string;
  tenantId: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  status: 'draft' | 'active' | 'paused' | 'archived';
  triggerType: 'manual' | 'scheduled' | 'webhook' | 'event';
  triggerConfig?: WorkflowTriggerConfig;
  createdAt: string;
  updatedAt: string;
}

export interface WorkflowNode {
  id: string;
  type: 'agent' | 'condition' | 'transform' | 'output' | 'input';
  agentId?: string;
  label: string;
  position: { x: number; y: number };
  config: WorkflowNodeConfig;
  inputSchema?: Record<string, unknown>;
  outputSchema?: Record<string, unknown>;
}

export interface WorkflowNodeConfig {
  parameters: Record<string, unknown>;
  retryOnFailure: boolean;
  timeout: number;
  condition?: WorkflowCondition;
  transformScript?: string;
}

export interface WorkflowCondition {
  field: string;
  operator: 'eq' | 'neq' | 'gt' | 'gte' | 'lt' | 'lte' | 'contains' | 'not_contains';
  value: unknown;
}

export interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
  label?: string;
  conditionBranch?: 'true' | 'false';
}

export interface WorkflowTriggerConfig {
  schedule?: string;
  webhookSecret?: string;
  eventType?: string;
  eventFilter?: Record<string, unknown>;
}

export interface WorkflowExecution {
  id: string;
  workflowId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  startedAt: string;
  completedAt?: string;
  duration?: number;
  nodeExecutions: NodeExecution[];
  input: Record<string, unknown>;
  output?: Record<string, unknown>;
  error?: string;
}

export interface NodeExecution {
  nodeId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  startedAt?: string;
  completedAt?: string;
  duration?: number;
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
  error?: string;
}

// ============================================================================
// Request/Response Types
// ============================================================================

export interface AgentListParams {
  page?: number;
  perPage?: number;
  category?: AgentCategory;
  search?: string;
  sortBy?: SortOption;
  tags?: string[];
  pricingTier?: AgentPricingTier;
  isVerified?: boolean;
  isFeatured?: boolean;
  regulatoryFramework?: string;
}

export interface AgentSearchResult {
  agents: MarketplaceAgent[];
  pagination: {
    page: number;
    perPage: number;
    totalItems: number;
    totalPages: number;
  };
  facets: {
    categories: { category: AgentCategory; count: number }[];
    tags: { tag: string; count: number }[];
    pricingTiers: { tier: AgentPricingTier; count: number }[];
  };
}

export interface DeployAgentRequest {
  agentId: string;
  version?: string;
  configuration?: Partial<AgentDeploymentConfig>;
  pricingTier?: AgentPricingTier;
}

export interface DeployAgentResponse {
  deployment: AgentDeployment;
  apiKey: string;
  endpoint: string;
  documentation: AgentDocumentation;
}

export interface CreateReviewRequest {
  agentId: string;
  rating: number;
  title: string;
  content: string;
  pros?: string[];
  cons?: string[];
  useCaseDescription?: string;
}

export interface WorkflowCreateRequest {
  name: string;
  description: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  triggerType: Workflow['triggerType'];
  triggerConfig?: WorkflowTriggerConfig;
}

export interface WorkflowTestRequest {
  workflowId: string;
  testInput: Record<string, unknown>;
}

export interface WorkflowTestResult {
  success: boolean;
  nodeResults: {
    nodeId: string;
    output?: Record<string, unknown>;
    error?: string;
    duration: number;
  }[];
  totalDuration: number;
  errors: string[];
}

// ============================================================================
// Comparison Types
// ============================================================================

export interface AgentComparison {
  agents: MarketplaceAgent[];
  featureMatrix: FeatureComparisonMatrix;
  performanceComparison: PerformanceComparison;
  pricingComparison: PricingComparison;
  useCaseFit: UseCaseFitScore[];
}

export interface FeatureComparisonMatrix {
  categories: string[];
  features: {
    name: string;
    category: string;
    agentSupport: { [agentId: string]: boolean | string };
  }[];
}

export interface PerformanceComparison {
  metrics: {
    name: string;
    unit: string;
    values: { [agentId: string]: number };
    higherIsBetter: boolean;
  }[];
}

export interface PricingComparison {
  tiers: {
    tierName: string;
    prices: { [agentId: string]: number };
    features: { [agentId: string]: string[] };
  }[];
}

export interface UseCaseFitScore {
  useCase: string;
  scores: { [agentId: string]: number };
  recommendation: string;
}

// ============================================================================
// Marketplace Store State Types
// ============================================================================

export interface MarketplaceFilters {
  category?: AgentCategory;
  search: string;
  sortBy: SortOption;
  tags: string[];
  pricingTier?: AgentPricingTier;
  isVerified?: boolean;
  regulatoryFramework?: string;
}

export interface MarketplaceState {
  filters: MarketplaceFilters;
  viewMode: 'grid' | 'list';
  selectedAgentsForComparison: string[];
  recentlyViewed: string[];
  favorites: string[];
}
