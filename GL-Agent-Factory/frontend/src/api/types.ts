/**
 * GreenLang Agent Factory - API Types
 *
 * Type definitions for all API responses and requests.
 */

// ============================================================================
// Common Types
// ============================================================================

export interface PaginatedResponse<T> {
  items: T[];
  pagination: {
    page: number;
    perPage: number;
    totalItems: number;
    totalPages: number;
  };
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export type AgentStatus = 'active' | 'inactive' | 'error' | 'maintenance' | 'deploying';
export type UserRole = 'admin' | 'analyst' | 'viewer' | 'api_user';
export type TenantPlan = 'starter' | 'professional' | 'enterprise';
export type ReportStatus = 'pending' | 'processing' | 'completed' | 'failed';
export type ComplianceStatus = 'compliant' | 'non_compliant' | 'pending_review' | 'requires_action';

// ============================================================================
// Agent Types
// ============================================================================

export interface Agent {
  id: string;
  name: string;
  description: string;
  version: string;
  status: AgentStatus;
  type: 'cbam' | 'eudr' | 'fuel' | 'building' | 'sb253';
  lastDeployedAt: string;
  createdAt: string;
  updatedAt: string;
  metrics: AgentMetrics;
  config: AgentConfig;
}

export interface AgentMetrics {
  requestsToday: number;
  requestsThisMonth: number;
  avgResponseTime: number;
  errorRate: number;
  uptime: number;
}

export interface AgentConfig {
  endpoint: string;
  maxConcurrency: number;
  timeout: number;
  retryAttempts: number;
  rateLimit: number;
  enableCache: boolean;
}

export interface AgentLog {
  id: string;
  agentId: string;
  level: 'debug' | 'info' | 'warn' | 'error';
  message: string;
  metadata?: Record<string, unknown>;
  timestamp: string;
}

// ============================================================================
// User Types
// ============================================================================

export interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  role: UserRole;
  tenantId: string;
  isActive: boolean;
  lastLoginAt: string | null;
  createdAt: string;
  updatedAt: string;
  avatar?: string;
}

export interface UserCreateRequest {
  email: string;
  firstName: string;
  lastName: string;
  role: UserRole;
  password: string;
}

export interface UserUpdateRequest {
  firstName?: string;
  lastName?: string;
  role?: UserRole;
  isActive?: boolean;
}

// ============================================================================
// Tenant Types
// ============================================================================

export interface Tenant {
  id: string;
  name: string;
  slug: string;
  plan: TenantPlan;
  isActive: boolean;
  settings: TenantSettings;
  usage: TenantUsage;
  createdAt: string;
  updatedAt: string;
}

export interface TenantSettings {
  maxUsers: number;
  maxApiCalls: number;
  enabledAgents: string[];
  customDomain?: string;
  ssoEnabled: boolean;
  dataRetentionDays: number;
}

export interface TenantUsage {
  currentUsers: number;
  apiCallsThisMonth: number;
  storageUsedMb: number;
  reportsGenerated: number;
}

// ============================================================================
// Auth Types
// ============================================================================

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginResponse {
  user: User;
  tokens: AuthTokens;
}

export interface RegisterRequest {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  organizationName?: string;
}

// ============================================================================
// Fuel Analysis Types
// ============================================================================

export interface FuelAnalysisRequest {
  fuelType: string;
  quantity: number;
  unit: 'liters' | 'gallons' | 'kg' | 'tonnes';
  period: {
    startDate: string;
    endDate: string;
  };
  scope?: 1 | 2 | 3;
  location?: string;
}

export interface FuelAnalysisResult {
  id: string;
  fuelType: string;
  quantity: number;
  unit: string;
  emissions: {
    co2: number;
    ch4: number;
    n2o: number;
    total: number;
  };
  emissionFactor: number;
  emissionFactorSource: string;
  scope: number;
  calculatedAt: string;
}

// ============================================================================
// CBAM Types
// ============================================================================

export interface CBAMCalculationRequest {
  productCategory: string;
  cnCode: string;
  weight: number;
  originCountry: string;
  supplierData?: {
    name: string;
    installationId?: string;
    specificEmissions?: number;
  };
  importDate: string;
}

export interface CBAMCalculationResult {
  id: string;
  productCategory: string;
  cnCode: string;
  weight: number;
  originCountry: string;
  embeddedEmissions: {
    direct: number;
    indirect: number;
    total: number;
  };
  carbonPrice: {
    euEtsPrice: number;
    originCountryPrice: number;
    cbamAdjustment: number;
  };
  dataQualityScore: number;
  reportingPeriod: string;
  calculatedAt: string;
}

export interface CBAMReport {
  id: string;
  reportingPeriod: string;
  status: ReportStatus;
  totalEmissions: number;
  totalCbamLiability: number;
  itemCount: number;
  submittedAt?: string;
  createdAt: string;
}

// ============================================================================
// Building Energy Types
// ============================================================================

export interface BuildingEnergyRequest {
  buildingType: string;
  floorArea: number;
  location: {
    country: string;
    region?: string;
    climateZone?: string;
  };
  energyConsumption: {
    electricity: number;
    naturalGas?: number;
    heating?: number;
    cooling?: number;
  };
  period: {
    startDate: string;
    endDate: string;
  };
}

export interface BuildingEnergyResult {
  id: string;
  buildingType: string;
  floorArea: number;
  emissions: {
    electricity: number;
    naturalGas: number;
    heating: number;
    cooling: number;
    total: number;
  };
  intensity: {
    perSquareMeter: number;
    perOccupant?: number;
  };
  benchmark: {
    average: number;
    percentile: number;
    rating: 'A' | 'B' | 'C' | 'D' | 'E' | 'F' | 'G';
  };
  recommendations: {
    category: string;
    description: string;
    potentialSavings: number;
  }[];
  calculatedAt: string;
}

// ============================================================================
// EUDR Types
// ============================================================================

export interface EUDRComplianceRequest {
  commodity: string;
  originCountry: string;
  productionDate: string;
  geoLocation: {
    latitude: number;
    longitude: number;
    polygon?: Array<{ lat: number; lng: number }>;
  };
  supplierChain: {
    name: string;
    role: string;
    country: string;
  }[];
  quantity: number;
  unit: string;
}

export interface EUDRComplianceResult {
  id: string;
  commodity: string;
  originCountry: string;
  complianceStatus: ComplianceStatus;
  riskAssessment: {
    deforestationRisk: 'low' | 'medium' | 'high';
    legalityRisk: 'low' | 'medium' | 'high';
    overallRisk: 'low' | 'medium' | 'high';
    score: number;
  };
  satelliteAnalysis?: {
    forestCoverChange: number;
    deforestationDetected: boolean;
    analysisDate: string;
    confidence: number;
  };
  dueDiligenceChecklist: {
    item: string;
    status: 'passed' | 'failed' | 'pending';
    notes?: string;
  }[];
  documentationRequired: string[];
  calculatedAt: string;
}

// ============================================================================
// Report Types
// ============================================================================

export interface Report {
  id: string;
  type: 'fuel' | 'cbam' | 'building' | 'eudr' | 'comprehensive';
  name: string;
  status: ReportStatus;
  format: 'pdf' | 'excel' | 'json' | 'xml';
  fileSize?: number;
  downloadUrl?: string;
  parameters: Record<string, unknown>;
  createdAt: string;
  completedAt?: string;
}

export interface ReportGenerateRequest {
  type: Report['type'];
  name: string;
  format: Report['format'];
  parameters: Record<string, unknown>;
  dateRange?: {
    startDate: string;
    endDate: string;
  };
}

// ============================================================================
// Dashboard Types
// ============================================================================

export interface DashboardMetrics {
  totalEmissions: number;
  emissionsTrend: number;
  totalCalculations: number;
  activeAgents: number;
  dataQualityScore: number;
  complianceRate: number;
  recentActivity: ActivityItem[];
  emissionsByCategory: {
    category: string;
    value: number;
    percentage: number;
  }[];
  emissionsTrendData: {
    date: string;
    emissions: number;
  }[];
  topCountries: {
    country: string;
    emissions: number;
    imports: number;
  }[];
}

export interface ActivityItem {
  id: string;
  type: 'calculation' | 'report' | 'alert' | 'user_action';
  title: string;
  description: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export interface SystemAlert {
  id: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  title: string;
  message: string;
  agentId?: string;
  acknowledgedAt?: string;
  createdAt: string;
}
