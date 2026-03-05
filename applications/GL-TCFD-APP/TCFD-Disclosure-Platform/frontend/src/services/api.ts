/**
 * GL-TCFD-APP API Client
 *
 * Comprehensive axios-based API client for the TCFD Disclosure & Scenario Analysis Platform.
 * Organized by domain with full type safety.
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
  GovernanceAssessment,
  GovernanceRole,
  GovernanceCommittee,
  CompetencyEntry,
  ClimateRisk,
  ClimateOpportunity,
  BusinessModelImpact,
  ValueChainNode,
  ScenarioDefinition,
  ScenarioResult,
  ScenarioParameter,
  SensitivityResult,
  StrandingDataPoint,
  PhysicalRiskAssessment,
  AssetLocation,
  InsuranceCostProjection,
  SupplyChainRiskNode,
  TransitionRiskAssessment,
  PolicyRisk,
  TechnologyRisk,
  MarketRisk,
  ReputationRisk,
  FinancialImpact,
  FinancialLineItem,
  MACCDataPoint,
  NPVResult,
  MonteCarloResult,
  RiskManagementRecord,
  RiskIndicator,
  HeatMapCell,
  ClimateMetric,
  ClimateTarget,
  TargetProgress,
  EmissionsSummary,
  PeerBenchmarkData,
  TCFDDisclosure,
  DisclosureSection,
  Evidence,
  ComplianceCheck,
  GapAssessment,
  GapAction,
  ISSBMapping,
  DualComplianceScore,
  MigrationChecklistItem,
  DashboardSummary,
  OrganizationSettings,
  PaginatedResponse,
  ExportFormat,
} from '../types';

const BASE_URL = '/api/v1/tcfd';

const apiClient: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('gl_access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    if (error.response?.status === 401) {
      const refreshToken = localStorage.getItem('gl_refresh_token');
      if (refreshToken) {
        try {
          const resp = await axios.post(`${BASE_URL}/auth/refresh`, { refresh_token: refreshToken });
          localStorage.setItem('gl_access_token', resp.data.access_token);
          if (error.config) {
            error.config.headers.Authorization = `Bearer ${resp.data.access_token}`;
            return apiClient.request(error.config);
          }
        } catch {
          localStorage.removeItem('gl_access_token');
          localStorage.removeItem('gl_refresh_token');
          window.location.href = '/login';
        }
      }
    }
    return Promise.reject(error);
  }
);

// ─── Governance API ───────────────────────────────────────────────────────────

export const governanceApi = {
  getAssessment: (orgId: string) =>
    apiClient.get<GovernanceAssessment>(`/governance/assessment/${orgId}`).then((r) => r.data),

  createAssessment: (data: Partial<GovernanceAssessment>) =>
    apiClient.post<GovernanceAssessment>('/governance/assessment', data).then((r) => r.data),

  updateAssessment: (id: string, data: Partial<GovernanceAssessment>) =>
    apiClient.put<GovernanceAssessment>(`/governance/assessment/${id}`, data).then((r) => r.data),

  getRoles: (orgId: string) =>
    apiClient.get<GovernanceRole[]>(`/governance/roles/${orgId}`).then((r) => r.data),

  createRole: (data: Partial<GovernanceRole>) =>
    apiClient.post<GovernanceRole>('/governance/roles', data).then((r) => r.data),

  updateRole: (id: string, data: Partial<GovernanceRole>) =>
    apiClient.put<GovernanceRole>(`/governance/roles/${id}`, data).then((r) => r.data),

  deleteRole: (id: string) =>
    apiClient.delete(`/governance/roles/${id}`).then((r) => r.data),

  getCommittees: (orgId: string) =>
    apiClient.get<GovernanceCommittee[]>(`/governance/committees/${orgId}`).then((r) => r.data),

  createCommittee: (data: Partial<GovernanceCommittee>) =>
    apiClient.post<GovernanceCommittee>('/governance/committees', data).then((r) => r.data),

  updateCommittee: (id: string, data: Partial<GovernanceCommittee>) =>
    apiClient.put<GovernanceCommittee>(`/governance/committees/${id}`, data).then((r) => r.data),

  getCompetencies: (orgId: string) =>
    apiClient.get<CompetencyEntry[]>(`/governance/competencies/${orgId}`).then((r) => r.data),

  updateCompetency: (id: string, data: Partial<CompetencyEntry>) =>
    apiClient.put<CompetencyEntry>(`/governance/competencies/${id}`, data).then((r) => r.data),
};

// ─── Strategy API ─────────────────────────────────────────────────────────────

export const strategyApi = {
  getRisks: (orgId: string, params?: { category?: string; level?: string; time_horizon?: string }) =>
    apiClient.get<PaginatedResponse<ClimateRisk>>(`/strategy/risks/${orgId}`, { params }).then((r) => r.data),

  createRisk: (data: Partial<ClimateRisk>) =>
    apiClient.post<ClimateRisk>('/strategy/risks', data).then((r) => r.data),

  updateRisk: (id: string, data: Partial<ClimateRisk>) =>
    apiClient.put<ClimateRisk>(`/strategy/risks/${id}`, data).then((r) => r.data),

  deleteRisk: (id: string) =>
    apiClient.delete(`/strategy/risks/${id}`).then((r) => r.data),

  getOpportunities: (orgId: string, params?: { type?: string; status?: string }) =>
    apiClient.get<PaginatedResponse<ClimateOpportunity>>(`/strategy/opportunities/${orgId}`, { params }).then((r) => r.data),

  createOpportunity: (data: Partial<ClimateOpportunity>) =>
    apiClient.post<ClimateOpportunity>('/strategy/opportunities', data).then((r) => r.data),

  updateOpportunity: (id: string, data: Partial<ClimateOpportunity>) =>
    apiClient.put<ClimateOpportunity>(`/strategy/opportunities/${id}`, data).then((r) => r.data),

  deleteOpportunity: (id: string) =>
    apiClient.delete(`/strategy/opportunities/${id}`).then((r) => r.data),

  getBusinessModelImpacts: (orgId: string) =>
    apiClient.get<BusinessModelImpact[]>(`/strategy/business-model/${orgId}`).then((r) => r.data),

  getValueChain: (orgId: string) =>
    apiClient.get<ValueChainNode[]>(`/strategy/value-chain/${orgId}`).then((r) => r.data),

  getRiskSummary: (orgId: string) =>
    apiClient.get<{ total: number; by_level: Record<string, number>; by_category: Record<string, number> }>(
      `/strategy/risks/${orgId}/summary`
    ).then((r) => r.data),

  getOpportunitySummary: (orgId: string) =>
    apiClient.get<{ total: number; by_type: Record<string, number>; total_value: number }>(
      `/strategy/opportunities/${orgId}/summary`
    ).then((r) => r.data),

  assessMateriality: (orgId: string) =>
    apiClient.post(`/strategy/materiality/${orgId}`).then((r) => r.data),
};

// ─── Scenario API ─────────────────────────────────────────────────────────────

export const scenarioApi = {
  getScenarios: (orgId: string) =>
    apiClient.get<ScenarioDefinition[]>(`/scenarios/${orgId}`).then((r) => r.data),

  getScenario: (id: string) =>
    apiClient.get<ScenarioDefinition>(`/scenarios/detail/${id}`).then((r) => r.data),

  createScenario: (data: Partial<ScenarioDefinition>) =>
    apiClient.post<ScenarioDefinition>('/scenarios', data).then((r) => r.data),

  updateScenario: (id: string, data: Partial<ScenarioDefinition>) =>
    apiClient.put<ScenarioDefinition>(`/scenarios/${id}`, data).then((r) => r.data),

  deleteScenario: (id: string) =>
    apiClient.delete(`/scenarios/${id}`).then((r) => r.data),

  runScenario: (id: string, params?: Record<string, number>) =>
    apiClient.post<ScenarioResult[]>(`/scenarios/${id}/run`, { parameters: params }).then((r) => r.data),

  getResults: (scenarioId: string) =>
    apiClient.get<ScenarioResult[]>(`/scenarios/${scenarioId}/results`).then((r) => r.data),

  compareScenarios: (scenarioIds: string[]) =>
    apiClient.post<ScenarioResult[]>('/scenarios/compare', { scenario_ids: scenarioIds }).then((r) => r.data),

  getSensitivity: (scenarioId: string) =>
    apiClient.get<SensitivityResult[]>(`/scenarios/${scenarioId}/sensitivity`).then((r) => r.data),

  getStrandingTimeline: (orgId: string) =>
    apiClient.get<StrandingDataPoint[]>(`/scenarios/stranding/${orgId}`).then((r) => r.data),

  getParameters: (scenarioType: string) =>
    apiClient.get<ScenarioParameter[]>(`/scenarios/parameters/${scenarioType}`).then((r) => r.data),

  duplicateScenario: (id: string) =>
    apiClient.post<ScenarioDefinition>(`/scenarios/${id}/duplicate`).then((r) => r.data),

  getDefaultScenarios: () =>
    apiClient.get<ScenarioDefinition[]>('/scenarios/defaults').then((r) => r.data),

  exportResults: (scenarioId: string, format: ExportFormat) =>
    apiClient.get(`/scenarios/${scenarioId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Physical Risk API ────────────────────────────────────────────────────────

export const physicalRiskApi = {
  getAssessment: (orgId: string) =>
    apiClient.get<PhysicalRiskAssessment>(`/physical-risk/assessment/${orgId}`).then((r) => r.data),

  createAssessment: (data: Partial<PhysicalRiskAssessment>) =>
    apiClient.post<PhysicalRiskAssessment>('/physical-risk/assessment', data).then((r) => r.data),

  getAssets: (orgId: string) =>
    apiClient.get<AssetLocation[]>(`/physical-risk/assets/${orgId}`).then((r) => r.data),

  createAsset: (data: Partial<AssetLocation>) =>
    apiClient.post<AssetLocation>('/physical-risk/assets', data).then((r) => r.data),

  updateAsset: (id: string, data: Partial<AssetLocation>) =>
    apiClient.put<AssetLocation>(`/physical-risk/assets/${id}`, data).then((r) => r.data),

  deleteAsset: (id: string) =>
    apiClient.delete(`/physical-risk/assets/${id}`).then((r) => r.data),

  getHazardAnalysis: (orgId: string) =>
    apiClient.get<{ hazard_type: string; asset_count: number; total_exposure: number }[]>(
      `/physical-risk/hazards/${orgId}`
    ).then((r) => r.data),

  getInsuranceProjections: (orgId: string) =>
    apiClient.get<InsuranceCostProjection[]>(`/physical-risk/insurance/${orgId}`).then((r) => r.data),

  getSupplyChainRisks: (orgId: string) =>
    apiClient.get<SupplyChainRiskNode[]>(`/physical-risk/supply-chain/${orgId}`).then((r) => r.data),

  runClimateProjection: (orgId: string, scenario: string) =>
    apiClient.post(`/physical-risk/project/${orgId}`, { scenario }).then((r) => r.data),

  getAdaptationMeasures: (assetId: string) =>
    apiClient.get<{ measure: string; cost: number; risk_reduction: number }[]>(
      `/physical-risk/assets/${assetId}/adaptations`
    ).then((r) => r.data),

  exportAssessment: (orgId: string, format: ExportFormat) =>
    apiClient.get(`/physical-risk/assessment/${orgId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Transition Risk API ──────────────────────────────────────────────────────

export const transitionRiskApi = {
  getAssessment: (orgId: string) =>
    apiClient.get<TransitionRiskAssessment>(`/transition-risk/assessment/${orgId}`).then((r) => r.data),

  createAssessment: (data: Partial<TransitionRiskAssessment>) =>
    apiClient.post<TransitionRiskAssessment>('/transition-risk/assessment', data).then((r) => r.data),

  getPolicyRisks: (orgId: string) =>
    apiClient.get<PolicyRisk[]>(`/transition-risk/policy/${orgId}`).then((r) => r.data),

  getTechnologyRisks: (orgId: string) =>
    apiClient.get<TechnologyRisk[]>(`/transition-risk/technology/${orgId}`).then((r) => r.data),

  getMarketRisks: (orgId: string) =>
    apiClient.get<MarketRisk[]>(`/transition-risk/market/${orgId}`).then((r) => r.data),

  getReputationRisks: (orgId: string) =>
    apiClient.get<ReputationRisk[]>(`/transition-risk/reputation/${orgId}`).then((r) => r.data),

  getStrandedAssets: (orgId: string) =>
    apiClient.get<{ asset: string; book_value: number; stranded_value: number; timeline: string }[]>(
      `/transition-risk/stranded-assets/${orgId}`
    ).then((r) => r.data),

  getCarbonPriceExposure: (orgId: string) =>
    apiClient.get<{ price: number; annual_cost: number; cumulative_cost: number }[]>(
      `/transition-risk/carbon-price/${orgId}`
    ).then((r) => r.data),

  runTransitionAnalysis: (orgId: string, scenarioId: string) =>
    apiClient.post(`/transition-risk/analyze/${orgId}`, { scenario_id: scenarioId }).then((r) => r.data),

  getComplianceTimeline: (orgId: string) =>
    apiClient.get<{ regulation: string; jurisdiction: string; effective_date: string; status: string }[]>(
      `/transition-risk/compliance-timeline/${orgId}`
    ).then((r) => r.data),

  exportAssessment: (orgId: string, format: ExportFormat) =>
    apiClient.get(`/transition-risk/assessment/${orgId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Opportunity API ──────────────────────────────────────────────────────────

export const opportunityApi = {
  getOpportunities: (orgId: string, params?: { type?: string; status?: string }) =>
    apiClient.get<PaginatedResponse<ClimateOpportunity>>(`/opportunities/${orgId}`, { params }).then((r) => r.data),

  getOpportunity: (id: string) =>
    apiClient.get<ClimateOpportunity>(`/opportunities/detail/${id}`).then((r) => r.data),

  createOpportunity: (data: Partial<ClimateOpportunity>) =>
    apiClient.post<ClimateOpportunity>('/opportunities', data).then((r) => r.data),

  updateOpportunity: (id: string, data: Partial<ClimateOpportunity>) =>
    apiClient.put<ClimateOpportunity>(`/opportunities/${id}`, data).then((r) => r.data),

  deleteOpportunity: (id: string) =>
    apiClient.delete(`/opportunities/${id}`).then((r) => r.data),

  getPipeline: (orgId: string) =>
    apiClient.get<{ stage: string; opportunities: ClimateOpportunity[] }[]>(`/opportunities/pipeline/${orgId}`).then((r) => r.data),

  getROIAnalysis: (orgId: string) =>
    apiClient.get<{ id: string; name: string; investment: number; npv: number; irr: number; payback: number }[]>(
      `/opportunities/roi/${orgId}`
    ).then((r) => r.data),

  getRevenueSizing: (orgId: string) =>
    apiClient.get<{ type: string; low: number; mid: number; high: number }[]>(
      `/opportunities/revenue/${orgId}`
    ).then((r) => r.data),

  getCostSavings: (orgId: string) =>
    apiClient.get<{ category: string; current_cost: number; savings: number; investment: number }[]>(
      `/opportunities/cost-savings/${orgId}`
    ).then((r) => r.data),

  getPriorityMatrix: (orgId: string) =>
    apiClient.get<{ id: string; name: string; impact: number; feasibility: number; size: number; type: string }[]>(
      `/opportunities/priority-matrix/${orgId}`
    ).then((r) => r.data),

  exportOpportunities: (orgId: string, format: ExportFormat) =>
    apiClient.get(`/opportunities/${orgId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Financial API ────────────────────────────────────────────────────────────

export const financialApi = {
  getImpacts: (orgId: string, scenarioId: string) =>
    apiClient.get<FinancialImpact[]>(`/financial/impacts/${orgId}`, { params: { scenario_id: scenarioId } }).then((r) => r.data),

  getIncomeStatement: (orgId: string, scenarioId: string) =>
    apiClient.get<FinancialLineItem[]>(`/financial/income-statement/${orgId}`, { params: { scenario_id: scenarioId } }).then((r) => r.data),

  getBalanceSheet: (orgId: string, scenarioId: string) =>
    apiClient.get<FinancialLineItem[]>(`/financial/balance-sheet/${orgId}`, { params: { scenario_id: scenarioId } }).then((r) => r.data),

  getCashFlow: (orgId: string, scenarioId: string) =>
    apiClient.get<FinancialLineItem[]>(`/financial/cash-flow/${orgId}`, { params: { scenario_id: scenarioId } }).then((r) => r.data),

  getMACCData: (orgId: string) =>
    apiClient.get<MACCDataPoint[]>(`/financial/macc/${orgId}`).then((r) => r.data),

  getNPVAnalysis: (orgId: string, scenarioId: string) =>
    apiClient.get<NPVResult>(`/financial/npv/${orgId}`, { params: { scenario_id: scenarioId } }).then((r) => r.data),

  getCarbonSensitivity: (orgId: string) =>
    apiClient.get<{ carbon_price: number; financial_impact: number; scenario: string }[]>(
      `/financial/carbon-sensitivity/${orgId}`
    ).then((r) => r.data),

  getMonteCarloResults: (orgId: string, scenarioId: string) =>
    apiClient.get<MonteCarloResult>(`/financial/monte-carlo/${orgId}`, { params: { scenario_id: scenarioId } }).then((r) => r.data),

  runFinancialAnalysis: (orgId: string, scenarioId: string) =>
    apiClient.post(`/financial/analyze/${orgId}`, { scenario_id: scenarioId }).then((r) => r.data),

  getFinancialSummary: (orgId: string) =>
    apiClient.get<{ scenario: string; revenue_impact: number; cost_impact: number; net_impact: number }[]>(
      `/financial/summary/${orgId}`
    ).then((r) => r.data),

  exportFinancials: (orgId: string, format: ExportFormat) =>
    apiClient.get(`/financial/${orgId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Risk Management API ──────────────────────────────────────────────────────

export const riskMgmtApi = {
  getRegister: (orgId: string) =>
    apiClient.get<PaginatedResponse<RiskManagementRecord>>(`/risk-management/register/${orgId}`).then((r) => r.data),

  getRecord: (id: string) =>
    apiClient.get<RiskManagementRecord>(`/risk-management/record/${id}`).then((r) => r.data),

  createRecord: (data: Partial<RiskManagementRecord>) =>
    apiClient.post<RiskManagementRecord>('/risk-management/record', data).then((r) => r.data),

  updateRecord: (id: string, data: Partial<RiskManagementRecord>) =>
    apiClient.put<RiskManagementRecord>(`/risk-management/record/${id}`, data).then((r) => r.data),

  deleteRecord: (id: string) =>
    apiClient.delete(`/risk-management/record/${id}`).then((r) => r.data),

  getHeatMap: (orgId: string) =>
    apiClient.get<HeatMapCell[]>(`/risk-management/heat-map/${orgId}`).then((r) => r.data),

  getIndicators: (orgId: string) =>
    apiClient.get<RiskIndicator[]>(`/risk-management/indicators/${orgId}`).then((r) => r.data),

  getERMStatus: (orgId: string) =>
    apiClient.get<{ category: string; integrated: boolean; status: string; last_sync: string }[]>(
      `/risk-management/erm/${orgId}`
    ).then((r) => r.data),

  updateResponseAction: (recordId: string, actionId: string, data: Partial<{ status: string; effectiveness: number }>) =>
    apiClient.put(`/risk-management/record/${recordId}/actions/${actionId}`, data).then((r) => r.data),

  exportRegister: (orgId: string, format: ExportFormat) =>
    apiClient.get(`/risk-management/register/${orgId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Metrics API ──────────────────────────────────────────────────────────────

export const metricsApi = {
  getMetrics: (orgId: string, params?: { category?: string; year?: number }) =>
    apiClient.get<PaginatedResponse<ClimateMetric>>(`/metrics/${orgId}`, { params }).then((r) => r.data),

  getMetric: (id: string) =>
    apiClient.get<ClimateMetric>(`/metrics/detail/${id}`).then((r) => r.data),

  createMetric: (data: Partial<ClimateMetric>) =>
    apiClient.post<ClimateMetric>('/metrics', data).then((r) => r.data),

  updateMetric: (id: string, data: Partial<ClimateMetric>) =>
    apiClient.put<ClimateMetric>(`/metrics/${id}`, data).then((r) => r.data),

  deleteMetric: (id: string) =>
    apiClient.delete(`/metrics/${id}`).then((r) => r.data),

  getTargets: (orgId: string) =>
    apiClient.get<ClimateTarget[]>(`/metrics/targets/${orgId}`).then((r) => r.data),

  createTarget: (data: Partial<ClimateTarget>) =>
    apiClient.post<ClimateTarget>('/metrics/targets', data).then((r) => r.data),

  updateTarget: (id: string, data: Partial<ClimateTarget>) =>
    apiClient.put<ClimateTarget>(`/metrics/targets/${id}`, data).then((r) => r.data),

  getTargetProgress: (targetId: string) =>
    apiClient.get<TargetProgress>(`/metrics/targets/${targetId}/progress`).then((r) => r.data),

  getEmissionsSummary: (orgId: string, year?: number) =>
    apiClient.get<EmissionsSummary>(`/metrics/emissions/${orgId}`, { params: { year } }).then((r) => r.data),

  getIntensityTrend: (orgId: string) =>
    apiClient.get<{ year: number; revenue_intensity: number; employee_intensity: number }[]>(
      `/metrics/intensity-trend/${orgId}`
    ).then((r) => r.data),

  getSBTiAlignment: (orgId: string) =>
    apiClient.get<{ target_id: string; target_name: string; sbti_pathway: number[]; actual: number[]; aligned: boolean }[]>(
      `/metrics/sbti-alignment/${orgId}`
    ).then((r) => r.data),

  getIndustryMetrics: (orgId: string) =>
    apiClient.get<{ metric: string; value: number; unit: string; industry_avg: number; percentile: number }[]>(
      `/metrics/industry/${orgId}`
    ).then((r) => r.data),

  getPeerBenchmark: (orgId: string) =>
    apiClient.get<PeerBenchmarkData[]>(`/metrics/peer-benchmark/${orgId}`).then((r) => r.data),

  exportMetrics: (orgId: string, format: ExportFormat) =>
    apiClient.get(`/metrics/${orgId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),

  recalculateMetrics: (orgId: string) =>
    apiClient.post(`/metrics/${orgId}/recalculate`).then((r) => r.data),
};

// ─── Disclosure API ───────────────────────────────────────────────────────────

export const disclosureApi = {
  getDisclosures: (orgId: string, params?: { year?: number; status?: string }) =>
    apiClient.get<PaginatedResponse<TCFDDisclosure>>(`/disclosure/${orgId}`, { params }).then((r) => r.data),

  getDisclosure: (id: string) =>
    apiClient.get<TCFDDisclosure>(`/disclosure/detail/${id}`).then((r) => r.data),

  createDisclosure: (data: Partial<TCFDDisclosure>) =>
    apiClient.post<TCFDDisclosure>('/disclosure', data).then((r) => r.data),

  updateDisclosure: (id: string, data: Partial<TCFDDisclosure>) =>
    apiClient.put<TCFDDisclosure>(`/disclosure/${id}`, data).then((r) => r.data),

  deleteDisclosure: (id: string) =>
    apiClient.delete(`/disclosure/${id}`).then((r) => r.data),

  getSections: (disclosureId: string) =>
    apiClient.get<DisclosureSection[]>(`/disclosure/${disclosureId}/sections`).then((r) => r.data),

  updateSection: (disclosureId: string, sectionId: string, data: Partial<DisclosureSection>) =>
    apiClient.put<DisclosureSection>(`/disclosure/${disclosureId}/sections/${sectionId}`, data).then((r) => r.data),

  getEvidence: (orgId: string) =>
    apiClient.get<Evidence[]>(`/disclosure/evidence/${orgId}`).then((r) => r.data),

  linkEvidence: (sectionId: string, evidenceId: string) =>
    apiClient.post(`/disclosure/sections/${sectionId}/evidence/${evidenceId}`).then((r) => r.data),

  unlinkEvidence: (sectionId: string, evidenceId: string) =>
    apiClient.delete(`/disclosure/sections/${sectionId}/evidence/${evidenceId}`).then((r) => r.data),

  getComplianceChecks: (disclosureId: string, framework?: string) =>
    apiClient.get<ComplianceCheck[]>(`/disclosure/${disclosureId}/compliance`, { params: { framework } }).then((r) => r.data),

  runComplianceCheck: (disclosureId: string) =>
    apiClient.post<ComplianceCheck[]>(`/disclosure/${disclosureId}/compliance/run`).then((r) => r.data),

  publishDisclosure: (id: string) =>
    apiClient.post<TCFDDisclosure>(`/disclosure/${id}/publish`).then((r) => r.data),

  exportDisclosure: (id: string, format: ExportFormat) =>
    apiClient.get(`/disclosure/${id}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),

  getDisclosureChecklist: (orgId: string) =>
    apiClient.get<{ code: string; title: string; pillar: string; status: string; completeness: number }[]>(
      `/disclosure/checklist/${orgId}`
    ).then((r) => r.data),
};

// ─── Dashboard API ────────────────────────────────────────────────────────────

export const dashboardApi = {
  getSummary: (orgId: string) =>
    apiClient.get<DashboardSummary>(`/dashboard/${orgId}`).then((r) => r.data),

  getRiskExposure: (orgId: string) =>
    apiClient.get<DashboardSummary['risk_exposure']>(`/dashboard/${orgId}/risk-exposure`).then((r) => r.data),

  getOpportunityValue: (orgId: string) =>
    apiClient.get<DashboardSummary['opportunity_value']>(`/dashboard/${orgId}/opportunity-value`).then((r) => r.data),

  getDisclosureMaturity: (orgId: string) =>
    apiClient.get<DashboardSummary['disclosure_maturity']>(`/dashboard/${orgId}/disclosure-maturity`).then((r) => r.data),

  getScenarioSummary: (orgId: string) =>
    apiClient.get<DashboardSummary['scenario_summary']>(`/dashboard/${orgId}/scenario-summary`).then((r) => r.data),

  getEmissionsTrend: (orgId: string) =>
    apiClient.get<{ year: number; scope_1: number; scope_2: number; scope_3: number; total: number }[]>(
      `/dashboard/${orgId}/emissions-trend`
    ).then((r) => r.data),

  getRecentActivity: (orgId: string) =>
    apiClient.get<DashboardSummary['recent_activity']>(`/dashboard/${orgId}/activity`).then((r) => r.data),

  getKeyMetrics: (orgId: string) =>
    apiClient.get<{ name: string; value: number; unit: string; change_pct: number; trend: string }[]>(
      `/dashboard/${orgId}/key-metrics`
    ).then((r) => r.data),
};

// ─── Gap Analysis API ─────────────────────────────────────────────────────────

export const gapApi = {
  getAssessment: (orgId: string) =>
    apiClient.get<GapAssessment>(`/gap-analysis/${orgId}`).then((r) => r.data),

  runAssessment: (orgId: string, framework?: string) =>
    apiClient.post<GapAssessment>(`/gap-analysis/${orgId}/run`, { framework }).then((r) => r.data),

  getActions: (orgId: string) =>
    apiClient.get<GapAction[]>(`/gap-analysis/${orgId}/actions`).then((r) => r.data),

  updateAction: (id: string, data: Partial<GapAction>) =>
    apiClient.put<GapAction>(`/gap-analysis/actions/${id}`, data).then((r) => r.data),

  getPeerComparison: (orgId: string) =>
    apiClient.get<{ pillar: string; org_score: number; peer_avg: number; best_in_class: number }[]>(
      `/gap-analysis/${orgId}/peer-comparison`
    ).then((r) => r.data),

  getMaturityTrend: (orgId: string) =>
    apiClient.get<{ date: string; score: number; maturity: string }[]>(
      `/gap-analysis/${orgId}/maturity-trend`
    ).then((r) => r.data),

  getActionTimeline: (orgId: string) =>
    apiClient.get<GapAction[]>(`/gap-analysis/${orgId}/timeline`).then((r) => r.data),

  exportAssessment: (orgId: string, format: ExportFormat) =>
    apiClient.get(`/gap-analysis/${orgId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── ISSB Cross-Walk API ──────────────────────────────────────────────────────

export const issbApi = {
  getMappings: (orgId: string) =>
    apiClient.get<ISSBMapping[]>(`/issb-crosswalk/${orgId}`).then((r) => r.data),

  getDualScorecard: (orgId: string) =>
    apiClient.get<DualComplianceScore[]>(`/issb-crosswalk/${orgId}/scorecard`).then((r) => r.data),

  getMigrationChecklist: (orgId: string) =>
    apiClient.get<MigrationChecklistItem[]>(`/issb-crosswalk/${orgId}/migration`).then((r) => r.data),

  updateChecklistItem: (id: string, data: Partial<MigrationChecklistItem>) =>
    apiClient.put<MigrationChecklistItem>(`/issb-crosswalk/checklist/${id}`, data).then((r) => r.data),

  runCrossWalk: (orgId: string) =>
    apiClient.post(`/issb-crosswalk/${orgId}/run`).then((r) => r.data),

  exportCrossWalk: (orgId: string, format: ExportFormat) =>
    apiClient.get(`/issb-crosswalk/${orgId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Settings API ─────────────────────────────────────────────────────────────

export const settingsApi = {
  getSettings: (orgId: string) =>
    apiClient.get<OrganizationSettings>(`/settings/${orgId}`).then((r) => r.data),

  updateSettings: (orgId: string, data: Partial<OrganizationSettings>) =>
    apiClient.put<OrganizationSettings>(`/settings/${orgId}`, data).then((r) => r.data),

  getOrganizations: () =>
    apiClient.get<{ id: string; name: string }[]>('/settings/organizations').then((r) => r.data),

  testConnection: () =>
    apiClient.get<{ status: string; latency_ms: number }>('/settings/test-connection').then((r) => r.data),

  getAuditLog: (orgId: string, params?: { page?: number; per_page?: number }) =>
    apiClient.get<PaginatedResponse<{ timestamp: string; user: string; action: string; details: string }>>(
      `/settings/${orgId}/audit-log`, { params }
    ).then((r) => r.data),

  exportSettings: (orgId: string) =>
    apiClient.get(`/settings/${orgId}/export`, { responseType: 'blob' }).then((r) => r.data),
};

export default apiClient;
