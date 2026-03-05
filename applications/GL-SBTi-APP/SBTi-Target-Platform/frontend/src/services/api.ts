/**
 * GL-SBTi-APP API Client
 *
 * Comprehensive axios-based API client for the SBTi Target Validation &
 * Progress Tracking Platform. Organized by domain with full type safety.
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
  Target,
  Pathway,
  PathwayComparison,
  ValidationResult,
  Scope3Screening,
  FLAGAssessment,
  SectorPathway,
  SectorBenchmark,
  ProgressRecord,
  ProgressSummary,
  VarianceAnalysis,
  TemperatureScore,
  PortfolioTemperature,
  TemperatureTimeSeries,
  PeerTemperatureRanking,
  Recalculation,
  ThresholdCheck,
  FiveYearReview,
  FIPortfolio,
  PortfolioHolding,
  EngagementRecord,
  FrameworkMapping,
  AlignmentItem,
  Report,
  SubmissionForm,
  DashboardSummary,
  GapAssessment,
  GapAction,
  OrganizationSettings,
  PaginatedResponse,
  ExportFormat,
  EmissionsInventory,
} from '../types';

const BASE_URL = '/api/v1/sbti';

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

// ─── Target API ──────────────────────────────────────────────────────────────

export const targetApi = {
  getTargets: (orgId: string, params?: { status?: string; scope?: string; timeframe?: string }) =>
    apiClient.get<PaginatedResponse<Target>>(`/targets/${orgId}`, { params }).then((r) => r.data),

  getTarget: (id: string) =>
    apiClient.get<Target>(`/targets/detail/${id}`).then((r) => r.data),

  createTarget: (data: Partial<Target>) =>
    apiClient.post<Target>('/targets', data).then((r) => r.data),

  updateTarget: (id: string, data: Partial<Target>) =>
    apiClient.put<Target>(`/targets/${id}`, data).then((r) => r.data),

  deleteTarget: (id: string) =>
    apiClient.delete(`/targets/${id}`).then((r) => r.data),

  getTargetStatus: (id: string) =>
    apiClient.get<{ status: string; timeline: { date: string; status: string; notes: string }[] }>(
      `/targets/${id}/status`
    ).then((r) => r.data),

  submitTarget: (id: string) =>
    apiClient.post<Target>(`/targets/${id}/submit`).then((r) => r.data),

  withdrawTarget: (id: string) =>
    apiClient.post<Target>(`/targets/${id}/withdraw`).then((r) => r.data),

  duplicateTarget: (id: string) =>
    apiClient.post<Target>(`/targets/${id}/duplicate`).then((r) => r.data),

  exportTarget: (id: string, format: ExportFormat) =>
    apiClient.get(`/targets/${id}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Pathway API ─────────────────────────────────────────────────────────────

export const pathwayApi = {
  getPathway: (targetId: string) =>
    apiClient.get<Pathway>(`/pathways/${targetId}`).then((r) => r.data),

  calculateACA: (params: { base_year: number; target_year: number; base_emissions: number; alignment: string }) =>
    apiClient.post<Pathway>('/pathways/aca', params).then((r) => r.data),

  calculateSDA: (params: { sector: string; base_year: number; target_year: number; base_intensity: number; alignment: string }) =>
    apiClient.post<Pathway>('/pathways/sda', params).then((r) => r.data),

  comparePathways: (targetId: string) =>
    apiClient.get<PathwayComparison[]>(`/pathways/${targetId}/compare`).then((r) => r.data),

  getAvailablePathways: (sector: string) =>
    apiClient.get<SectorPathway[]>(`/pathways/available/${sector}`).then((r) => r.data),

  recalculatePathway: (targetId: string) =>
    apiClient.post<Pathway>(`/pathways/${targetId}/recalculate`).then((r) => r.data),

  exportPathway: (targetId: string, format: ExportFormat) =>
    apiClient.get(`/pathways/${targetId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Validation API ──────────────────────────────────────────────────────────

export const validationApi = {
  validate: (targetId: string) =>
    apiClient.post<ValidationResult>(`/validation/${targetId}/run`).then((r) => r.data),

  getResult: (targetId: string) =>
    apiClient.get<ValidationResult>(`/validation/${targetId}`).then((r) => r.data),

  getChecklist: (orgId: string) =>
    apiClient.get<{ criterion_code: string; criterion_name: string; category: string; status: string }[]>(
      `/validation/checklist/${orgId}`
    ).then((r) => r.data),

  getReadiness: (orgId: string) =>
    apiClient.get<{ readiness_score: number; category_scores: { category: string; score: number }[]; blockers: string[] }>(
      `/validation/readiness/${orgId}`
    ).then((r) => r.data),

  getHistory: (targetId: string) =>
    apiClient.get<ValidationResult[]>(`/validation/${targetId}/history`).then((r) => r.data),

  exportValidation: (targetId: string, format: ExportFormat) =>
    apiClient.get(`/validation/${targetId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Scope 3 API ─────────────────────────────────────────────────────────────

export const scope3Api = {
  getTriggerAssessment: (orgId: string) =>
    apiClient.get<Scope3Screening>(`/scope3/trigger/${orgId}`).then((r) => r.data),

  runTriggerAssessment: (orgId: string) =>
    apiClient.post<Scope3Screening>(`/scope3/trigger/${orgId}/run`).then((r) => r.data),

  getCategoryBreakdown: (orgId: string) =>
    apiClient.get<Scope3Screening['category_breakdown']>(`/scope3/breakdown/${orgId}`).then((r) => r.data),

  getCoverage: (orgId: string) =>
    apiClient.get<{ total_pct: number; included_categories: number[]; excluded_categories: number[]; two_thirds_met: boolean }>(
      `/scope3/coverage/${orgId}`
    ).then((r) => r.data),

  updateCategoryInclusion: (orgId: string, categoryNumber: number, included: boolean) =>
    apiClient.put(`/scope3/categories/${orgId}/${categoryNumber}`, { included }).then((r) => r.data),

  getHotspots: (orgId: string) =>
    apiClient.get<{ category_number: number; category_name: string; emissions: number; significance: string; hotspot_rank: number }[]>(
      `/scope3/hotspots/${orgId}`
    ).then((r) => r.data),

  exportScreening: (orgId: string, format: ExportFormat) =>
    apiClient.get(`/scope3/screening/${orgId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── FLAG API ────────────────────────────────────────────────────────────────

export const flagApi = {
  getTriggerAssessment: (orgId: string) =>
    apiClient.get<FLAGAssessment>(`/flag/trigger/${orgId}`).then((r) => r.data),

  runTriggerAssessment: (orgId: string) =>
    apiClient.post<FLAGAssessment>(`/flag/trigger/${orgId}/run`).then((r) => r.data),

  getCommodityData: (orgId: string) =>
    apiClient.get<FLAGAssessment['commodities']>(`/flag/commodities/${orgId}`).then((r) => r.data),

  getSectorPathways: (commodity: string) =>
    apiClient.get<SectorPathway[]>(`/flag/pathways/${commodity}`).then((r) => r.data),

  updateDeforestationCommitment: (orgId: string, data: { commitment: boolean; zero_by_year: number }) =>
    apiClient.put(`/flag/deforestation/${orgId}`, data).then((r) => r.data),

  getEmissionsSplit: (orgId: string) =>
    apiClient.get<{ flag_emissions: number; non_flag_emissions: number; flag_pct: number; by_commodity: { commodity: string; emissions: number }[] }>(
      `/flag/split/${orgId}`
    ).then((r) => r.data),

  exportAssessment: (orgId: string, format: ExportFormat) =>
    apiClient.get(`/flag/assessment/${orgId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Sector API ──────────────────────────────────────────────────────────────

export const sectorApi = {
  getSectors: () =>
    apiClient.get<SectorPathway[]>('/sectors').then((r) => r.data),

  getSectorPathway: (sector: string, alignment: string) =>
    apiClient.get<SectorPathway>(`/sectors/${sector}/pathway`, { params: { alignment } }).then((r) => r.data),

  getBenchmarks: (orgId: string) =>
    apiClient.get<SectorBenchmark[]>(`/sectors/benchmarks/${orgId}`).then((r) => r.data),

  detectSector: (orgId: string) =>
    apiClient.post<{ detected_sector: string; confidence: number; isic_code: string }>(
      `/sectors/detect/${orgId}`
    ).then((r) => r.data),

  compareSectors: (sectors: string[]) =>
    apiClient.post<SectorPathway[]>('/sectors/compare', { sectors }).then((r) => r.data),
};

// ─── Progress API ────────────────────────────────────────────────────────────

export const progressApi = {
  recordProgress: (data: Partial<ProgressRecord>) =>
    apiClient.post<ProgressRecord>('/progress', data).then((r) => r.data),

  getHistory: (targetId: string) =>
    apiClient.get<ProgressRecord[]>(`/progress/${targetId}/history`).then((r) => r.data),

  getSummary: (targetId: string) =>
    apiClient.get<ProgressSummary>(`/progress/${targetId}/summary`).then((r) => r.data),

  getVarianceAnalysis: (targetId: string, year: number) =>
    apiClient.get<VarianceAnalysis>(`/progress/${targetId}/variance`, { params: { year } }).then((r) => r.data),

  getDashboard: (orgId: string) =>
    apiClient.get<ProgressSummary[]>(`/progress/dashboard/${orgId}`).then((r) => r.data),

  getProjection: (targetId: string) =>
    apiClient.get<{ year: number; projected_emissions: number; target_emissions: number }[]>(
      `/progress/${targetId}/projection`
    ).then((r) => r.data),

  updateRecord: (id: string, data: Partial<ProgressRecord>) =>
    apiClient.put<ProgressRecord>(`/progress/${id}`, data).then((r) => r.data),

  exportProgress: (targetId: string, format: ExportFormat) =>
    apiClient.get(`/progress/${targetId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Temperature API ─────────────────────────────────────────────────────────

export const temperatureApi = {
  getScore: (orgId: string) =>
    apiClient.get<TemperatureScore>(`/temperature/${orgId}`).then((r) => r.data),

  getScoreByTarget: (targetId: string) =>
    apiClient.get<TemperatureScore>(`/temperature/target/${targetId}`).then((r) => r.data),

  getTimeSeries: (orgId: string) =>
    apiClient.get<TemperatureTimeSeries[]>(`/temperature/${orgId}/time-series`).then((r) => r.data),

  getPeerRanking: (orgId: string) =>
    apiClient.get<PeerTemperatureRanking[]>(`/temperature/${orgId}/peer-ranking`).then((r) => r.data),

  getPortfolioTemperature: (portfolioId: string) =>
    apiClient.get<PortfolioTemperature>(`/temperature/portfolio/${portfolioId}`).then((r) => r.data),

  recalculate: (orgId: string) =>
    apiClient.post<TemperatureScore>(`/temperature/${orgId}/recalculate`).then((r) => r.data),

  exportScore: (orgId: string, format: ExportFormat) =>
    apiClient.get(`/temperature/${orgId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Recalculation API ───────────────────────────────────────────────────────

export const recalculationApi = {
  checkThresholds: (orgId: string) =>
    apiClient.get<ThresholdCheck[]>(`/recalculation/thresholds/${orgId}`).then((r) => r.data),

  getRecalculations: (orgId: string) =>
    apiClient.get<Recalculation[]>(`/recalculation/${orgId}`).then((r) => r.data),

  createRecalculation: (data: Partial<Recalculation>) =>
    apiClient.post<Recalculation>('/recalculation', data).then((r) => r.data),

  updateRecalculation: (id: string, data: Partial<Recalculation>) =>
    apiClient.put<Recalculation>(`/recalculation/${id}`, data).then((r) => r.data),

  approveRecalculation: (id: string) =>
    apiClient.post<Recalculation>(`/recalculation/${id}/approve`).then((r) => r.data),

  getAuditTrail: (id: string) =>
    apiClient.get<Recalculation['audit_trail']>(`/recalculation/${id}/audit`).then((r) => r.data),

  exportRecalculation: (id: string, format: ExportFormat) =>
    apiClient.get(`/recalculation/${id}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Review API ──────────────────────────────────────────────────────────────

export const reviewApi = {
  getReview: (orgId: string) =>
    apiClient.get<FiveYearReview>(`/review/${orgId}`).then((r) => r.data),

  createReview: (orgId: string) =>
    apiClient.post<FiveYearReview>(`/review/${orgId}`).then((r) => r.data),

  updateReview: (id: string, data: Partial<FiveYearReview>) =>
    apiClient.put<FiveYearReview>(`/review/${id}`, data).then((r) => r.data),

  getReadiness: (orgId: string) =>
    apiClient.get<{ readiness_pct: number; items_completed: number; items_total: number; blockers: string[] }>(
      `/review/readiness/${orgId}`
    ).then((r) => r.data),

  submitOutcome: (id: string, outcome: string) =>
    apiClient.post<FiveYearReview>(`/review/${id}/outcome`, { outcome }).then((r) => r.data),

  getHistory: (orgId: string) =>
    apiClient.get<FiveYearReview[]>(`/review/${orgId}/history`).then((r) => r.data),

  exportReview: (id: string, format: ExportFormat) =>
    apiClient.get(`/review/${id}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Financial Institutions API ──────────────────────────────────────────────

export const fiApi = {
  getPortfolios: (orgId: string) =>
    apiClient.get<FIPortfolio[]>(`/fi/portfolios/${orgId}`).then((r) => r.data),

  getPortfolio: (id: string) =>
    apiClient.get<FIPortfolio>(`/fi/portfolios/detail/${id}`).then((r) => r.data),

  createPortfolio: (data: Partial<FIPortfolio>) =>
    apiClient.post<FIPortfolio>('/fi/portfolios', data).then((r) => r.data),

  updatePortfolio: (id: string, data: Partial<FIPortfolio>) =>
    apiClient.put<FIPortfolio>(`/fi/portfolios/${id}`, data).then((r) => r.data),

  deletePortfolio: (id: string) =>
    apiClient.delete(`/fi/portfolios/${id}`).then((r) => r.data),

  getCoverage: (orgId: string) =>
    apiClient.get<{ overall_pct: number; by_asset_class: { asset_class: string; coverage_pct: number; target_2030: number; target_2040: number }[]; path_to_100: { year: number; coverage_pct: number }[] }>(
      `/fi/coverage/${orgId}`
    ).then((r) => r.data),

  getFinancedEmissions: (orgId: string) =>
    apiClient.get<{ total: number; by_asset_class: { asset_class: string; emissions: number; pct: number }[]; by_sector: { sector: string; emissions: number; pct: number }[] }>(
      `/fi/financed-emissions/${orgId}`
    ).then((r) => r.data),

  getWACI: (orgId: string) =>
    apiClient.get<{ current_waci: number; trend: { year: number; waci: number }[]; benchmark: number }>(
      `/fi/waci/${orgId}`
    ).then((r) => r.data),

  getPortfolioTemperature: (portfolioId: string) =>
    apiClient.get<PortfolioTemperature>(`/fi/temperature/${portfolioId}`).then((r) => r.data),

  addHolding: (portfolioId: string, data: Partial<PortfolioHolding>) =>
    apiClient.post<PortfolioHolding>(`/fi/portfolios/${portfolioId}/holdings`, data).then((r) => r.data),

  updateHolding: (portfolioId: string, holdingId: string, data: Partial<PortfolioHolding>) =>
    apiClient.put<PortfolioHolding>(`/fi/portfolios/${portfolioId}/holdings/${holdingId}`, data).then((r) => r.data),

  addEngagement: (portfolioId: string, data: Partial<EngagementRecord>) =>
    apiClient.post<EngagementRecord>(`/fi/portfolios/${portfolioId}/engagements`, data).then((r) => r.data),

  getEngagements: (portfolioId: string) =>
    apiClient.get<EngagementRecord[]>(`/fi/portfolios/${portfolioId}/engagements`).then((r) => r.data),

  exportPortfolio: (portfolioId: string, format: ExportFormat) =>
    apiClient.get(`/fi/portfolios/${portfolioId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Framework API ───────────────────────────────────────────────────────────

export const frameworkApi = {
  getAlignmentSummary: (orgId: string) =>
    apiClient.get<AlignmentItem[]>(`/frameworks/alignment/${orgId}`).then((r) => r.data),

  getMappings: (orgId: string, framework?: string) =>
    apiClient.get<FrameworkMapping[]>(`/frameworks/mappings/${orgId}`, { params: { framework } }).then((r) => r.data),

  getGaps: (orgId: string) =>
    apiClient.get<FrameworkMapping[]>(`/frameworks/gaps/${orgId}`).then((r) => r.data),

  runAlignment: (orgId: string) =>
    apiClient.post<AlignmentItem[]>(`/frameworks/alignment/${orgId}/run`).then((r) => r.data),

  exportAlignment: (orgId: string, format: ExportFormat) =>
    apiClient.get(`/frameworks/alignment/${orgId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Report API ──────────────────────────────────────────────────────────────

export const reportApi = {
  getReports: (orgId: string) =>
    apiClient.get<Report[]>(`/reports/${orgId}`).then((r) => r.data),

  generateReport: (data: { org_id: string; report_type: string; target_ids: string[]; year: number }) =>
    apiClient.post<Report>('/reports/generate', data).then((r) => r.data),

  getReport: (id: string) =>
    apiClient.get<Report>(`/reports/detail/${id}`).then((r) => r.data),

  exportReport: (id: string, format: ExportFormat) =>
    apiClient.get(`/reports/${id}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),

  getSubmissionForm: (orgId: string) =>
    apiClient.get<SubmissionForm>(`/reports/submission/${orgId}`).then((r) => r.data),

  submitToSBTi: (data: SubmissionForm) =>
    apiClient.post('/reports/submit', data).then((r) => r.data),

  deleteReport: (id: string) =>
    apiClient.delete(`/reports/${id}`).then((r) => r.data),
};

// ─── Dashboard API ───────────────────────────────────────────────────────────

export const dashboardApi = {
  getSummary: (orgId: string) =>
    apiClient.get<DashboardSummary>(`/dashboard/${orgId}`).then((r) => r.data),

  getReadiness: (orgId: string) =>
    apiClient.get<{ readiness_score: number; category_scores: { category: string; score: number }[] }>(
      `/dashboard/${orgId}/readiness`
    ).then((r) => r.data),

  getTemperatureSummary: (orgId: string) =>
    apiClient.get<{ score: number; alignment: string; trend: { year: number; score: number }[] }>(
      `/dashboard/${orgId}/temperature`
    ).then((r) => r.data),

  getEmissionsTrend: (orgId: string) =>
    apiClient.get<{ year: number; scope_1: number; scope_2: number; scope_3: number; total: number }[]>(
      `/dashboard/${orgId}/emissions-trend`
    ).then((r) => r.data),

  getRecentActivity: (orgId: string) =>
    apiClient.get<DashboardSummary['recent_activity']>(`/dashboard/${orgId}/activity`).then((r) => r.data),
};

// ─── Gap Analysis API ────────────────────────────────────────────────────────

export const gapApi = {
  runAssessment: (orgId: string) =>
    apiClient.post<GapAssessment>(`/gap-analysis/${orgId}/run`).then((r) => r.data),

  getResults: (orgId: string) =>
    apiClient.get<GapAssessment>(`/gap-analysis/${orgId}`).then((r) => r.data),

  getActionPlan: (orgId: string) =>
    apiClient.get<GapAction[]>(`/gap-analysis/${orgId}/actions`).then((r) => r.data),

  updateAction: (id: string, data: Partial<GapAction>) =>
    apiClient.put<GapAction>(`/gap-analysis/actions/${id}`, data).then((r) => r.data),

  exportAssessment: (orgId: string, format: ExportFormat) =>
    apiClient.get(`/gap-analysis/${orgId}/export`, { params: { format }, responseType: 'blob' }).then((r) => r.data),
};

// ─── Settings API ────────────────────────────────────────────────────────────

export const settingsApi = {
  getSettings: (orgId: string) =>
    apiClient.get<OrganizationSettings>(`/settings/${orgId}`).then((r) => r.data),

  updateSettings: (orgId: string, data: Partial<OrganizationSettings>) =>
    apiClient.put<OrganizationSettings>(`/settings/${orgId}`, data).then((r) => r.data),

  getOrganizations: () =>
    apiClient.get<{ id: string; name: string }[]>('/settings/organizations').then((r) => r.data),

  getInventory: (orgId: string, year?: number) =>
    apiClient.get<EmissionsInventory>(`/settings/${orgId}/inventory`, { params: { year } }).then((r) => r.data),

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
