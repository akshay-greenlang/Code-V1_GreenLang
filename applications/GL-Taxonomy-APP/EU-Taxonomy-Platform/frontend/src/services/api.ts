/**
 * GL-Taxonomy-APP API Client
 *
 * Axios-based API client with namespaced endpoints for all taxonomy services.
 * Matches backend router structure at /api/v1/taxonomy/.
 */

import axios from 'axios';
import type {
  EconomicActivity,
  NACEMapping,
  ActivityStatistics,
  EligibilityScreening,
  ScreeningSummary,
  BatchScreenRequest,
  ActivityEligibility,
  SCAssessment,
  SCCriteria,
  SCProfile,
  ThresholdCheck,
  DNSHAssessment,
  ObjectiveDNSH,
  ClimateRiskAssessment,
  DNSHMatrix,
  SafeguardAssessment,
  TopicAssessment,
  DueDiligenceRecord,
  AdverseFinding,
  KPICalculation,
  KPISummary,
  KPIDetail,
  CapExPlan,
  ObjectiveBreakdown,
  GARCalculation,
  GARDetail,
  BTARCalculation,
  ExposureBreakdown,
  SectorGAR,
  EBATemplateData,
  MortgageAlignment,
  AutoLoanAlignment,
  AlignmentResult,
  PortfolioAlignment,
  AlignmentProgress,
  AlignmentFunnelData,
  BatchAlignmentRequest,
  DisclosureReport,
  Article8Data,
  ReportHistory,
  ReportComparison,
  Portfolio,
  Holding,
  CounterpartySearchResult,
  DataQualityScore,
  DimensionScore,
  EvidenceTracker,
  ImprovementPlan,
  DelegatedActVersion,
  RegulatoryUpdate,
  OmnibusImpact,
  GapAssessment,
  GapItem,
  ActionPlanItem,
  TaxonomySettings,
  ReportingPeriod,
  MRVMapping,
  DashboardOverview,
  AlignmentSummaryCard,
  TrendDataPoint,
  SectorBreakdownItem,
  PaginatedResponse,
  EnvironmentalObjective,
  SafeguardTopic,
  ExportFormat,
  ReportTemplateType,
  ActivitySearchParams,
  KPIDashboardParams,
  GARParams,
  ReportExportParams,
  PortfolioUploadParams,
} from '../types';

const api = axios.create({
  baseURL: '/api/v1/taxonomy',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('gl_access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('gl_access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ---------------------------------------------------------------------------
// Activities API
// ---------------------------------------------------------------------------
export const activitiesApi = {
  list: async (params?: ActivitySearchParams): Promise<PaginatedResponse<EconomicActivity>> => {
    const { data } = await api.get('/activities', { params });
    return data;
  },
  get: async (id: string): Promise<EconomicActivity> => {
    const { data } = await api.get(`/activities/${id}`);
    return data;
  },
  searchByNACE: async (naceCode: string): Promise<NACEMapping[]> => {
    const { data } = await api.get('/activities/nace/search', { params: { code: naceCode } });
    return data;
  },
  bySector: async (sector: string): Promise<EconomicActivity[]> => {
    const { data } = await api.get('/activities/by-sector', { params: { sector } });
    return data;
  },
  byObjective: async (objective: EnvironmentalObjective): Promise<EconomicActivity[]> => {
    const { data } = await api.get('/activities/by-objective', { params: { objective } });
    return data;
  },
  search: async (query: string): Promise<EconomicActivity[]> => {
    const { data } = await api.get('/activities/search', { params: { q: query } });
    return data;
  },
  statistics: async (orgId: string): Promise<ActivityStatistics> => {
    const { data } = await api.get(`/activities/statistics/${orgId}`);
    return data;
  },
};

// ---------------------------------------------------------------------------
// Screening API
// ---------------------------------------------------------------------------
export const screeningApi = {
  screenEligibility: async (activityId: string): Promise<ActivityEligibility> => {
    const { data } = await api.post(`/screening/eligibility/${activityId}`);
    return data;
  },
  batchScreen: async (request: BatchScreenRequest): Promise<EligibilityScreening> => {
    const { data } = await api.post('/screening/batch', request);
    return data;
  },
  getResults: async (orgId: string): Promise<EligibilityScreening> => {
    const { data } = await api.get(`/screening/results/${orgId}`);
    return data;
  },
  getSummary: async (orgId: string): Promise<ScreeningSummary> => {
    const { data } = await api.get(`/screening/summary/${orgId}`);
    return data;
  },
  applyDeMinimis: async (orgId: string, threshold: number): Promise<EligibilityScreening> => {
    const { data } = await api.post(`/screening/de-minimis/${orgId}`, { threshold });
    return data;
  },
};

// ---------------------------------------------------------------------------
// Substantial Contribution API
// ---------------------------------------------------------------------------
export const scApi = {
  assess: async (activityId: string, objective: EnvironmentalObjective): Promise<SCAssessment> => {
    const { data } = await api.post(`/sc/assess/${activityId}`, { objective });
    return data;
  },
  batchAssess: async (orgId: string, objective: EnvironmentalObjective): Promise<SCAssessment[]> => {
    const { data } = await api.post(`/sc/batch/${orgId}`, { objective });
    return data;
  },
  getResults: async (activityId: string): Promise<SCAssessment[]> => {
    const { data } = await api.get(`/sc/results/${activityId}`);
    return data;
  },
  getCriteria: async (activityId: string, objective: EnvironmentalObjective): Promise<SCCriteria> => {
    const { data } = await api.get(`/sc/criteria/${activityId}`, { params: { objective } });
    return data;
  },
  getProfile: async (activityId: string): Promise<SCProfile> => {
    const { data } = await api.get(`/sc/profile/${activityId}`);
    return data;
  },
  checkThreshold: async (activityId: string, criterionId: string, value: number): Promise<ThresholdCheck> => {
    const { data } = await api.post(`/sc/threshold/${activityId}`, { criterion_id: criterionId, value });
    return data;
  },
};

// ---------------------------------------------------------------------------
// DNSH API
// ---------------------------------------------------------------------------
export const dnshApi = {
  assess: async (activityId: string, scObjective: EnvironmentalObjective): Promise<DNSHAssessment> => {
    const { data } = await api.post(`/dnsh/assess/${activityId}`, { sc_objective: scObjective });
    return data;
  },
  assessObjective: async (activityId: string, objective: EnvironmentalObjective): Promise<ObjectiveDNSH> => {
    const { data } = await api.post(`/dnsh/objective/${activityId}`, { objective });
    return data;
  },
  climateRisk: async (activityId: string): Promise<ClimateRiskAssessment> => {
    const { data } = await api.post(`/dnsh/climate-risk/${activityId}`);
    return data;
  },
  water: async (activityId: string, payload: Record<string, unknown>): Promise<ObjectiveDNSH> => {
    const { data } = await api.post(`/dnsh/water/${activityId}`, payload);
    return data;
  },
  circular: async (activityId: string, payload: Record<string, unknown>): Promise<ObjectiveDNSH> => {
    const { data } = await api.post(`/dnsh/circular/${activityId}`, payload);
    return data;
  },
  pollution: async (activityId: string, payload: Record<string, unknown>): Promise<ObjectiveDNSH> => {
    const { data } = await api.post(`/dnsh/pollution/${activityId}`, payload);
    return data;
  },
  biodiversity: async (activityId: string, payload: Record<string, unknown>): Promise<ObjectiveDNSH> => {
    const { data } = await api.post(`/dnsh/biodiversity/${activityId}`, payload);
    return data;
  },
  getMatrix: async (orgId: string): Promise<DNSHMatrix[]> => {
    const { data } = await api.get(`/dnsh/matrix/${orgId}`);
    return data;
  },
};

// ---------------------------------------------------------------------------
// Safeguards API
// ---------------------------------------------------------------------------
export const safeguardsApi = {
  assess: async (orgId: string): Promise<SafeguardAssessment> => {
    const { data } = await api.post(`/safeguards/assess/${orgId}`);
    return data;
  },
  assessTopic: async (orgId: string, topic: SafeguardTopic): Promise<TopicAssessment> => {
    const { data } = await api.post(`/safeguards/topic/${orgId}`, { topic });
    return data;
  },
  procedural: async (orgId: string, topic: SafeguardTopic): Promise<TopicAssessment> => {
    const { data } = await api.get(`/safeguards/procedural/${orgId}`, { params: { topic } });
    return data;
  },
  outcome: async (orgId: string, topic: SafeguardTopic): Promise<TopicAssessment> => {
    const { data } = await api.get(`/safeguards/outcome/${orgId}`, { params: { topic } });
    return data;
  },
  recordFinding: async (orgId: string, finding: Partial<AdverseFinding>): Promise<AdverseFinding> => {
    const { data } = await api.post(`/safeguards/findings/${orgId}`, finding);
    return data;
  },
  getResults: async (orgId: string): Promise<SafeguardAssessment> => {
    const { data } = await api.get(`/safeguards/results/${orgId}`);
    return data;
  },
  getDueDiligence: async (orgId: string): Promise<DueDiligenceRecord[]> => {
    const { data } = await api.get(`/safeguards/due-diligence/${orgId}`);
    return data;
  },
};

// ---------------------------------------------------------------------------
// KPI API
// ---------------------------------------------------------------------------
export const kpiApi = {
  calculate: async (orgId: string, period: string): Promise<KPICalculation> => {
    const { data } = await api.post(`/kpi/calculate/${orgId}`, { period });
    return data;
  },
  turnover: async (orgId: string, period: string): Promise<KPIDetail> => {
    const { data } = await api.get(`/kpi/turnover/${orgId}`, { params: { period } });
    return data;
  },
  capex: async (orgId: string, period: string): Promise<KPIDetail> => {
    const { data } = await api.get(`/kpi/capex/${orgId}`, { params: { period } });
    return data;
  },
  opex: async (orgId: string, period: string): Promise<KPIDetail> => {
    const { data } = await api.get(`/kpi/opex/${orgId}`, { params: { period } });
    return data;
  },
  capexPlan: async (activityId: string): Promise<CapExPlan[]> => {
    const { data } = await api.get(`/kpi/capex-plan/${activityId}`);
    return data;
  },
  dashboard: async (params: KPIDashboardParams): Promise<KPISummary> => {
    const { data } = await api.get('/kpi/dashboard', { params });
    return data;
  },
  objectiveBreakdown: async (orgId: string, period: string): Promise<ObjectiveBreakdown[]> => {
    const { data } = await api.get(`/kpi/objective-breakdown/${orgId}`, { params: { period } });
    return data;
  },
};

// ---------------------------------------------------------------------------
// GAR API
// ---------------------------------------------------------------------------
export const garApi = {
  stock: async (params: GARParams): Promise<GARDetail> => {
    const { data } = await api.get('/gar/stock', { params });
    return data;
  },
  flow: async (params: GARParams): Promise<GARDetail> => {
    const { data } = await api.get('/gar/flow', { params });
    return data;
  },
  btar: async (orgId: string, reportingDate: string): Promise<BTARCalculation> => {
    const { data } = await api.get(`/gar/btar/${orgId}`, { params: { reporting_date: reportingDate } });
    return data;
  },
  classifyExposure: async (orgId: string, exposureId: string): Promise<ExposureBreakdown> => {
    const { data } = await api.post(`/gar/classify/${orgId}`, { exposure_id: exposureId });
    return data;
  },
  sectorBreakdown: async (orgId: string, reportingDate: string): Promise<SectorGAR[]> => {
    const { data } = await api.get(`/gar/sectors/${orgId}`, { params: { reporting_date: reportingDate } });
    return data;
  },
  trends: async (orgId: string, periods: number): Promise<GARDetail[]> => {
    const { data } = await api.get(`/gar/trends/${orgId}`, { params: { periods } });
    return data;
  },
  ebaTemplate: async (orgId: string, templateId: string): Promise<EBATemplateData> => {
    const { data } = await api.get(`/gar/eba-template/${orgId}`, { params: { template_id: templateId } });
    return data;
  },
  checkMortgage: async (orgId: string, payload: Record<string, unknown>): Promise<MortgageAlignment> => {
    const { data } = await api.post(`/gar/mortgage/${orgId}`, payload);
    return data;
  },
  checkAutoLoan: async (orgId: string, payload: Record<string, unknown>): Promise<AutoLoanAlignment> => {
    const { data } = await api.post(`/gar/auto-loan/${orgId}`, payload);
    return data;
  },
};

// ---------------------------------------------------------------------------
// Alignment API
// ---------------------------------------------------------------------------
export const alignmentApi = {
  full: async (activityId: string, objective: EnvironmentalObjective): Promise<AlignmentResult> => {
    const { data } = await api.post(`/alignment/full/${activityId}`, { objective });
    return data;
  },
  portfolio: async (orgId: string): Promise<PortfolioAlignment> => {
    const { data } = await api.get(`/alignment/portfolio/${orgId}`);
    return data;
  },
  batch: async (request: BatchAlignmentRequest): Promise<AlignmentResult[]> => {
    const { data } = await api.post('/alignment/batch', request);
    return data;
  },
  getStatus: async (activityId: string): Promise<AlignmentResult> => {
    const { data } = await api.get(`/alignment/status/${activityId}`);
    return data;
  },
  progress: async (orgId: string): Promise<AlignmentProgress[]> => {
    const { data } = await api.get(`/alignment/progress/${orgId}`);
    return data;
  },
  dashboard: async (orgId: string): Promise<AlignmentSummaryCard> => {
    const { data } = await api.get(`/alignment/dashboard/${orgId}`);
    return data;
  },
  eligibleVsAligned: async (orgId: string): Promise<AlignmentFunnelData> => {
    const { data } = await api.get(`/alignment/funnel/${orgId}`);
    return data;
  },
};

// ---------------------------------------------------------------------------
// Reporting API
// ---------------------------------------------------------------------------
export const reportingApi = {
  article8: async (orgId: string, period: string): Promise<Article8Data> => {
    const { data } = await api.get(`/reporting/article8/${orgId}`, { params: { period } });
    return data;
  },
  eba: async (orgId: string, templateId: string): Promise<EBATemplateData> => {
    const { data } = await api.get(`/reporting/eba/${orgId}`, { params: { template_id: templateId } });
    return data;
  },
  export: async (params: ReportExportParams): Promise<Blob> => {
    const { data } = await api.get('/reporting/export', {
      params,
      responseType: 'blob',
    });
    return data;
  },
  xbrl: async (orgId: string, period: string): Promise<Blob> => {
    const { data } = await api.get(`/reporting/xbrl/${orgId}`, {
      params: { period },
      responseType: 'blob',
    });
    return data;
  },
  history: async (orgId: string): Promise<ReportHistory[]> => {
    const { data } = await api.get(`/reporting/history/${orgId}`);
    return data;
  },
  compare: async (orgId: string, period1: string, period2: string): Promise<ReportComparison> => {
    const { data } = await api.get(`/reporting/compare/${orgId}`, { params: { period_1: period1, period_2: period2 } });
    return data;
  },
  qualitative: async (orgId: string, reportId: string): Promise<DisclosureReport> => {
    const { data } = await api.get(`/reporting/qualitative/${orgId}/${reportId}`);
    return data;
  },
  create: async (orgId: string, payload: Partial<DisclosureReport>): Promise<DisclosureReport> => {
    const { data } = await api.post(`/reporting/create/${orgId}`, payload);
    return data;
  },
};

// ---------------------------------------------------------------------------
// Portfolio API
// ---------------------------------------------------------------------------
export const portfolioApi = {
  create: async (portfolio: Partial<Portfolio>): Promise<Portfolio> => {
    const { data } = await api.post('/portfolio', portfolio);
    return data;
  },
  get: async (id: string): Promise<Portfolio> => {
    const { data } = await api.get(`/portfolio/${id}`);
    return data;
  },
  update: async (id: string, portfolio: Partial<Portfolio>): Promise<Portfolio> => {
    const { data } = await api.put(`/portfolio/${id}`, portfolio);
    return data;
  },
  delete: async (id: string): Promise<void> => {
    await api.delete(`/portfolio/${id}`);
  },
  list: async (orgId: string): Promise<Portfolio[]> => {
    const { data } = await api.get('/portfolio', { params: { organization_id: orgId } });
    return data;
  },
  addHoldings: async (portfolioId: string, holdings: Partial<Holding>[]): Promise<Holding[]> => {
    const { data } = await api.post(`/portfolio/${portfolioId}/holdings`, { holdings });
    return data;
  },
  getHoldings: async (portfolioId: string): Promise<Holding[]> => {
    const { data } = await api.get(`/portfolio/${portfolioId}/holdings`);
    return data;
  },
  upload: async (params: PortfolioUploadParams, file: File): Promise<Holding[]> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('mapping', JSON.stringify(params.mapping));
    formData.append('file_format', params.file_format);
    const { data } = await api.post(`/portfolio/${params.portfolio_id}/upload`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  },
  searchCounterparty: async (query: string): Promise<CounterpartySearchResult[]> => {
    const { data } = await api.get('/portfolio/counterparty/search', { params: { q: query } });
    return data;
  },
};

// ---------------------------------------------------------------------------
// Dashboard API
// ---------------------------------------------------------------------------
export const dashboardApi = {
  overview: async (orgId: string, period: string): Promise<DashboardOverview> => {
    const { data } = await api.get(`/dashboard/overview/${orgId}`, { params: { period } });
    return data;
  },
  alignmentSummary: async (orgId: string): Promise<AlignmentSummaryCard> => {
    const { data } = await api.get(`/dashboard/alignment/${orgId}`);
    return data;
  },
  kpiCards: async (orgId: string, period: string): Promise<KPISummary> => {
    const { data } = await api.get(`/dashboard/kpi/${orgId}`, { params: { period } });
    return data;
  },
  sectorBreakdown: async (orgId: string): Promise<SectorBreakdownItem[]> => {
    const { data } = await api.get(`/dashboard/sectors/${orgId}`);
    return data;
  },
  trends: async (orgId: string, periods: number): Promise<TrendDataPoint[]> => {
    const { data } = await api.get(`/dashboard/trends/${orgId}`, { params: { periods } });
    return data;
  },
};

// ---------------------------------------------------------------------------
// Data Quality API
// ---------------------------------------------------------------------------
export const dataQualityApi = {
  assess: async (orgId: string): Promise<DataQualityScore> => {
    const { data } = await api.post(`/data-quality/assess/${orgId}`);
    return data;
  },
  dashboard: async (orgId: string): Promise<DataQualityScore> => {
    const { data } = await api.get(`/data-quality/dashboard/${orgId}`);
    return data;
  },
  dimensions: async (orgId: string): Promise<DimensionScore[]> => {
    const { data } = await api.get(`/data-quality/dimensions/${orgId}`);
    return data;
  },
  evidence: async (orgId: string): Promise<EvidenceTracker> => {
    const { data } = await api.get(`/data-quality/evidence/${orgId}`);
    return data;
  },
  improvementPlan: async (orgId: string): Promise<ImprovementPlan> => {
    const { data } = await api.get(`/data-quality/improvement/${orgId}`);
    return data;
  },
};

// ---------------------------------------------------------------------------
// Regulatory API
// ---------------------------------------------------------------------------
export const regulatoryApi = {
  delegatedActs: async (): Promise<DelegatedActVersion[]> => {
    const { data } = await api.get('/regulatory/delegated-acts');
    return data;
  },
  updates: async (limit?: number): Promise<RegulatoryUpdate[]> => {
    const { data } = await api.get('/regulatory/updates', { params: { limit } });
    return data;
  },
  omnibusImpact: async (orgId: string): Promise<OmnibusImpact[]> => {
    const { data } = await api.get(`/regulatory/omnibus/${orgId}`);
    return data;
  },
  applicableVersion: async (orgId: string): Promise<DelegatedActVersion> => {
    const { data } = await api.get(`/regulatory/applicable/${orgId}`);
    return data;
  },
};

// ---------------------------------------------------------------------------
// Gap Analysis API
// ---------------------------------------------------------------------------
export const gapApi = {
  analyze: async (orgId: string): Promise<GapAssessment> => {
    const { data } = await api.post(`/gap/analyze/${orgId}`);
    return data;
  },
  results: async (orgId: string): Promise<GapAssessment> => {
    const { data } = await api.get(`/gap/results/${orgId}`);
    return data;
  },
  dnshGaps: async (orgId: string): Promise<GapItem[]> => {
    const { data } = await api.get(`/gap/dnsh/${orgId}`);
    return data;
  },
  safeguardGaps: async (orgId: string): Promise<GapItem[]> => {
    const { data } = await api.get(`/gap/safeguards/${orgId}`);
    return data;
  },
  dataGaps: async (orgId: string): Promise<GapItem[]> => {
    const { data } = await api.get(`/gap/data/${orgId}`);
    return data;
  },
  actionPlan: async (orgId: string): Promise<ActionPlanItem[]> => {
    const { data } = await api.get(`/gap/action-plan/${orgId}`);
    return data;
  },
};

// ---------------------------------------------------------------------------
// Settings API
// ---------------------------------------------------------------------------
export const settingsApi = {
  get: async (orgId: string): Promise<TaxonomySettings> => {
    const { data } = await api.get(`/settings/${orgId}`);
    return data;
  },
  update: async (orgId: string, settings: Partial<TaxonomySettings>): Promise<TaxonomySettings> => {
    const { data } = await api.put(`/settings/${orgId}`, settings);
    return data;
  },
  reportingPeriods: async (orgId: string): Promise<ReportingPeriod[]> => {
    const { data } = await api.get(`/settings/periods/${orgId}`);
    return data;
  },
  thresholds: async (orgId: string): Promise<Record<string, number>> => {
    const { data } = await api.get(`/settings/thresholds/${orgId}`);
    return data;
  },
  mrvMapping: async (orgId: string): Promise<MRVMapping[]> => {
    const { data } = await api.get(`/settings/mrv-mapping/${orgId}`);
    return data;
  },
};

export default api;
