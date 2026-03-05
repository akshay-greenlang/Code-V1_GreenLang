/**
 * GL-CDP-APP v1.0 - API Client Service
 *
 * Axios-based HTTP client for communicating with the CDP backend.
 * Provides 60+ typed methods across 10 route groups:
 *   1.  Questionnaire (8 methods)
 *   2.  Response      (10 methods)
 *   3.  Scoring       (6 methods)
 *   4.  Gap Analysis  (5 methods)
 *   5.  Benchmarking  (5 methods)
 *   6.  Supply Chain  (7 methods)
 *   7.  Transition    (7 methods)
 *   8.  Verification  (5 methods)
 *   9.  Reports       (6 methods)
 *   10. Dashboard     (4 methods)
 *   11. Settings      (4 methods)
 *   12. Historical    (4 methods)
 *
 * Features:
 *   - JWT bearer token injection via request interceptor
 *   - Automatic 401 redirect to login
 *   - Token refresh on expiry with failed-queue replay
 *   - Typed request/response contracts
 */

import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import type {
  Questionnaire,
  Module,
  Question,
  Response as CDPResponse,
  ResponseVersion,
  Evidence,
  ReviewComment,
  ScoringResult,
  CategoryScore,
  WhatIfScenario,
  GapAnalysis,
  GapRecommendation,
  Benchmark,
  PeerComparison,
  SupplierRequest,
  SupplierResponse,
  SupplyChainSummary,
  TransitionPlan,
  TransitionMilestone,
  PathwayPoint,
  VerificationRecord,
  VerificationSummary,
  HistoricalScore,
  YearComparison,
  CDPReport,
  SubmissionChecklist,
  DashboardData,
  DashboardAlert,
  OrganizationSettings,
  ExportResult,
  ApiResponse,
  CreateQuestionnaireRequest,
  SaveResponseRequest,
  BulkSaveResponsesRequest,
  SubmitForReviewRequest,
  ApproveResponsesRequest,
  SimulateScoringRequest,
  WhatIfRequest,
  RunGapAnalysisRequest,
  InviteSupplierRequest,
  CreateTransitionPlanRequest,
  AddMilestoneRequest,
  CreateVerificationRequest,
  GenerateReportRequest,
  UpdateSettingsRequest,
  ScoringLevel,
  ARequirement,
} from '../types';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';
const TOKEN_KEY = 'cdp_access_token';
const REFRESH_KEY = 'cdp_refresh_token';

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

class CDPApiClient {
  private client: AxiosInstance;
  private isRefreshing = false;
  private failedQueue: Array<{
    resolve: (token: string) => void;
    reject: (err: unknown) => void;
  }> = [];

  constructor() {
    this.client = axios.create({
      baseURL: BASE_URL,
      timeout: 30_000,
      headers: { 'Content-Type': 'application/json' },
    });

    // -- Request interceptor: attach JWT --
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        const token = localStorage.getItem(TOKEN_KEY);
        if (token && config.headers) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error),
    );

    // -- Response interceptor: handle 401 / token refresh --
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };

        if (error.response?.status === 401 && !originalRequest._retry) {
          const refreshToken = localStorage.getItem(REFRESH_KEY);

          if (refreshToken) {
            if (this.isRefreshing) {
              return new Promise((resolve, reject) => {
                this.failedQueue.push({ resolve, reject });
              }).then((token) => {
                if (originalRequest.headers) {
                  originalRequest.headers.Authorization = `Bearer ${token}`;
                }
                return this.client(originalRequest);
              });
            }

            originalRequest._retry = true;
            this.isRefreshing = true;

            try {
              const { data } = await axios.post(`${BASE_URL}/auth/refresh`, {
                refresh_token: refreshToken,
              });
              const newToken = data.access_token;
              localStorage.setItem(TOKEN_KEY, newToken);
              if (data.refresh_token) {
                localStorage.setItem(REFRESH_KEY, data.refresh_token);
              }
              this.processQueue(null, newToken);
              if (originalRequest.headers) {
                originalRequest.headers.Authorization = `Bearer ${newToken}`;
              }
              return this.client(originalRequest);
            } catch (refreshError) {
              this.processQueue(refreshError, null);
              localStorage.removeItem(TOKEN_KEY);
              localStorage.removeItem(REFRESH_KEY);
              window.location.href = '/login';
              return Promise.reject(refreshError);
            } finally {
              this.isRefreshing = false;
            }
          } else {
            localStorage.removeItem(TOKEN_KEY);
            window.location.href = '/login';
          }
        }

        return Promise.reject(error);
      },
    );
  }

  private processQueue(error: unknown, token: string | null): void {
    this.failedQueue.forEach((prom) => {
      if (error) {
        prom.reject(error);
      } else {
        prom.resolve(token!);
      }
    });
    this.failedQueue = [];
  }

  // =========================================================================
  // 1. QUESTIONNAIRE ROUTES (8 methods)
  // =========================================================================

  async createQuestionnaire(payload: CreateQuestionnaireRequest): Promise<Questionnaire> {
    const { data } = await this.client.post<ApiResponse<Questionnaire>>(
      '/cdp/questionnaires',
      payload,
    );
    return data.data;
  }

  async getQuestionnaire(questionnaireId: string): Promise<Questionnaire> {
    const { data } = await this.client.get<ApiResponse<Questionnaire>>(
      `/cdp/questionnaires/${questionnaireId}`,
    );
    return data.data;
  }

  async listQuestionnaires(orgId: string): Promise<Questionnaire[]> {
    const { data } = await this.client.get<ApiResponse<Questionnaire[]>>(
      '/cdp/questionnaires',
      { params: { org_id: orgId } },
    );
    return data.data;
  }

  async getModules(questionnaireId: string): Promise<Module[]> {
    const { data } = await this.client.get<ApiResponse<Module[]>>(
      `/cdp/questionnaires/${questionnaireId}/modules`,
    );
    return data.data;
  }

  async getModuleQuestions(questionnaireId: string, moduleId: string): Promise<Question[]> {
    const { data } = await this.client.get<ApiResponse<Question[]>>(
      `/cdp/questionnaires/${questionnaireId}/modules/${moduleId}/questions`,
    );
    return data.data;
  }

  async getQuestionnaireProgress(questionnaireId: string): Promise<ModuleProgress[]> {
    const { data } = await this.client.get<ApiResponse<ModuleProgress[]>>(
      `/cdp/questionnaires/${questionnaireId}/progress`,
    );
    return data.data;
  }

  async importPreviousYear(questionnaireId: string, sourceYear: number): Promise<void> {
    await this.client.post(
      `/cdp/questionnaires/${questionnaireId}/import`,
      { source_year: sourceYear },
    );
  }

  async autoPopulateModule(questionnaireId: string, moduleId: string): Promise<void> {
    await this.client.post(
      `/cdp/questionnaires/${questionnaireId}/modules/${moduleId}/auto-populate`,
    );
  }

  // =========================================================================
  // 2. RESPONSE ROUTES (10 methods)
  // =========================================================================

  async getResponse(questionId: string): Promise<CDPResponse> {
    const { data } = await this.client.get<ApiResponse<CDPResponse>>(
      `/cdp/responses/${questionId}`,
    );
    return data.data;
  }

  async getResponsesByModule(questionnaireId: string, moduleId: string): Promise<Record<string, CDPResponse>> {
    const { data } = await this.client.get<ApiResponse<Record<string, CDPResponse>>>(
      `/cdp/questionnaires/${questionnaireId}/modules/${moduleId}/responses`,
    );
    return data.data;
  }

  async saveResponse(questionId: string, payload: SaveResponseRequest): Promise<CDPResponse> {
    const { data } = await this.client.put<ApiResponse<CDPResponse>>(
      `/cdp/responses/${questionId}`,
      payload,
    );
    return data.data;
  }

  async bulkSaveResponses(questionnaireId: string, payload: BulkSaveResponsesRequest): Promise<void> {
    await this.client.post(
      `/cdp/questionnaires/${questionnaireId}/responses/bulk`,
      payload,
    );
  }

  async submitForReview(payload: SubmitForReviewRequest): Promise<void> {
    await this.client.post('/cdp/responses/submit-review', payload);
  }

  async approveResponses(payload: ApproveResponsesRequest): Promise<void> {
    await this.client.post('/cdp/responses/approve', payload);
  }

  async getResponseVersions(responseId: string): Promise<ResponseVersion[]> {
    const { data } = await this.client.get<ApiResponse<ResponseVersion[]>>(
      `/cdp/responses/${responseId}/versions`,
    );
    return data.data;
  }

  async uploadEvidence(responseId: string, formData: FormData): Promise<Evidence> {
    const { data } = await this.client.post<ApiResponse<Evidence>>(
      `/cdp/responses/${responseId}/evidence`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } },
    );
    return data.data;
  }

  async deleteEvidence(responseId: string, evidenceId: string): Promise<void> {
    await this.client.delete(`/cdp/responses/${responseId}/evidence/${evidenceId}`);
  }

  async getComments(responseId: string): Promise<ReviewComment[]> {
    const { data } = await this.client.get<ApiResponse<ReviewComment[]>>(
      `/cdp/responses/${responseId}/comments`,
    );
    return data.data;
  }

  // =========================================================================
  // 3. SCORING ROUTES (6 methods)
  // =========================================================================

  async simulateScore(payload: SimulateScoringRequest): Promise<ScoringResult> {
    const { data } = await this.client.post<ApiResponse<ScoringResult>>(
      '/cdp/scoring/simulate',
      payload,
    );
    return data.data;
  }

  async getCategoryScores(questionnaireId: string): Promise<CategoryScore[]> {
    const { data } = await this.client.get<ApiResponse<CategoryScore[]>>(
      `/cdp/scoring/${questionnaireId}/categories`,
    );
    return data.data;
  }

  async runWhatIf(payload: WhatIfRequest): Promise<WhatIfScenario> {
    const { data } = await this.client.post<ApiResponse<WhatIfScenario>>(
      '/cdp/scoring/what-if',
      payload,
    );
    return data.data;
  }

  async checkALevelEligibility(questionnaireId: string): Promise<ARequirement[]> {
    const { data } = await this.client.get<ApiResponse<ARequirement[]>>(
      `/cdp/scoring/${questionnaireId}/a-level-check`,
    );
    return data.data;
  }

  async getScoreHistory(orgId: string): Promise<HistoricalScore[]> {
    const { data } = await this.client.get<ApiResponse<HistoricalScore[]>>(
      `/cdp/scoring/${orgId}/history`,
    );
    return data.data;
  }

  async exportScoreReport(questionnaireId: string): Promise<ExportResult> {
    const { data } = await this.client.post<ApiResponse<ExportResult>>(
      `/cdp/scoring/${questionnaireId}/export`,
    );
    return data.data;
  }

  // =========================================================================
  // 4. GAP ANALYSIS ROUTES (5 methods)
  // =========================================================================

  async runGapAnalysis(payload: RunGapAnalysisRequest): Promise<GapAnalysis> {
    const { data } = await this.client.post<ApiResponse<GapAnalysis>>(
      '/cdp/gaps/analyze',
      payload,
    );
    return data.data;
  }

  async getGapAnalysis(questionnaireId: string): Promise<GapAnalysis> {
    const { data } = await this.client.get<ApiResponse<GapAnalysis>>(
      `/cdp/gaps/${questionnaireId}`,
    );
    return data.data;
  }

  async getRecommendations(questionnaireId: string): Promise<GapRecommendation[]> {
    const { data } = await this.client.get<ApiResponse<GapRecommendation[]>>(
      `/cdp/gaps/${questionnaireId}/recommendations`,
    );
    return data.data;
  }

  async getUpliftPrediction(questionnaireId: string, gapIds: string[]): Promise<{ total_uplift: number; projected_score: number; projected_level: ScoringLevel }> {
    const { data } = await this.client.post<ApiResponse<{ total_uplift: number; projected_score: number; projected_level: ScoringLevel }>>(
      `/cdp/gaps/${questionnaireId}/uplift`,
      { gap_ids: gapIds },
    );
    return data.data;
  }

  async resolveGap(gapId: string): Promise<void> {
    await this.client.post(`/cdp/gaps/${gapId}/resolve`);
  }

  // =========================================================================
  // 5. BENCHMARKING ROUTES (5 methods)
  // =========================================================================

  async getSectorBenchmark(sector: string, year?: number): Promise<Benchmark> {
    const { data } = await this.client.get<ApiResponse<Benchmark>>(
      '/cdp/benchmarking/sector',
      { params: { sector, year } },
    );
    return data.data;
  }

  async getRegionalBenchmark(region: string, year?: number): Promise<Benchmark> {
    const { data } = await this.client.get<ApiResponse<Benchmark>>(
      '/cdp/benchmarking/regional',
      { params: { region, year } },
    );
    return data.data;
  }

  async getPeerComparison(orgId: string, sector?: string): Promise<PeerComparison> {
    const { data } = await this.client.get<ApiResponse<PeerComparison>>(
      `/cdp/benchmarking/${orgId}/peers`,
      { params: { sector } },
    );
    return data.data;
  }

  async getScoreDistribution(sector: string, year?: number): Promise<Benchmark> {
    const { data } = await this.client.get<ApiResponse<Benchmark>>(
      '/cdp/benchmarking/distribution',
      { params: { sector, year } },
    );
    return data.data;
  }

  async getBenchmarkTrend(sector: string, startYear: number, endYear: number): Promise<Benchmark[]> {
    const { data } = await this.client.get<ApiResponse<Benchmark[]>>(
      '/cdp/benchmarking/trend',
      { params: { sector, start_year: startYear, end_year: endYear } },
    );
    return data.data;
  }

  // =========================================================================
  // 6. SUPPLY CHAIN ROUTES (7 methods)
  // =========================================================================

  async getSuppliers(orgId: string): Promise<SupplierRequest[]> {
    const { data } = await this.client.get<ApiResponse<SupplierRequest[]>>(
      `/cdp/supply-chain/${orgId}/suppliers`,
    );
    return data.data;
  }

  async inviteSupplier(orgId: string, payload: InviteSupplierRequest): Promise<SupplierRequest> {
    const { data } = await this.client.post<ApiResponse<SupplierRequest>>(
      `/cdp/supply-chain/${orgId}/suppliers/invite`,
      payload,
    );
    return data.data;
  }

  async getSupplierResponses(orgId: string): Promise<SupplierResponse[]> {
    const { data } = await this.client.get<ApiResponse<SupplierResponse[]>>(
      `/cdp/supply-chain/${orgId}/responses`,
    );
    return data.data;
  }

  async getSupplyChainSummary(orgId: string): Promise<SupplyChainSummary> {
    const { data } = await this.client.get<ApiResponse<SupplyChainSummary>>(
      `/cdp/supply-chain/${orgId}/summary`,
    );
    return data.data;
  }

  async getEmissionHotspots(orgId: string): Promise<SupplyChainSummary> {
    const { data } = await this.client.get<ApiResponse<SupplyChainSummary>>(
      `/cdp/supply-chain/${orgId}/hotspots`,
    );
    return data.data;
  }

  async removeSupplier(orgId: string, supplierId: string): Promise<void> {
    await this.client.delete(`/cdp/supply-chain/${orgId}/suppliers/${supplierId}`);
  }

  async resendInvitation(orgId: string, supplierId: string): Promise<void> {
    await this.client.post(`/cdp/supply-chain/${orgId}/suppliers/${supplierId}/resend`);
  }

  // =========================================================================
  // 7. TRANSITION PLAN ROUTES (7 methods)
  // =========================================================================

  async getTransitionPlan(orgId: string): Promise<TransitionPlan> {
    const { data } = await this.client.get<ApiResponse<TransitionPlan>>(
      `/cdp/transition/${orgId}/plan`,
    );
    return data.data;
  }

  async createTransitionPlan(orgId: string, payload: CreateTransitionPlanRequest): Promise<TransitionPlan> {
    const { data } = await this.client.post<ApiResponse<TransitionPlan>>(
      `/cdp/transition/${orgId}/plan`,
      payload,
    );
    return data.data;
  }

  async updateTransitionPlan(orgId: string, payload: Partial<CreateTransitionPlanRequest>): Promise<TransitionPlan> {
    const { data } = await this.client.put<ApiResponse<TransitionPlan>>(
      `/cdp/transition/${orgId}/plan`,
      payload,
    );
    return data.data;
  }

  async addMilestone(orgId: string, payload: AddMilestoneRequest): Promise<TransitionMilestone> {
    const { data } = await this.client.post<ApiResponse<TransitionMilestone>>(
      `/cdp/transition/${orgId}/milestones`,
      payload,
    );
    return data.data;
  }

  async updateMilestone(orgId: string, milestoneId: string, payload: Partial<AddMilestoneRequest>): Promise<TransitionMilestone> {
    const { data } = await this.client.put<ApiResponse<TransitionMilestone>>(
      `/cdp/transition/${orgId}/milestones/${milestoneId}`,
      payload,
    );
    return data.data;
  }

  async getPathway(orgId: string): Promise<PathwayPoint[]> {
    const { data } = await this.client.get<ApiResponse<PathwayPoint[]>>(
      `/cdp/transition/${orgId}/pathway`,
    );
    return data.data;
  }

  async checkSBTiAlignment(orgId: string): Promise<{ aligned: boolean; status: string; details: string }> {
    const { data } = await this.client.get<ApiResponse<{ aligned: boolean; status: string; details: string }>>(
      `/cdp/transition/${orgId}/sbti-check`,
    );
    return data.data;
  }

  // =========================================================================
  // 8. VERIFICATION ROUTES (5 methods)
  // =========================================================================

  async getVerificationRecords(orgId: string): Promise<VerificationRecord[]> {
    const { data } = await this.client.get<ApiResponse<VerificationRecord[]>>(
      `/cdp/verification/${orgId}/records`,
    );
    return data.data;
  }

  async createVerification(orgId: string, payload: CreateVerificationRequest): Promise<VerificationRecord> {
    const { data } = await this.client.post<ApiResponse<VerificationRecord>>(
      `/cdp/verification/${orgId}/records`,
      payload,
    );
    return data.data;
  }

  async updateVerification(orgId: string, recordId: string, payload: Partial<CreateVerificationRequest>): Promise<VerificationRecord> {
    const { data } = await this.client.put<ApiResponse<VerificationRecord>>(
      `/cdp/verification/${orgId}/records/${recordId}`,
      payload,
    );
    return data.data;
  }

  async getVerificationSummary(orgId: string): Promise<VerificationSummary> {
    const { data } = await this.client.get<ApiResponse<VerificationSummary>>(
      `/cdp/verification/${orgId}/summary`,
    );
    return data.data;
  }

  async deleteVerification(orgId: string, recordId: string): Promise<void> {
    await this.client.delete(`/cdp/verification/${orgId}/records/${recordId}`);
  }

  // =========================================================================
  // 9. REPORTS ROUTES (6 methods)
  // =========================================================================

  async generateReport(payload: GenerateReportRequest): Promise<CDPReport> {
    const { data } = await this.client.post<ApiResponse<CDPReport>>(
      '/cdp/reports/generate',
      payload,
    );
    return data.data;
  }

  async getReports(questionnaireId: string): Promise<CDPReport[]> {
    const { data } = await this.client.get<ApiResponse<CDPReport[]>>(
      `/cdp/reports/questionnaire/${questionnaireId}`,
    );
    return data.data;
  }

  async downloadReport(reportId: string): Promise<Blob> {
    const { data } = await this.client.get(`/cdp/reports/${reportId}/download`, {
      responseType: 'blob',
    });
    return data;
  }

  async getSubmissionChecklist(questionnaireId: string): Promise<SubmissionChecklist> {
    const { data } = await this.client.get<ApiResponse<SubmissionChecklist>>(
      `/cdp/reports/${questionnaireId}/checklist`,
    );
    return data.data;
  }

  async submitToORS(questionnaireId: string): Promise<void> {
    await this.client.post(`/cdp/reports/${questionnaireId}/submit`);
  }

  async exportVerificationPackage(questionnaireId: string): Promise<ExportResult> {
    const { data } = await this.client.post<ApiResponse<ExportResult>>(
      `/cdp/reports/${questionnaireId}/verification-package`,
    );
    return data.data;
  }

  // =========================================================================
  // 10. DASHBOARD ROUTES (4 methods)
  // =========================================================================

  async getDashboard(orgId: string, reportingYear: number): Promise<DashboardData> {
    const { data } = await this.client.get<ApiResponse<DashboardData>>(
      `/cdp/dashboard/${orgId}`,
      { params: { reporting_year: reportingYear } },
    );
    return data.data;
  }

  async getDashboardAlerts(orgId: string): Promise<DashboardAlert[]> {
    const { data } = await this.client.get<ApiResponse<DashboardAlert[]>>(
      `/cdp/dashboard/${orgId}/alerts`,
    );
    return data.data;
  }

  async markAlertRead(alertId: string): Promise<void> {
    await this.client.put(`/cdp/dashboard/alerts/${alertId}/read`);
  }

  async getActivityFeed(orgId: string, limit?: number): Promise<import('../types').TimelineEvent[]> {
    const { data } = await this.client.get<ApiResponse<import('../types').TimelineEvent[]>>(
      `/cdp/dashboard/${orgId}/activity`,
      { params: { limit } },
    );
    return data.data;
  }

  // =========================================================================
  // 11. SETTINGS ROUTES (4 methods)
  // =========================================================================

  async getSettings(orgId: string): Promise<OrganizationSettings> {
    const { data } = await this.client.get<ApiResponse<OrganizationSettings>>(
      `/cdp/settings/${orgId}`,
    );
    return data.data;
  }

  async updateSettings(orgId: string, payload: UpdateSettingsRequest): Promise<OrganizationSettings> {
    const { data } = await this.client.put<ApiResponse<OrganizationSettings>>(
      `/cdp/settings/${orgId}`,
      payload,
    );
    return data.data;
  }

  async addTeamMember(orgId: string, member: { name: string; email: string; role: string }): Promise<void> {
    await this.client.post(`/cdp/settings/${orgId}/team`, member);
  }

  async removeTeamMember(orgId: string, memberId: string): Promise<void> {
    await this.client.delete(`/cdp/settings/${orgId}/team/${memberId}`);
  }

  // =========================================================================
  // 12. HISTORICAL ROUTES (4 methods)
  // =========================================================================

  async getHistoricalScores(orgId: string): Promise<HistoricalScore[]> {
    const { data } = await this.client.get<ApiResponse<HistoricalScore[]>>(
      `/cdp/historical/${orgId}/scores`,
    );
    return data.data;
  }

  async getYearComparison(orgId: string, yearA: number, yearB: number): Promise<YearComparison> {
    const { data } = await this.client.get<ApiResponse<YearComparison>>(
      `/cdp/historical/${orgId}/compare`,
      { params: { year_a: yearA, year_b: yearB } },
    );
    return data.data;
  }

  async getChangeLog(orgId: string, year: number): Promise<import('../types').ChangeLogEntry[]> {
    const { data } = await this.client.get<ApiResponse<import('../types').ChangeLogEntry[]>>(
      `/cdp/historical/${orgId}/changelog`,
      { params: { year } },
    );
    return data.data;
  }

  async carryForwardResponses(questionnaireId: string, sourceYear: number): Promise<void> {
    await this.client.post(`/cdp/historical/carry-forward`, {
      questionnaire_id: questionnaireId,
      source_year: sourceYear,
    });
  }
}

// ---------------------------------------------------------------------------
// Singleton export
// ---------------------------------------------------------------------------

export const cdpApi = new CDPApiClient();
export default cdpApi;
