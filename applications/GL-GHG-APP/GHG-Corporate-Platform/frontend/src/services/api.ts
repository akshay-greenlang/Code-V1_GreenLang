/**
 * GL-GHG Corporate Platform - API Client Service
 *
 * Axios-based HTTP client for communicating with the GL-GHG backend.
 * Provides 57 typed methods across 9 route groups:
 *   1. Inventory    (8 methods)
 *   2. Scope 1      (6 methods)
 *   3. Scope 2      (7 methods)
 *   4. Scope 3      (8 methods)
 *   5. Reporting     (7 methods)
 *   6. Dashboard     (5 methods)
 *   7. Verification  (7 methods)
 *   8. Targets       (6 methods)
 *   9. Settings      (3 methods)
 *
 * Features:
 *   - JWT bearer token injection via request interceptor
 *   - Automatic 401 redirect to login
 *   - Token refresh on expiry
 *   - Typed request/response contracts
 */

import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import type {
  Organization,
  Entity,
  InventoryBoundary,
  BaseYear,
  Recalculation,
  GHGInventory,
  ScopeEmissions,
  Scope1Summary,
  Scope1CategoryBreakdown,
  Scope2Summary,
  ReconciliationData,
  Scope3Summary,
  Scope3CategoryBreakdown,
  MaterialityResult,
  DashboardMetrics,
  TrendDataPoint,
  ScopeBreakdown,
  DashboardAlert,
  Report,
  Disclosure,
  CompletenessResult,
  ExportResult,
  VerificationRecord,
  Finding,
  Target,
  TargetProgress,
  SBTiAlignmentCheck,
  SettingsResponse,
  ApiResponse,
  PaginatedResponse,
  AggregationResult,
  CreateOrganizationRequest,
  AddEntityRequest,
  SetBoundaryRequest,
  CreateInventoryRequest,
  SubmitScope1DataRequest,
  SubmitScope2DataRequest,
  SubmitScope3DataRequest,
  GenerateReportRequest,
  ExportDataRequest,
  SetTargetRequest,
  StartVerificationRequest,
  AddFindingRequest,
  UpdateSettingsRequest,
  Scope,
} from '../types';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';
const TOKEN_KEY = 'ghg_access_token';
const REFRESH_KEY = 'ghg_refresh_token';

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

class GHGApiClient {
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
  // 1. INVENTORY ROUTES (8 methods)
  // =========================================================================

  /** Create a new organization profile. */
  async createOrganization(payload: CreateOrganizationRequest): Promise<Organization> {
    const { data } = await this.client.post<ApiResponse<Organization>>(
      '/inventory/organizations',
      payload,
    );
    return data.data;
  }

  /** Get organization by ID. */
  async getOrganization(orgId: string): Promise<Organization> {
    const { data } = await this.client.get<ApiResponse<Organization>>(
      `/inventory/organizations/${orgId}`,
    );
    return data.data;
  }

  /** Add a reporting entity under an organization. */
  async addEntity(orgId: string, payload: AddEntityRequest): Promise<Entity> {
    const { data } = await this.client.post<ApiResponse<Entity>>(
      `/inventory/organizations/${orgId}/entities`,
      payload,
    );
    return data.data;
  }

  /** Get all entities for an organization. */
  async getEntities(orgId: string): Promise<Entity[]> {
    const { data } = await this.client.get<ApiResponse<Entity[]>>(
      `/inventory/organizations/${orgId}/entities`,
    );
    return data.data;
  }

  /** Set the inventory boundary for a reporting year. */
  async setBoundary(orgId: string, payload: SetBoundaryRequest): Promise<InventoryBoundary> {
    const { data } = await this.client.post<ApiResponse<InventoryBoundary>>(
      `/inventory/organizations/${orgId}/boundary`,
      payload,
    );
    return data.data;
  }

  /** Create a new GHG inventory for a reporting year. */
  async createInventory(payload: CreateInventoryRequest): Promise<GHGInventory> {
    const { data } = await this.client.post<ApiResponse<GHGInventory>>(
      '/inventory/inventories',
      payload,
    );
    return data.data;
  }

  /** Get a GHG inventory by ID. */
  async getInventory(inventoryId: string): Promise<GHGInventory> {
    const { data } = await this.client.get<ApiResponse<GHGInventory>>(
      `/inventory/inventories/${inventoryId}`,
    );
    return data.data;
  }

  /** Get base year configuration and recalculation history. */
  async getBaseYear(orgId: string): Promise<BaseYear> {
    const { data } = await this.client.get<ApiResponse<BaseYear>>(
      `/inventory/organizations/${orgId}/base-year`,
    );
    return data.data;
  }

  // =========================================================================
  // 2. SCOPE 1 ROUTES (6 methods)
  // =========================================================================

  /** Aggregate Scope 1 emissions across all entities. */
  async aggregateScope1(inventoryId: string): Promise<AggregationResult> {
    const { data } = await this.client.post<ApiResponse<AggregationResult>>(
      `/scope1/${inventoryId}/aggregate`,
    );
    return data.data;
  }

  /** Get Scope 1 summary with category and entity breakdowns. */
  async getScope1Summary(inventoryId: string): Promise<Scope1Summary> {
    const { data } = await this.client.get<ApiResponse<Scope1Summary>>(
      `/scope1/${inventoryId}/summary`,
    );
    return data.data;
  }

  /** Get Scope 1 emissions broken down by source category. */
  async getScope1Categories(inventoryId: string): Promise<Scope1CategoryBreakdown[]> {
    const { data } = await this.client.get<ApiResponse<Scope1CategoryBreakdown[]>>(
      `/scope1/${inventoryId}/categories`,
    );
    return data.data;
  }

  /** Submit activity data for a Scope 1 emission source. */
  async submitScope1Data(inventoryId: string, payload: SubmitScope1DataRequest): Promise<ScopeEmissions> {
    const { data } = await this.client.post<ApiResponse<ScopeEmissions>>(
      `/scope1/${inventoryId}/data`,
      payload,
    );
    return data.data;
  }

  /** Get Scope 1 gas breakdown (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3). */
  async getScope1GasBreakdown(inventoryId: string): Promise<Record<string, number>> {
    const { data } = await this.client.get<ApiResponse<Record<string, number>>>(
      `/scope1/${inventoryId}/gas-breakdown`,
    );
    return data.data;
  }

  /** Get Scope 1 emissions by entity. */
  async getScope1ByEntity(inventoryId: string): Promise<Entity[]> {
    const { data } = await this.client.get<ApiResponse<Entity[]>>(
      `/scope1/${inventoryId}/by-entity`,
    );
    return data.data;
  }

  // =========================================================================
  // 3. SCOPE 2 ROUTES (7 methods)
  // =========================================================================

  /** Aggregate Scope 2 emissions (location and market-based). */
  async aggregateScope2(inventoryId: string): Promise<AggregationResult> {
    const { data } = await this.client.post<ApiResponse<AggregationResult>>(
      `/scope2/${inventoryId}/aggregate`,
    );
    return data.data;
  }

  /** Get Scope 2 summary with dual reporting totals. */
  async getScope2Summary(inventoryId: string): Promise<Scope2Summary> {
    const { data } = await this.client.get<ApiResponse<Scope2Summary>>(
      `/scope2/${inventoryId}/summary`,
    );
    return data.data;
  }

  /** Get location-based Scope 2 emissions by grid region. */
  async getScope2LocationBased(inventoryId: string): Promise<ScopeEmissions> {
    const { data } = await this.client.get<ApiResponse<ScopeEmissions>>(
      `/scope2/${inventoryId}/location-based`,
    );
    return data.data;
  }

  /** Get market-based Scope 2 emissions with contractual instruments. */
  async getScope2MarketBased(inventoryId: string): Promise<ScopeEmissions> {
    const { data } = await this.client.get<ApiResponse<ScopeEmissions>>(
      `/scope2/${inventoryId}/market-based`,
    );
    return data.data;
  }

  /** Get location vs. market-based reconciliation. */
  async getScope2Reconciliation(inventoryId: string): Promise<ReconciliationData> {
    const { data } = await this.client.get<ApiResponse<ReconciliationData>>(
      `/scope2/${inventoryId}/reconciliation`,
    );
    return data.data;
  }

  /** Submit Scope 2 activity data (electricity, steam, heat, cooling). */
  async submitScope2Data(inventoryId: string, payload: SubmitScope2DataRequest): Promise<ScopeEmissions> {
    const { data } = await this.client.post<ApiResponse<ScopeEmissions>>(
      `/scope2/${inventoryId}/data`,
      payload,
    );
    return data.data;
  }

  /** Get Scope 2 emissions by energy type. */
  async getScope2ByEnergyType(inventoryId: string): Promise<Scope2Summary> {
    const { data } = await this.client.get<ApiResponse<Scope2Summary>>(
      `/scope2/${inventoryId}/by-energy-type`,
    );
    return data.data;
  }

  // =========================================================================
  // 4. SCOPE 3 ROUTES (8 methods)
  // =========================================================================

  /** Aggregate Scope 3 emissions across all 15 categories. */
  async aggregateScope3(inventoryId: string): Promise<AggregationResult> {
    const { data } = await this.client.post<ApiResponse<AggregationResult>>(
      `/scope3/${inventoryId}/aggregate`,
    );
    return data.data;
  }

  /** Get Scope 3 summary with upstream/downstream breakdown. */
  async getScope3Summary(inventoryId: string): Promise<Scope3Summary> {
    const { data } = await this.client.get<ApiResponse<Scope3Summary>>(
      `/scope3/${inventoryId}/summary`,
    );
    return data.data;
  }

  /** Get all Scope 3 category breakdowns. */
  async getScope3Categories(inventoryId: string): Promise<Scope3CategoryBreakdown[]> {
    const { data } = await this.client.get<ApiResponse<Scope3CategoryBreakdown[]>>(
      `/scope3/${inventoryId}/categories`,
    );
    return data.data;
  }

  /** Get detailed data for a specific Scope 3 category. */
  async getScope3CategoryDetail(inventoryId: string, categoryKey: string): Promise<Scope3CategoryBreakdown> {
    const { data } = await this.client.get<ApiResponse<Scope3CategoryBreakdown>>(
      `/scope3/${inventoryId}/categories/${categoryKey}`,
    );
    return data.data;
  }

  /** Submit activity data for a Scope 3 category. */
  async submitScope3Data(inventoryId: string, payload: SubmitScope3DataRequest): Promise<ScopeEmissions> {
    const { data } = await this.client.post<ApiResponse<ScopeEmissions>>(
      `/scope3/${inventoryId}/data`,
      payload,
    );
    return data.data;
  }

  /** Get materiality assessment for Scope 3 categories. */
  async getScope3Materiality(inventoryId: string): Promise<MaterialityResult[]> {
    const { data } = await this.client.get<ApiResponse<MaterialityResult[]>>(
      `/scope3/${inventoryId}/materiality`,
    );
    return data.data;
  }

  /** Get upstream-only Scope 3 emissions (Categories 1-8). */
  async getScope3Upstream(inventoryId: string): Promise<Scope3CategoryBreakdown[]> {
    const { data } = await this.client.get<ApiResponse<Scope3CategoryBreakdown[]>>(
      `/scope3/${inventoryId}/upstream`,
    );
    return data.data;
  }

  /** Get downstream-only Scope 3 emissions (Categories 9-15). */
  async getScope3Downstream(inventoryId: string): Promise<Scope3CategoryBreakdown[]> {
    const { data } = await this.client.get<ApiResponse<Scope3CategoryBreakdown[]>>(
      `/scope3/${inventoryId}/downstream`,
    );
    return data.data;
  }

  // =========================================================================
  // 5. REPORTING ROUTES (7 methods)
  // =========================================================================

  /** Generate a GHG Protocol compliant report. */
  async generateReport(payload: GenerateReportRequest): Promise<Report> {
    const { data } = await this.client.post<ApiResponse<Report>>(
      '/reports/generate',
      payload,
    );
    return data.data;
  }

  /** Get all reports for an inventory. */
  async getReports(inventoryId: string): Promise<Report[]> {
    const { data } = await this.client.get<ApiResponse<Report[]>>(
      `/reports/inventory/${inventoryId}`,
    );
    return data.data;
  }

  /** Get report by ID. */
  async getReport(reportId: string): Promise<Report> {
    const { data } = await this.client.get<ApiResponse<Report>>(
      `/reports/${reportId}`,
    );
    return data.data;
  }

  /** Download a generated report file. */
  async downloadReport(reportId: string): Promise<Blob> {
    const { data } = await this.client.get(`/reports/${reportId}/download`, {
      responseType: 'blob',
    });
    return data;
  }

  /** Get disclosure checklist and completion status. */
  async getDisclosures(inventoryId: string): Promise<Disclosure[]> {
    const { data } = await this.client.get<ApiResponse<Disclosure[]>>(
      `/reports/inventory/${inventoryId}/disclosures`,
    );
    return data.data;
  }

  /** Get completeness check results. */
  async getCompleteness(inventoryId: string): Promise<CompletenessResult> {
    const { data } = await this.client.get<ApiResponse<CompletenessResult>>(
      `/reports/inventory/${inventoryId}/completeness`,
    );
    return data.data;
  }

  /** Export inventory data in specified format. */
  async exportData(payload: ExportDataRequest): Promise<ExportResult> {
    const { data } = await this.client.post<ApiResponse<ExportResult>>(
      '/reports/export',
      payload,
    );
    return data.data;
  }

  // =========================================================================
  // 6. DASHBOARD ROUTES (5 methods)
  // =========================================================================

  /** Get executive dashboard metrics. */
  async getDashboardMetrics(orgId: string, reportingYear: number): Promise<DashboardMetrics> {
    const { data } = await this.client.get<ApiResponse<DashboardMetrics>>(
      `/dashboard/${orgId}/metrics`,
      { params: { reporting_year: reportingYear } },
    );
    return data.data;
  }

  /** Get emissions trend data over time. */
  async getTrendData(
    orgId: string,
    startYear: number,
    endYear: number,
    granularity: 'monthly' | 'quarterly' | 'yearly' = 'yearly',
  ): Promise<TrendDataPoint[]> {
    const { data } = await this.client.get<ApiResponse<TrendDataPoint[]>>(
      `/dashboard/${orgId}/trends`,
      { params: { start_year: startYear, end_year: endYear, granularity } },
    );
    return data.data;
  }

  /** Get scope breakdown for a specific inventory. */
  async getScopeBreakdown(inventoryId: string, scope: Scope): Promise<ScopeBreakdown> {
    const { data } = await this.client.get<ApiResponse<ScopeBreakdown>>(
      `/dashboard/inventories/${inventoryId}/breakdown`,
      { params: { scope } },
    );
    return data.data;
  }

  /** Get active alerts for the dashboard. */
  async getDashboardAlerts(orgId: string): Promise<DashboardAlert[]> {
    const { data } = await this.client.get<ApiResponse<DashboardAlert[]>>(
      `/dashboard/${orgId}/alerts`,
    );
    return data.data;
  }

  /** Mark an alert as read. */
  async markAlertRead(alertId: string): Promise<void> {
    await this.client.put(`/dashboard/alerts/${alertId}/read`);
  }

  // =========================================================================
  // 7. VERIFICATION ROUTES (7 methods)
  // =========================================================================

  /** Start a verification engagement. */
  async startVerification(payload: StartVerificationRequest): Promise<VerificationRecord> {
    const { data } = await this.client.post<ApiResponse<VerificationRecord>>(
      '/verification/start',
      payload,
    );
    return data.data;
  }

  /** Get verification record by ID. */
  async getVerification(verificationId: string): Promise<VerificationRecord> {
    const { data } = await this.client.get<ApiResponse<VerificationRecord>>(
      `/verification/${verificationId}`,
    );
    return data.data;
  }

  /** Get all verification records for an inventory. */
  async getVerifications(inventoryId: string): Promise<VerificationRecord[]> {
    const { data } = await this.client.get<ApiResponse<VerificationRecord[]>>(
      `/verification/inventory/${inventoryId}`,
    );
    return data.data;
  }

  /** Approve (issue positive opinion on) a verification. */
  async approveVerification(verificationId: string, opinion: string): Promise<VerificationRecord> {
    const { data } = await this.client.post<ApiResponse<VerificationRecord>>(
      `/verification/${verificationId}/approve`,
      { opinion },
    );
    return data.data;
  }

  /** Reject a verification. */
  async rejectVerification(verificationId: string, reason: string): Promise<VerificationRecord> {
    const { data } = await this.client.post<ApiResponse<VerificationRecord>>(
      `/verification/${verificationId}/reject`,
      { reason },
    );
    return data.data;
  }

  /** Add a finding to a verification. */
  async addFinding(verificationId: string, payload: AddFindingRequest): Promise<Finding> {
    const { data } = await this.client.post<ApiResponse<Finding>>(
      `/verification/${verificationId}/findings`,
      payload,
    );
    return data.data;
  }

  /** Resolve a finding. */
  async resolveFinding(
    verificationId: string,
    findingId: string,
    resolution: string,
  ): Promise<Finding> {
    const { data } = await this.client.post<ApiResponse<Finding>>(
      `/verification/${verificationId}/findings/${findingId}/resolve`,
      { resolution },
    );
    return data.data;
  }

  // =========================================================================
  // 8. TARGETS ROUTES (6 methods)
  // =========================================================================

  /** Set a new emission reduction target. */
  async setTarget(orgId: string, payload: SetTargetRequest): Promise<Target> {
    const { data } = await this.client.post<ApiResponse<Target>>(
      `/targets/${orgId}`,
      payload,
    );
    return data.data;
  }

  /** Get all targets for an organization. */
  async getTargets(orgId: string): Promise<Target[]> {
    const { data } = await this.client.get<ApiResponse<Target[]>>(
      `/targets/${orgId}`,
    );
    return data.data;
  }

  /** Get a specific target by ID. */
  async getTarget(orgId: string, targetId: string): Promise<Target> {
    const { data } = await this.client.get<ApiResponse<Target>>(
      `/targets/${orgId}/${targetId}`,
    );
    return data.data;
  }

  /** Get progress tracking for a target. */
  async getTargetProgress(orgId: string, targetId: string): Promise<TargetProgress> {
    const { data } = await this.client.get<ApiResponse<TargetProgress>>(
      `/targets/${orgId}/${targetId}/progress`,
    );
    return data.data;
  }

  /** Check SBTi alignment for a target. */
  async checkSBTiAlignment(orgId: string, targetId: string): Promise<SBTiAlignmentCheck> {
    const { data } = await this.client.get<ApiResponse<SBTiAlignmentCheck>>(
      `/targets/${orgId}/${targetId}/sbti-check`,
    );
    return data.data;
  }

  /** Delete a target. */
  async deleteTarget(orgId: string, targetId: string): Promise<void> {
    await this.client.delete(`/targets/${orgId}/${targetId}`);
  }

  // =========================================================================
  // 9. SETTINGS ROUTES (3 methods)
  // =========================================================================

  /** Get organization settings. */
  async getSettings(orgId: string): Promise<SettingsResponse> {
    const { data } = await this.client.get<ApiResponse<SettingsResponse>>(
      `/settings/${orgId}`,
    );
    return data.data;
  }

  /** Update organization settings. */
  async updateSettings(orgId: string, payload: UpdateSettingsRequest): Promise<SettingsResponse> {
    const { data } = await this.client.put<ApiResponse<SettingsResponse>>(
      `/settings/${orgId}`,
      payload,
    );
    return data.data;
  }

  /** Reset settings to defaults. */
  async resetSettings(orgId: string): Promise<SettingsResponse> {
    const { data } = await this.client.post<ApiResponse<SettingsResponse>>(
      `/settings/${orgId}/reset`,
    );
    return data.data;
  }
}

// ---------------------------------------------------------------------------
// Singleton export
// ---------------------------------------------------------------------------

export const ghgApi = new GHGApiClient();
export default ghgApi;
