/**
 * GL-ISO14064-APP v1.0 - API Client Service
 *
 * Axios-based HTTP client for communicating with the ISO 14064-1 backend.
 * Provides 65+ typed methods across 12 route groups:
 *   1.  Organization    (6 methods)
 *   2.  Entity          (5 methods)
 *   3.  Boundary        (4 methods)
 *   4.  Inventory       (7 methods)
 *   5.  Emissions       (7 methods)
 *   6.  Removals        (5 methods)
 *   7.  Significance    (4 methods)
 *   8.  Verification    (8 methods)
 *   9.  Reports         (6 methods)
 *   10. Management      (7 methods)
 *   11. Crosswalk       (3 methods)
 *   12. Dashboard       (5 methods)
 *
 * Features:
 *   - JWT bearer token injection via request interceptor
 *   - Automatic 401 redirect to login
 *   - Token refresh on expiry with failed-queue replay
 *   - Typed request/response contracts
 */

import axios, { AxiosInstance, AxiosError, InternalAxiosRequestConfig } from 'axios';
import type {
  Organization,
  Entity,
  OrganizationalBoundary,
  OperationalBoundary,
  ISOInventory,
  InventoryTotals,
  CategoryResult,
  EmissionSource,
  QuantificationResult,
  DataQualityIndicator,
  RemovalSource,
  SignificanceAssessment,
  UncertaintyResult,
  BaseYearRecord,
  BaseYearTrigger,
  VerificationRecord,
  Finding,
  ISOReport,
  MandatoryElement,
  ManagementAction,
  ManagementPlan,
  QualityManagementPlan,
  CrosswalkResult,
  DashboardMetrics,
  TrendDataPoint,
  CategoryBreakdownItem,
  DashboardAlert,
  ExportResult,
  ApiResponse,
  PaginatedResponse,
  CreateOrganizationRequest,
  AddEntityRequest,
  UpdateEntityRequest,
  CreateInventoryRequest,
  AddEmissionSourceRequest,
  AddRemovalSourceRequest,
  SetBaseYearRequest,
  RecalculateBaseYearRequest,
  AddFindingRequest,
  CreateVerificationRequest,
  CreateManagementActionRequest,
  GenerateReportRequest,
  ExportDataRequest,
  SetOrganizationalBoundaryRequest,
  SetOperationalBoundaryRequest,
  ISOCategory,
} from '../types';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';
const TOKEN_KEY = 'iso14064_access_token';
const REFRESH_KEY = 'iso14064_refresh_token';

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

class ISO14064ApiClient {
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
  // 1. ORGANIZATION ROUTES (6 methods)
  // =========================================================================

  /** Create a new organization profile. */
  async createOrganization(payload: CreateOrganizationRequest): Promise<Organization> {
    const { data } = await this.client.post<ApiResponse<Organization>>(
      '/organizations',
      payload,
    );
    return data.data;
  }

  /** Get organization by ID. */
  async getOrganization(orgId: string): Promise<Organization> {
    const { data } = await this.client.get<ApiResponse<Organization>>(
      `/organizations/${orgId}`,
    );
    return data.data;
  }

  /** Update organization details. */
  async updateOrganization(orgId: string, payload: Partial<CreateOrganizationRequest>): Promise<Organization> {
    const { data } = await this.client.put<ApiResponse<Organization>>(
      `/organizations/${orgId}`,
      payload,
    );
    return data.data;
  }

  /** Delete an organization. */
  async deleteOrganization(orgId: string): Promise<void> {
    await this.client.delete(`/organizations/${orgId}`);
  }

  /** List all organizations. */
  async listOrganizations(): Promise<Organization[]> {
    const { data } = await this.client.get<ApiResponse<Organization[]>>(
      '/organizations',
    );
    return data.data;
  }

  /** Search organizations by name or industry. */
  async searchOrganizations(query: string): Promise<Organization[]> {
    const { data } = await this.client.get<ApiResponse<Organization[]>>(
      '/organizations/search',
      { params: { q: query } },
    );
    return data.data;
  }

  // =========================================================================
  // 2. ENTITY ROUTES (5 methods)
  // =========================================================================

  /** Add a reporting entity under an organization. */
  async addEntity(orgId: string, payload: AddEntityRequest): Promise<Entity> {
    const { data } = await this.client.post<ApiResponse<Entity>>(
      `/organizations/${orgId}/entities`,
      payload,
    );
    return data.data;
  }

  /** Get all entities for an organization. */
  async getEntities(orgId: string): Promise<Entity[]> {
    const { data } = await this.client.get<ApiResponse<Entity[]>>(
      `/organizations/${orgId}/entities`,
    );
    return data.data;
  }

  /** Get a single entity by ID. */
  async getEntity(orgId: string, entityId: string): Promise<Entity> {
    const { data } = await this.client.get<ApiResponse<Entity>>(
      `/organizations/${orgId}/entities/${entityId}`,
    );
    return data.data;
  }

  /** Update an entity. */
  async updateEntity(orgId: string, entityId: string, payload: UpdateEntityRequest): Promise<Entity> {
    const { data } = await this.client.put<ApiResponse<Entity>>(
      `/organizations/${orgId}/entities/${entityId}`,
      payload,
    );
    return data.data;
  }

  /** Delete an entity. */
  async deleteEntity(orgId: string, entityId: string): Promise<void> {
    await this.client.delete(`/organizations/${orgId}/entities/${entityId}`);
  }

  // =========================================================================
  // 3. BOUNDARY ROUTES (4 methods)
  // =========================================================================

  /** Set the organizational boundary (consolidation approach + entities). */
  async setOrganizationalBoundary(orgId: string, payload: SetOrganizationalBoundaryRequest): Promise<OrganizationalBoundary> {
    const { data } = await this.client.post<ApiResponse<OrganizationalBoundary>>(
      `/organizations/${orgId}/boundary/organizational`,
      payload,
    );
    return data.data;
  }

  /** Get the organizational boundary. */
  async getOrganizationalBoundary(orgId: string): Promise<OrganizationalBoundary> {
    const { data } = await this.client.get<ApiResponse<OrganizationalBoundary>>(
      `/organizations/${orgId}/boundary/organizational`,
    );
    return data.data;
  }

  /** Set the operational boundary (category inclusions + significance). */
  async setOperationalBoundary(orgId: string, payload: SetOperationalBoundaryRequest): Promise<OperationalBoundary> {
    const { data } = await this.client.post<ApiResponse<OperationalBoundary>>(
      `/organizations/${orgId}/boundary/operational`,
      payload,
    );
    return data.data;
  }

  /** Get the operational boundary. */
  async getOperationalBoundary(orgId: string): Promise<OperationalBoundary> {
    const { data } = await this.client.get<ApiResponse<OperationalBoundary>>(
      `/organizations/${orgId}/boundary/operational`,
    );
    return data.data;
  }

  // =========================================================================
  // 4. INVENTORY ROUTES (7 methods)
  // =========================================================================

  /** Create a new ISO 14064-1 inventory for a reporting year. */
  async createInventory(payload: CreateInventoryRequest): Promise<ISOInventory> {
    const { data } = await this.client.post<ApiResponse<ISOInventory>>(
      '/inventories',
      payload,
    );
    return data.data;
  }

  /** Get an inventory by ID. */
  async getInventory(inventoryId: string): Promise<ISOInventory> {
    const { data } = await this.client.get<ApiResponse<ISOInventory>>(
      `/inventories/${inventoryId}`,
    );
    return data.data;
  }

  /** List all inventories for an organization. */
  async listInventories(orgId: string): Promise<ISOInventory[]> {
    const { data } = await this.client.get<ApiResponse<ISOInventory[]>>(
      `/inventories`,
      { params: { org_id: orgId } },
    );
    return data.data;
  }

  /** Get inventory totals (grand totals across all categories). */
  async getInventoryTotals(inventoryId: string): Promise<InventoryTotals> {
    const { data } = await this.client.get<ApiResponse<InventoryTotals>>(
      `/inventories/${inventoryId}/totals`,
    );
    return data.data;
  }

  /** Get category-level results for an inventory. */
  async getCategoryResults(inventoryId: string): Promise<CategoryResult[]> {
    const { data } = await this.client.get<ApiResponse<CategoryResult[]>>(
      `/inventories/${inventoryId}/categories`,
    );
    return data.data;
  }

  /** Set base year for an organization. */
  async setBaseYear(orgId: string, payload: SetBaseYearRequest): Promise<BaseYearRecord> {
    const { data } = await this.client.post<ApiResponse<BaseYearRecord>>(
      `/inventories/organizations/${orgId}/base-year`,
      payload,
    );
    return data.data;
  }

  /** Trigger base year recalculation. */
  async recalculateBaseYear(orgId: string, payload: RecalculateBaseYearRequest): Promise<BaseYearTrigger> {
    const { data } = await this.client.post<ApiResponse<BaseYearTrigger>>(
      `/inventories/organizations/${orgId}/base-year/recalculate`,
      payload,
    );
    return data.data;
  }

  // =========================================================================
  // 5. EMISSIONS ROUTES (7 methods)
  // =========================================================================

  /** Add an emission source to an inventory. */
  async addEmissionSource(inventoryId: string, payload: AddEmissionSourceRequest): Promise<EmissionSource> {
    const { data } = await this.client.post<ApiResponse<EmissionSource>>(
      `/inventories/${inventoryId}/emissions`,
      payload,
    );
    return data.data;
  }

  /** Get all emission sources for an inventory. */
  async getEmissionSources(inventoryId: string): Promise<EmissionSource[]> {
    const { data } = await this.client.get<ApiResponse<EmissionSource[]>>(
      `/inventories/${inventoryId}/emissions`,
    );
    return data.data;
  }

  /** Get emission sources filtered by category. */
  async getEmissionSourcesByCategory(inventoryId: string, category: ISOCategory): Promise<EmissionSource[]> {
    const { data } = await this.client.get<ApiResponse<EmissionSource[]>>(
      `/inventories/${inventoryId}/emissions`,
      { params: { category } },
    );
    return data.data;
  }

  /** Delete an emission source. */
  async deleteEmissionSource(inventoryId: string, sourceId: string): Promise<void> {
    await this.client.delete(`/inventories/${inventoryId}/emissions/${sourceId}`);
  }

  /** Quantify (calculate) emissions for a source. */
  async quantifyEmissions(inventoryId: string, sourceId: string): Promise<QuantificationResult> {
    const { data } = await this.client.post<ApiResponse<QuantificationResult>>(
      `/inventories/${inventoryId}/emissions/${sourceId}/quantify`,
    );
    return data.data;
  }

  /** Get data quality indicators for an inventory. */
  async getDataQuality(inventoryId: string): Promise<DataQualityIndicator> {
    const { data } = await this.client.get<ApiResponse<DataQualityIndicator>>(
      `/inventories/${inventoryId}/data-quality`,
    );
    return data.data;
  }

  /** Run uncertainty analysis for an inventory. */
  async runUncertaintyAnalysis(inventoryId: string): Promise<UncertaintyResult> {
    const { data } = await this.client.post<ApiResponse<UncertaintyResult>>(
      `/inventories/${inventoryId}/uncertainty`,
    );
    return data.data;
  }

  // =========================================================================
  // 6. REMOVALS ROUTES (5 methods)
  // =========================================================================

  /** Add a removal source to an inventory. */
  async addRemovalSource(inventoryId: string, payload: AddRemovalSourceRequest): Promise<RemovalSource> {
    const { data } = await this.client.post<ApiResponse<RemovalSource>>(
      `/inventories/${inventoryId}/removals`,
      payload,
    );
    return data.data;
  }

  /** Get all removal sources for an inventory. */
  async getRemovalSources(inventoryId: string): Promise<RemovalSource[]> {
    const { data } = await this.client.get<ApiResponse<RemovalSource[]>>(
      `/inventories/${inventoryId}/removals`,
    );
    return data.data;
  }

  /** Get a single removal source. */
  async getRemovalSource(inventoryId: string, removalId: string): Promise<RemovalSource> {
    const { data } = await this.client.get<ApiResponse<RemovalSource>>(
      `/inventories/${inventoryId}/removals/${removalId}`,
    );
    return data.data;
  }

  /** Update a removal source. */
  async updateRemovalSource(inventoryId: string, removalId: string, payload: Partial<AddRemovalSourceRequest>): Promise<RemovalSource> {
    const { data } = await this.client.put<ApiResponse<RemovalSource>>(
      `/inventories/${inventoryId}/removals/${removalId}`,
      payload,
    );
    return data.data;
  }

  /** Delete a removal source. */
  async deleteRemovalSource(inventoryId: string, removalId: string): Promise<void> {
    await this.client.delete(`/inventories/${inventoryId}/removals/${removalId}`);
  }

  // =========================================================================
  // 7. SIGNIFICANCE ROUTES (4 methods)
  // =========================================================================

  /** Run significance assessment for an indirect category. */
  async assessSignificance(inventoryId: string, category: ISOCategory): Promise<SignificanceAssessment> {
    const { data } = await this.client.post<ApiResponse<SignificanceAssessment>>(
      `/inventories/${inventoryId}/significance`,
      { category },
    );
    return data.data;
  }

  /** Get all significance assessments for an inventory. */
  async getSignificanceAssessments(inventoryId: string): Promise<SignificanceAssessment[]> {
    const { data } = await this.client.get<ApiResponse<SignificanceAssessment[]>>(
      `/inventories/${inventoryId}/significance`,
    );
    return data.data;
  }

  /** Get significance assessment for a specific category. */
  async getSignificanceForCategory(inventoryId: string, category: ISOCategory): Promise<SignificanceAssessment> {
    const { data } = await this.client.get<ApiResponse<SignificanceAssessment>>(
      `/inventories/${inventoryId}/significance/${category}`,
    );
    return data.data;
  }

  /** Update significance assessment. */
  async updateSignificanceAssessment(inventoryId: string, assessmentId: string, payload: Partial<SignificanceAssessment>): Promise<SignificanceAssessment> {
    const { data } = await this.client.put<ApiResponse<SignificanceAssessment>>(
      `/inventories/${inventoryId}/significance/${assessmentId}`,
      payload,
    );
    return data.data;
  }

  // =========================================================================
  // 8. VERIFICATION ROUTES (8 methods)
  // =========================================================================

  /** Start a verification engagement. */
  async startVerification(payload: CreateVerificationRequest): Promise<VerificationRecord> {
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

  /** Advance verification to next stage. */
  async advanceVerificationStage(verificationId: string): Promise<VerificationRecord> {
    const { data } = await this.client.post<ApiResponse<VerificationRecord>>(
      `/verification/${verificationId}/advance`,
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
  // 9. REPORTS ROUTES (6 methods)
  // =========================================================================

  /** Generate an ISO 14064-1 compliant report. */
  async generateReport(payload: GenerateReportRequest): Promise<ISOReport> {
    const { data } = await this.client.post<ApiResponse<ISOReport>>(
      '/reports/generate',
      payload,
    );
    return data.data;
  }

  /** Get all reports for an inventory. */
  async getReports(inventoryId: string): Promise<ISOReport[]> {
    const { data } = await this.client.get<ApiResponse<ISOReport[]>>(
      `/reports/inventory/${inventoryId}`,
    );
    return data.data;
  }

  /** Get report by ID. */
  async getReport(reportId: string): Promise<ISOReport> {
    const { data } = await this.client.get<ApiResponse<ISOReport>>(
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

  /** Get mandatory element completion status for an inventory. */
  async getMandatoryElements(inventoryId: string): Promise<MandatoryElement[]> {
    const { data } = await this.client.get<ApiResponse<MandatoryElement[]>>(
      `/reports/inventory/${inventoryId}/mandatory-elements`,
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
  // 10. MANAGEMENT PLAN ROUTES (7 methods)
  // =========================================================================

  /** Get management plan for an organization-year. */
  async getManagementPlan(orgId: string, reportingYear: number): Promise<ManagementPlan> {
    const { data } = await this.client.get<ApiResponse<ManagementPlan>>(
      `/management/${orgId}/plan`,
      { params: { reporting_year: reportingYear } },
    );
    return data.data;
  }

  /** Create or update the management plan. */
  async upsertManagementPlan(orgId: string, payload: Partial<ManagementPlan>): Promise<ManagementPlan> {
    const { data } = await this.client.put<ApiResponse<ManagementPlan>>(
      `/management/${orgId}/plan`,
      payload,
    );
    return data.data;
  }

  /** Add a management action. */
  async addManagementAction(orgId: string, payload: CreateManagementActionRequest): Promise<ManagementAction> {
    const { data } = await this.client.post<ApiResponse<ManagementAction>>(
      `/management/${orgId}/actions`,
      payload,
    );
    return data.data;
  }

  /** Get all management actions for an organization. */
  async getManagementActions(orgId: string): Promise<ManagementAction[]> {
    const { data } = await this.client.get<ApiResponse<ManagementAction[]>>(
      `/management/${orgId}/actions`,
    );
    return data.data;
  }

  /** Update a management action. */
  async updateManagementAction(orgId: string, actionId: string, payload: Partial<CreateManagementActionRequest>): Promise<ManagementAction> {
    const { data } = await this.client.put<ApiResponse<ManagementAction>>(
      `/management/${orgId}/actions/${actionId}`,
      payload,
    );
    return data.data;
  }

  /** Delete a management action. */
  async deleteManagementAction(orgId: string, actionId: string): Promise<void> {
    await this.client.delete(`/management/${orgId}/actions/${actionId}`);
  }

  /** Get quality management plan. */
  async getQualityManagementPlan(orgId: string): Promise<QualityManagementPlan> {
    const { data } = await this.client.get<ApiResponse<QualityManagementPlan>>(
      `/management/${orgId}/quality`,
    );
    return data.data;
  }

  // =========================================================================
  // 11. CROSSWALK ROUTES (3 methods)
  // =========================================================================

  /** Generate ISO 14064-1 to GHG Protocol crosswalk. */
  async generateCrosswalk(inventoryId: string): Promise<CrosswalkResult> {
    const { data } = await this.client.post<ApiResponse<CrosswalkResult>>(
      `/crosswalk/${inventoryId}/generate`,
    );
    return data.data;
  }

  /** Get existing crosswalk for an inventory. */
  async getCrosswalk(inventoryId: string): Promise<CrosswalkResult> {
    const { data } = await this.client.get<ApiResponse<CrosswalkResult>>(
      `/crosswalk/${inventoryId}`,
    );
    return data.data;
  }

  /** Export crosswalk in specified format. */
  async exportCrosswalk(inventoryId: string, format: string): Promise<ExportResult> {
    const { data } = await this.client.post<ApiResponse<ExportResult>>(
      `/crosswalk/${inventoryId}/export`,
      { format },
    );
    return data.data;
  }

  // =========================================================================
  // 12. DASHBOARD ROUTES (5 methods)
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

  /** Get category breakdown for a specific inventory. */
  async getCategoryBreakdown(inventoryId: string): Promise<CategoryBreakdownItem[]> {
    const { data } = await this.client.get<ApiResponse<CategoryBreakdownItem[]>>(
      `/dashboard/inventories/${inventoryId}/breakdown`,
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
}

// ---------------------------------------------------------------------------
// Singleton export
// ---------------------------------------------------------------------------

export const iso14064Api = new ISO14064ApiClient();
export default iso14064Api;
