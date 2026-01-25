/**
 * GreenLang Agent Factory - API Client
 *
 * Axios-based API client with authentication, interceptors, and error handling.
 */

import axios, {
  AxiosInstance,
  AxiosError,
  InternalAxiosRequestConfig,
} from 'axios';
import type {
  AuthTokens,
  LoginRequest,
  LoginResponse,
  RegisterRequest,
  User,
  Agent,
  AgentLog,
  Tenant,
  UserCreateRequest,
  UserUpdateRequest,
  FuelAnalysisRequest,
  FuelAnalysisResult,
  CBAMCalculationRequest,
  CBAMCalculationResult,
  CBAMReport,
  BuildingEnergyRequest,
  BuildingEnergyResult,
  EUDRComplianceRequest,
  EUDRComplianceResult,
  Report,
  ReportGenerateRequest,
  DashboardMetrics,
  SystemAlert,
  PaginatedResponse,
} from './types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

// Token storage keys
const ACCESS_TOKEN_KEY = 'gl_access_token';
const REFRESH_TOKEN_KEY = 'gl_refresh_token';
const TOKEN_EXPIRY_KEY = 'gl_token_expiry';

/**
 * GreenLang API Client
 */
class GreenLangAPIClient {
  private client: AxiosInstance;
  private isRefreshing = false;
  private refreshSubscribers: Array<(token: string) => void> = [];

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor: Add auth token
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        const token = this.getAccessToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor: Handle token refresh
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config;

        if (error.response?.status === 401 && originalRequest) {
          if (this.isRefreshing) {
            return new Promise((resolve) => {
              this.refreshSubscribers.push((token: string) => {
                originalRequest.headers.Authorization = `Bearer ${token}`;
                resolve(this.client(originalRequest));
              });
            });
          }

          this.isRefreshing = true;

          try {
            const tokens = await this.refreshAccessToken();
            this.setTokens(tokens);
            this.refreshSubscribers.forEach((callback) => callback(tokens.accessToken));
            this.refreshSubscribers = [];

            originalRequest.headers.Authorization = `Bearer ${tokens.accessToken}`;
            return this.client(originalRequest);
          } catch (refreshError) {
            this.clearTokens();
            window.location.href = '/login';
            return Promise.reject(refreshError);
          } finally {
            this.isRefreshing = false;
          }
        }

        return Promise.reject(error);
      }
    );
  }

  // ============================================================================
  // Token Management
  // ============================================================================

  private getAccessToken(): string | null {
    return localStorage.getItem(ACCESS_TOKEN_KEY);
  }

  private getRefreshToken(): string | null {
    return localStorage.getItem(REFRESH_TOKEN_KEY);
  }

  public setTokens(tokens: AuthTokens): void {
    localStorage.setItem(ACCESS_TOKEN_KEY, tokens.accessToken);
    localStorage.setItem(REFRESH_TOKEN_KEY, tokens.refreshToken);
    localStorage.setItem(TOKEN_EXPIRY_KEY, tokens.expiresAt.toString());
  }

  public clearTokens(): void {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
    localStorage.removeItem(REFRESH_TOKEN_KEY);
    localStorage.removeItem(TOKEN_EXPIRY_KEY);
  }

  public isAuthenticated(): boolean {
    const token = this.getAccessToken();
    const expiry = localStorage.getItem(TOKEN_EXPIRY_KEY);

    if (!token || !expiry) return false;
    return Date.now() < parseInt(expiry, 10);
  }

  private async refreshAccessToken(): Promise<AuthTokens> {
    const refreshToken = this.getRefreshToken();
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await axios.post<{ tokens: AuthTokens }>(
      `${API_BASE_URL}/auth/refresh`,
      { refreshToken }
    );

    return response.data.tokens;
  }

  // ============================================================================
  // Auth Endpoints
  // ============================================================================

  async login(credentials: LoginRequest): Promise<LoginResponse> {
    const response = await this.client.post<LoginResponse>('/auth/login', credentials);
    this.setTokens(response.data.tokens);
    return response.data;
  }

  async register(data: RegisterRequest): Promise<LoginResponse> {
    const response = await this.client.post<LoginResponse>('/auth/register', data);
    this.setTokens(response.data.tokens);
    return response.data;
  }

  async logout(): Promise<void> {
    try {
      await this.client.post('/auth/logout');
    } finally {
      this.clearTokens();
    }
  }

  async forgotPassword(email: string): Promise<void> {
    await this.client.post('/auth/forgot-password', { email });
  }

  async resetPassword(token: string, password: string): Promise<void> {
    await this.client.post('/auth/reset-password', { token, password });
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.client.get<User>('/auth/me');
    return response.data;
  }

  // ============================================================================
  // Agent Endpoints
  // ============================================================================

  async getAgents(params?: {
    status?: string;
    type?: string;
    page?: number;
    perPage?: number;
  }): Promise<PaginatedResponse<Agent>> {
    const response = await this.client.get<PaginatedResponse<Agent>>('/agents', { params });
    return response.data;
  }

  async getAgent(agentId: string): Promise<Agent> {
    const response = await this.client.get<Agent>(`/agents/${agentId}`);
    return response.data;
  }

  async updateAgentConfig(
    agentId: string,
    config: Partial<Agent['config']>
  ): Promise<Agent> {
    const response = await this.client.patch<Agent>(`/agents/${agentId}/config`, config);
    return response.data;
  }

  async getAgentLogs(
    agentId: string,
    params?: {
      level?: string;
      startDate?: string;
      endDate?: string;
      page?: number;
      perPage?: number;
    }
  ): Promise<PaginatedResponse<AgentLog>> {
    const response = await this.client.get<PaginatedResponse<AgentLog>>(
      `/agents/${agentId}/logs`,
      { params }
    );
    return response.data;
  }

  async restartAgent(agentId: string): Promise<Agent> {
    const response = await this.client.post<Agent>(`/agents/${agentId}/restart`);
    return response.data;
  }

  async deployAgent(agentId: string, version: string): Promise<Agent> {
    const response = await this.client.post<Agent>(`/agents/${agentId}/deploy`, { version });
    return response.data;
  }

  // ============================================================================
  // User Management Endpoints
  // ============================================================================

  async getUsers(params?: {
    role?: string;
    isActive?: boolean;
    search?: string;
    page?: number;
    perPage?: number;
  }): Promise<PaginatedResponse<User>> {
    const response = await this.client.get<PaginatedResponse<User>>('/users', { params });
    return response.data;
  }

  async getUser(userId: string): Promise<User> {
    const response = await this.client.get<User>(`/users/${userId}`);
    return response.data;
  }

  async createUser(data: UserCreateRequest): Promise<User> {
    const response = await this.client.post<User>('/users', data);
    return response.data;
  }

  async updateUser(userId: string, data: UserUpdateRequest): Promise<User> {
    const response = await this.client.patch<User>(`/users/${userId}`, data);
    return response.data;
  }

  async deleteUser(userId: string): Promise<void> {
    await this.client.delete(`/users/${userId}`);
  }

  async resendInvitation(userId: string): Promise<void> {
    await this.client.post(`/users/${userId}/resend-invitation`);
  }

  // ============================================================================
  // Tenant Management Endpoints
  // ============================================================================

  async getTenants(params?: {
    plan?: string;
    isActive?: boolean;
    search?: string;
    page?: number;
    perPage?: number;
  }): Promise<PaginatedResponse<Tenant>> {
    const response = await this.client.get<PaginatedResponse<Tenant>>('/tenants', { params });
    return response.data;
  }

  async getTenant(tenantId: string): Promise<Tenant> {
    const response = await this.client.get<Tenant>(`/tenants/${tenantId}`);
    return response.data;
  }

  async createTenant(data: Partial<Tenant>): Promise<Tenant> {
    const response = await this.client.post<Tenant>('/tenants', data);
    return response.data;
  }

  async updateTenant(tenantId: string, data: Partial<Tenant>): Promise<Tenant> {
    const response = await this.client.patch<Tenant>(`/tenants/${tenantId}`, data);
    return response.data;
  }

  async deleteTenant(tenantId: string): Promise<void> {
    await this.client.delete(`/tenants/${tenantId}`);
  }

  // ============================================================================
  // Fuel Analysis Endpoints
  // ============================================================================

  async calculateFuelEmissions(data: FuelAnalysisRequest): Promise<FuelAnalysisResult> {
    const response = await this.client.post<FuelAnalysisResult>(
      '/calculations/fuel',
      data
    );
    return response.data;
  }

  async getFuelCalculationHistory(params?: {
    startDate?: string;
    endDate?: string;
    fuelType?: string;
    page?: number;
    perPage?: number;
  }): Promise<PaginatedResponse<FuelAnalysisResult>> {
    const response = await this.client.get<PaginatedResponse<FuelAnalysisResult>>(
      '/calculations/fuel/history',
      { params }
    );
    return response.data;
  }

  // ============================================================================
  // CBAM Endpoints
  // ============================================================================

  async calculateCBAM(data: CBAMCalculationRequest): Promise<CBAMCalculationResult> {
    const response = await this.client.post<CBAMCalculationResult>(
      '/calculations/cbam',
      data
    );
    return response.data;
  }

  async getCBAMHistory(params?: {
    startDate?: string;
    endDate?: string;
    productCategory?: string;
    originCountry?: string;
    page?: number;
    perPage?: number;
  }): Promise<PaginatedResponse<CBAMCalculationResult>> {
    const response = await this.client.get<PaginatedResponse<CBAMCalculationResult>>(
      '/calculations/cbam/history',
      { params }
    );
    return response.data;
  }

  async getCBAMReports(params?: {
    status?: string;
    page?: number;
    perPage?: number;
  }): Promise<PaginatedResponse<CBAMReport>> {
    const response = await this.client.get<PaginatedResponse<CBAMReport>>(
      '/cbam/reports',
      { params }
    );
    return response.data;
  }

  // ============================================================================
  // Building Energy Endpoints
  // ============================================================================

  async calculateBuildingEnergy(
    data: BuildingEnergyRequest
  ): Promise<BuildingEnergyResult> {
    const response = await this.client.post<BuildingEnergyResult>(
      '/calculations/building-energy',
      data
    );
    return response.data;
  }

  async getBuildingEnergyHistory(params?: {
    startDate?: string;
    endDate?: string;
    buildingType?: string;
    page?: number;
    perPage?: number;
  }): Promise<PaginatedResponse<BuildingEnergyResult>> {
    const response = await this.client.get<PaginatedResponse<BuildingEnergyResult>>(
      '/calculations/building-energy/history',
      { params }
    );
    return response.data;
  }

  // ============================================================================
  // EUDR Compliance Endpoints
  // ============================================================================

  async checkEUDRCompliance(
    data: EUDRComplianceRequest
  ): Promise<EUDRComplianceResult> {
    const response = await this.client.post<EUDRComplianceResult>(
      '/compliance/eudr',
      data
    );
    return response.data;
  }

  async getEUDRComplianceHistory(params?: {
    status?: string;
    commodity?: string;
    page?: number;
    perPage?: number;
  }): Promise<PaginatedResponse<EUDRComplianceResult>> {
    const response = await this.client.get<PaginatedResponse<EUDRComplianceResult>>(
      '/compliance/eudr/history',
      { params }
    );
    return response.data;
  }

  // ============================================================================
  // Report Endpoints
  // ============================================================================

  async getReports(params?: {
    type?: string;
    status?: string;
    page?: number;
    perPage?: number;
  }): Promise<PaginatedResponse<Report>> {
    const response = await this.client.get<PaginatedResponse<Report>>('/reports', {
      params,
    });
    return response.data;
  }

  async getReport(reportId: string): Promise<Report> {
    const response = await this.client.get<Report>(`/reports/${reportId}`);
    return response.data;
  }

  async generateReport(data: ReportGenerateRequest): Promise<Report> {
    const response = await this.client.post<Report>('/reports', data);
    return response.data;
  }

  async downloadReport(reportId: string): Promise<Blob> {
    const response = await this.client.get(`/reports/${reportId}/download`, {
      responseType: 'blob',
    });
    return response.data;
  }

  async deleteReport(reportId: string): Promise<void> {
    await this.client.delete(`/reports/${reportId}`);
  }

  // ============================================================================
  // Dashboard Endpoints
  // ============================================================================

  async getDashboardMetrics(params?: {
    tenantId?: string;
    startDate?: string;
    endDate?: string;
  }): Promise<DashboardMetrics> {
    const response = await this.client.get<DashboardMetrics>('/dashboard/metrics', {
      params,
    });
    return response.data;
  }

  async getSystemAlerts(params?: {
    severity?: string;
    acknowledged?: boolean;
    page?: number;
    perPage?: number;
  }): Promise<PaginatedResponse<SystemAlert>> {
    const response = await this.client.get<PaginatedResponse<SystemAlert>>(
      '/dashboard/alerts',
      { params }
    );
    return response.data;
  }

  async acknowledgeAlert(alertId: string): Promise<SystemAlert> {
    const response = await this.client.post<SystemAlert>(
      `/dashboard/alerts/${alertId}/acknowledge`
    );
    return response.data;
  }
}

// Export singleton instance
export const apiClient = new GreenLangAPIClient();

// Export class for testing
export { GreenLangAPIClient };
