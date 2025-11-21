/**
 * GL-007 Furnace Performance Monitor - API Client
 *
 * Type-safe API client for GreenLang Furnace Monitoring System
 */

import axios, { AxiosInstance, AxiosError, AxiosRequestConfig } from 'axios';
import type {
  ApiResponse,
  PaginatedResponse,
  FurnaceConfig,
  FurnacePerformance,
  Alert,
  AlertConfiguration,
  MaintenanceSchedule,
  MaintenanceTask,
  RefractoryCondition,
  AnalyticsData,
  Report,
  TimePeriod,
  DashboardConfig,
  OptimizationOpportunity,
  RootCauseAnalysis,
  WhatIfScenario,
} from '../types';

// ============================================================================
// API CLIENT CONFIGURATION
// ============================================================================

interface ApiClientConfig {
  baseURL: string;
  timeout?: number;
  apiKey?: string;
  refreshTokenUrl?: string;
}

interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
}

interface QueryParams {
  [key: string]: string | number | boolean | undefined;
}

// ============================================================================
// API CLIENT CLASS
// ============================================================================

export class FurnaceMonitorAPIClient {
  private client: AxiosInstance;
  private tokens: AuthTokens | null = null;
  private refreshPromise: Promise<void> | null = null;

  constructor(config: ApiClientConfig) {
    const { baseURL, timeout = 30000, apiKey } = config;

    this.client = axios.create({
      baseURL,
      timeout,
      headers: {
        'Content-Type': 'application/json',
        ...(apiKey && { 'X-API-Key': apiKey }),
      },
    });

    this.setupInterceptors();
  }

  // ==========================================================================
  // AUTHENTICATION
  // ==========================================================================

  /**
   * Authenticate with client credentials
   */
  async authenticate(clientId: string, clientSecret: string): Promise<void> {
    try {
      const response = await this.client.post<ApiResponse<AuthTokens>>('/auth/token', {
        grant_type: 'client_credentials',
        client_id: clientId,
        client_secret: clientSecret,
      });

      if (response.data.success && response.data.data) {
        this.tokens = response.data.data;
        this.scheduleTokenRefresh();
      } else {
        throw new Error(response.data.error?.message || 'Authentication failed');
      }
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Refresh access token
   */
  private async refreshAccessToken(): Promise<void> {
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    this.refreshPromise = (async () => {
      try {
        if (!this.tokens?.refreshToken) {
          throw new Error('No refresh token available');
        }

        const response = await this.client.post<ApiResponse<AuthTokens>>('/auth/token', {
          grant_type: 'refresh_token',
          refresh_token: this.tokens.refreshToken,
        });

        if (response.data.success && response.data.data) {
          this.tokens = response.data.data;
          this.scheduleTokenRefresh();
        }
      } finally {
        this.refreshPromise = null;
      }
    })();

    return this.refreshPromise;
  }

  /**
   * Schedule automatic token refresh
   */
  private scheduleTokenRefresh(): void {
    if (!this.tokens) return;

    const expiresIn = this.tokens.expiresAt - Date.now();
    const refreshTime = expiresIn - 60000; // Refresh 1 minute before expiry

    if (refreshTime > 0) {
      setTimeout(() => this.refreshAccessToken(), refreshTime);
    }
  }

  /**
   * Setup request/response interceptors
   */
  private setupInterceptors(): void {
    // Request interceptor: Add auth token
    this.client.interceptors.request.use(
      (config) => {
        if (this.tokens?.accessToken) {
          config.headers.Authorization = `Bearer ${this.tokens.accessToken}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor: Handle token refresh
    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean };

        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          try {
            await this.refreshAccessToken();
            return this.client.request(originalRequest);
          } catch (refreshError) {
            // Token refresh failed, clear tokens
            this.tokens = null;
            throw refreshError;
          }
        }

        return Promise.reject(error);
      }
    );
  }

  // ==========================================================================
  // FURNACE MANAGEMENT
  // ==========================================================================

  /**
   * Get all furnaces
   */
  async getFurnaces(): Promise<FurnaceConfig[]> {
    const response = await this.client.get<ApiResponse<FurnaceConfig[]>>('/furnaces');
    return this.unwrapResponse(response.data);
  }

  /**
   * Get furnace by ID
   */
  async getFurnace(furnaceId: string): Promise<FurnaceConfig> {
    const response = await this.client.get<ApiResponse<FurnaceConfig>>(
      `/furnaces/${furnaceId}`
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Update furnace configuration
   */
  async updateFurnace(
    furnaceId: string,
    config: Partial<FurnaceConfig>
  ): Promise<FurnaceConfig> {
    const response = await this.client.put<ApiResponse<FurnaceConfig>>(
      `/furnaces/${furnaceId}`,
      config
    );
    return this.unwrapResponse(response.data);
  }

  // ==========================================================================
  // REAL-TIME PERFORMANCE
  // ==========================================================================

  /**
   * Get current furnace performance
   */
  async getPerformance(furnaceId: string): Promise<FurnacePerformance> {
    const response = await this.client.get<ApiResponse<FurnacePerformance>>(
      `/furnaces/${furnaceId}/performance`
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Get performance history
   */
  async getPerformanceHistory(
    furnaceId: string,
    period: TimePeriod
  ): Promise<FurnacePerformance[]> {
    const response = await this.client.get<ApiResponse<FurnacePerformance[]>>(
      `/furnaces/${furnaceId}/performance/history`,
      {
        params: {
          start: period.start,
          end: period.end,
          granularity: period.granularity,
        },
      }
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Get specific KPIs
   */
  async getKPIs(furnaceId: string, kpis: string[]): Promise<Record<string, number>> {
    const response = await this.client.get<ApiResponse<Record<string, number>>>(
      `/furnaces/${furnaceId}/kpis`,
      {
        params: { kpis: kpis.join(',') },
      }
    );
    return this.unwrapResponse(response.data);
  }

  // ==========================================================================
  // ALERTS & ALARMS
  // ==========================================================================

  /**
   * Get alerts
   */
  async getAlerts(
    furnaceId: string,
    params?: {
      status?: string;
      severity?: string;
      category?: string;
      page?: number;
      perPage?: number;
    }
  ): Promise<PaginatedResponse<Alert>> {
    const response = await this.client.get<ApiResponse<PaginatedResponse<Alert>>>(
      `/furnaces/${furnaceId}/alerts`,
      { params }
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Get alert by ID
   */
  async getAlert(furnaceId: string, alertId: string): Promise<Alert> {
    const response = await this.client.get<ApiResponse<Alert>>(
      `/furnaces/${furnaceId}/alerts/${alertId}`
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Acknowledge alert
   */
  async acknowledgeAlert(
    furnaceId: string,
    alertId: string,
    userId: string,
    notes?: string
  ): Promise<Alert> {
    const response = await this.client.post<ApiResponse<Alert>>(
      `/furnaces/${furnaceId}/alerts/${alertId}/acknowledge`,
      { userId, notes }
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Resolve alert
   */
  async resolveAlert(
    furnaceId: string,
    alertId: string,
    resolution: string
  ): Promise<Alert> {
    const response = await this.client.post<ApiResponse<Alert>>(
      `/furnaces/${furnaceId}/alerts/${alertId}/resolve`,
      { resolution }
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Get alert configuration
   */
  async getAlertConfig(furnaceId: string): Promise<AlertConfiguration> {
    const response = await this.client.get<ApiResponse<AlertConfiguration>>(
      `/furnaces/${furnaceId}/alerts/config`
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Update alert configuration
   */
  async updateAlertConfig(
    furnaceId: string,
    config: Partial<AlertConfiguration>
  ): Promise<AlertConfiguration> {
    const response = await this.client.put<ApiResponse<AlertConfiguration>>(
      `/furnaces/${furnaceId}/alerts/config`,
      config
    );
    return this.unwrapResponse(response.data);
  }

  // ==========================================================================
  // MAINTENANCE
  // ==========================================================================

  /**
   * Get maintenance schedule
   */
  async getMaintenanceSchedule(furnaceId: string): Promise<MaintenanceSchedule> {
    const response = await this.client.get<ApiResponse<MaintenanceSchedule>>(
      `/furnaces/${furnaceId}/maintenance/schedule`
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Get maintenance task
   */
  async getMaintenanceTask(furnaceId: string, taskId: string): Promise<MaintenanceTask> {
    const response = await this.client.get<ApiResponse<MaintenanceTask>>(
      `/furnaces/${furnaceId}/maintenance/tasks/${taskId}`
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Update maintenance task
   */
  async updateMaintenanceTask(
    furnaceId: string,
    taskId: string,
    updates: Partial<MaintenanceTask>
  ): Promise<MaintenanceTask> {
    const response = await this.client.put<ApiResponse<MaintenanceTask>>(
      `/furnaces/${furnaceId}/maintenance/tasks/${taskId}`,
      updates
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Complete maintenance task
   */
  async completeMaintenanceTask(
    furnaceId: string,
    taskId: string,
    completionData: {
      completedBy: string;
      notes: string;
      actualDuration: number;
      actualCost: number;
    }
  ): Promise<MaintenanceTask> {
    const response = await this.client.post<ApiResponse<MaintenanceTask>>(
      `/furnaces/${furnaceId}/maintenance/tasks/${taskId}/complete`,
      completionData
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Get refractory condition
   */
  async getRefractoryCondition(furnaceId: string): Promise<RefractoryCondition> {
    const response = await this.client.get<ApiResponse<RefractoryCondition>>(
      `/furnaces/${furnaceId}/refractory`
    );
    return this.unwrapResponse(response.data);
  }

  // ==========================================================================
  // ANALYTICS
  // ==========================================================================

  /**
   * Get analytics data
   */
  async getAnalytics(furnaceId: string, period: TimePeriod): Promise<AnalyticsData> {
    const response = await this.client.get<ApiResponse<AnalyticsData>>(
      `/furnaces/${furnaceId}/analytics`,
      {
        params: {
          start: period.start,
          end: period.end,
          granularity: period.granularity,
        },
      }
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Get optimization opportunities
   */
  async getOptimizationOpportunities(
    furnaceId: string
  ): Promise<OptimizationOpportunity[]> {
    const response = await this.client.get<ApiResponse<OptimizationOpportunity[]>>(
      `/furnaces/${furnaceId}/optimization`
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Get root cause analysis
   */
  async getRootCauseAnalysis(
    furnaceId: string,
    params?: { start?: string; end?: string }
  ): Promise<RootCauseAnalysis[]> {
    const response = await this.client.get<ApiResponse<RootCauseAnalysis[]>>(
      `/furnaces/${furnaceId}/analytics/root-cause`,
      { params }
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Run what-if scenario
   */
  async runWhatIfScenario(
    furnaceId: string,
    scenario: Omit<WhatIfScenario, 'id' | 'results'>
  ): Promise<WhatIfScenario> {
    const response = await this.client.post<ApiResponse<WhatIfScenario>>(
      `/furnaces/${furnaceId}/analytics/what-if`,
      scenario
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Get benchmarking data
   */
  async getBenchmarking(furnaceId: string): Promise<any> {
    const response = await this.client.get<ApiResponse<any>>(
      `/furnaces/${furnaceId}/analytics/benchmark`
    );
    return this.unwrapResponse(response.data);
  }

  // ==========================================================================
  // REPORTS
  // ==========================================================================

  /**
   * Get reports list
   */
  async getReports(
    furnaceId: string,
    params?: { type?: string; page?: number; perPage?: number }
  ): Promise<PaginatedResponse<Report>> {
    const response = await this.client.get<ApiResponse<PaginatedResponse<Report>>>(
      `/furnaces/${furnaceId}/reports`,
      { params }
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Get report by ID
   */
  async getReport(furnaceId: string, reportId: string): Promise<Report> {
    const response = await this.client.get<ApiResponse<Report>>(
      `/furnaces/${furnaceId}/reports/${reportId}`
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Generate report
   */
  async generateReport(
    furnaceId: string,
    config: {
      type: string;
      period: TimePeriod;
      sections: string[];
      format: string;
    }
  ): Promise<Report> {
    const response = await this.client.post<ApiResponse<Report>>(
      `/furnaces/${furnaceId}/reports/generate`,
      config
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Export report
   */
  async exportReport(
    furnaceId: string,
    reportId: string,
    format: 'pdf' | 'excel' | 'csv' | 'json'
  ): Promise<Blob> {
    const response = await this.client.get(`/furnaces/${furnaceId}/reports/${reportId}/export`, {
      params: { format },
      responseType: 'blob',
    });
    return response.data;
  }

  /**
   * Schedule report
   */
  async scheduleReport(
    furnaceId: string,
    config: {
      type: string;
      frequency: string;
      recipients: string[];
      format: string;
    }
  ): Promise<any> {
    const response = await this.client.post<ApiResponse<any>>(
      `/furnaces/${furnaceId}/reports/schedule`,
      config
    );
    return this.unwrapResponse(response.data);
  }

  // ==========================================================================
  // DASHBOARD CONFIGURATION
  // ==========================================================================

  /**
   * Get dashboard configuration
   */
  async getDashboardConfig(userId: string): Promise<DashboardConfig> {
    const response = await this.client.get<ApiResponse<DashboardConfig>>(
      `/users/${userId}/dashboard`
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Save dashboard configuration
   */
  async saveDashboardConfig(
    userId: string,
    config: DashboardConfig
  ): Promise<DashboardConfig> {
    const response = await this.client.put<ApiResponse<DashboardConfig>>(
      `/users/${userId}/dashboard`,
      config
    );
    return this.unwrapResponse(response.data);
  }

  // ==========================================================================
  // THERMAL IMAGING
  // ==========================================================================

  /**
   * Get thermal profile
   */
  async getThermalProfile(furnaceId: string): Promise<any> {
    const response = await this.client.get<ApiResponse<any>>(
      `/furnaces/${furnaceId}/thermal/profile`
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Get thermal imaging data
   */
  async getThermalImaging(
    furnaceId: string,
    zoneId?: string
  ): Promise<{ imageUrl: string; data: any }> {
    const response = await this.client.get<ApiResponse<{ imageUrl: string; data: any }>>(
      `/furnaces/${furnaceId}/thermal/imaging`,
      { params: { zoneId } }
    );
    return this.unwrapResponse(response.data);
  }

  /**
   * Get hot spots
   */
  async getHotSpots(furnaceId: string): Promise<any[]> {
    const response = await this.client.get<ApiResponse<any[]>>(
      `/furnaces/${furnaceId}/thermal/hotspots`
    );
    return this.unwrapResponse(response.data);
  }

  // ==========================================================================
  // HELPER METHODS
  // ==========================================================================

  /**
   * Unwrap API response
   */
  private unwrapResponse<T>(response: ApiResponse<T>): T {
    if (response.success && response.data !== undefined) {
      return response.data;
    }
    throw new Error(response.error?.message || 'API request failed');
  }

  /**
   * Handle errors
   */
  private handleError(error: unknown): Error {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<ApiResponse<any>>;
      if (axiosError.response?.data?.error) {
        return new Error(axiosError.response.data.error.message);
      }
      return new Error(axiosError.message);
    }
    return error instanceof Error ? error : new Error('Unknown error');
  }

  /**
   * Build query string
   */
  private buildQueryString(params: QueryParams): string {
    const filtered = Object.entries(params)
      .filter(([_, value]) => value !== undefined)
      .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`)
      .join('&');
    return filtered ? `?${filtered}` : '';
  }
}

// ============================================================================
// SINGLETON INSTANCE
// ============================================================================

const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://api.greenlang.io/v1';
const API_KEY = import.meta.env.VITE_API_KEY;

export const apiClient = new FurnaceMonitorAPIClient({
  baseURL: API_BASE_URL,
  apiKey: API_KEY,
});

export default apiClient;
