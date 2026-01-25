import axios, { AxiosInstance, AxiosError, AxiosRequestConfig, AxiosResponse } from 'axios';
import type {
  ApiResponse,
  ApiError,
  PaginatedResponse,
  Transaction,
  Supplier,
  DashboardMetrics,
  Report,
  ReportRequest,
  UploadResponse,
  HotspotAnalysis,
  EmissionResult,
} from '../types';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';
const API_TIMEOUT = 30000; // 30 seconds

// Create Axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor - Add auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor - Handle errors globally
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: AxiosError<ApiError>) => {
    if (error.response?.status === 401) {
      // Unauthorized - clear token and redirect to login
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }

    // Format error response
    const apiError: ApiError = {
      error: error.response?.data?.error || 'API Error',
      message: error.response?.data?.message || error.message || 'An unexpected error occurred',
      statusCode: error.response?.status || 500,
      requestId: error.response?.headers?.['x-request-id'],
    };

    return Promise.reject(apiError);
  }
);

// ==============================================================================
// API Service Functions
// ==============================================================================

export const api = {
  // ============================================================================
  // Dashboard & Analytics
  // ============================================================================
  getDashboardMetrics: async (startDate?: string, endDate?: string): Promise<DashboardMetrics> => {
    const params: any = {};
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;

    const response = await apiClient.get<ApiResponse<DashboardMetrics>>('/dashboard/metrics', { params });
    return response.data.data;
  },

  getHotspotAnalysis: async (params?: { minEmissions?: number; topN?: number }): Promise<HotspotAnalysis> => {
    const response = await apiClient.get<ApiResponse<HotspotAnalysis>>('/hotspot/analysis', { params });
    return response.data.data;
  },

  // ============================================================================
  // Transactions / Data Upload
  // ============================================================================
  getTransactions: async (params?: {
    page?: number;
    pageSize?: number;
    search?: string;
    startDate?: string;
    endDate?: string;
    status?: string[];
    categories?: number[];
  }): Promise<PaginatedResponse<Transaction>> => {
    const response = await apiClient.get<PaginatedResponse<Transaction>>('/intake/transactions', { params });
    return response.data;
  },

  getTransaction: async (id: string): Promise<Transaction> => {
    const response = await apiClient.get<ApiResponse<Transaction>>(`/intake/transactions/${id}`);
    return response.data.data;
  },

  uploadTransactions: async (file: File, format: string): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('format', format);

    const response = await apiClient.post<ApiResponse<UploadResponse>>(
      '/intake/upload',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // 60 seconds for file upload
      }
    );
    return response.data.data;
  },

  getUploadStatus: async (jobId: string): Promise<UploadResponse> => {
    const response = await apiClient.get<ApiResponse<UploadResponse>>(`/intake/upload/${jobId}/status`);
    return response.data.data;
  },

  deleteTransaction: async (id: string): Promise<void> => {
    await apiClient.delete(`/intake/transactions/${id}`);
  },

  // ============================================================================
  // Suppliers
  // ============================================================================
  getSuppliers: async (params?: {
    page?: number;
    pageSize?: number;
    search?: string;
    status?: string[];
  }): Promise<PaginatedResponse<Supplier>> => {
    const response = await apiClient.get<PaginatedResponse<Supplier>>('/engagement/suppliers', { params });
    return response.data;
  },

  getSupplier: async (id: string): Promise<Supplier> => {
    const response = await apiClient.get<ApiResponse<Supplier>>(`/engagement/suppliers/${id}`);
    return response.data.data;
  },

  createEngagementCampaign: async (supplierIds: string[], message?: string): Promise<{ campaignId: string }> => {
    const response = await apiClient.post<ApiResponse<{ campaignId: string }>>(
      '/engagement/campaigns',
      { supplier_ids: supplierIds, message }
    );
    return response.data.data;
  },

  getSupplierEngagements: async (supplierId: string): Promise<any[]> => {
    const response = await apiClient.get<ApiResponse<any[]>>(`/engagement/suppliers/${supplierId}/engagements`);
    return response.data.data;
  },

  // ============================================================================
  // Calculations
  // ============================================================================
  calculateEmissions: async (transactionIds: string[], options?: {
    category?: number;
    methodology?: string;
    uncertainty?: boolean;
  }): Promise<EmissionResult[]> => {
    const response = await apiClient.post<ApiResponse<EmissionResult[]>>(
      '/calculator/calculate',
      { transaction_ids: transactionIds, options }
    );
    return response.data.data;
  },

  getEmissionResults: async (transactionId: string): Promise<EmissionResult> => {
    const response = await apiClient.get<ApiResponse<EmissionResult>>(`/calculator/results/${transactionId}`);
    return response.data.data;
  },

  // ============================================================================
  // Reports
  // ============================================================================
  getReports: async (params?: {
    page?: number;
    pageSize?: number;
    type?: string;
    status?: string[];
  }): Promise<PaginatedResponse<Report>> => {
    const response = await apiClient.get<PaginatedResponse<Report>>('/reporting/reports', { params });
    return response.data;
  },

  getReport: async (id: string): Promise<Report> => {
    const response = await apiClient.get<ApiResponse<Report>>(`/reporting/reports/${id}`);
    return response.data.data;
  },

  generateReport: async (request: ReportRequest): Promise<Report> => {
    const response = await apiClient.post<ApiResponse<Report>>('/reporting/generate', {
      report_type: request.type,
      start_date: request.startDate,
      end_date: request.endDate,
      format: request.format,
      options: request.options,
    });
    return response.data.data;
  },

  downloadReport: async (reportId: string): Promise<Blob> => {
    const response = await apiClient.get(`/reporting/reports/${reportId}/download`, {
      responseType: 'blob',
    });
    return response.data;
  },

  deleteReport: async (id: string): Promise<void> => {
    await apiClient.delete(`/reporting/reports/${id}`);
  },

  // ============================================================================
  // Health & Status
  // ============================================================================
  checkHealth: async (): Promise<{ status: string; version: string }> => {
    const response = await apiClient.get('/health/ready');
    return response.data;
  },
};

export default api;
