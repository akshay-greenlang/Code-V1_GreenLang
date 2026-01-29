/**
 * API Client for GreenLang Review Console
 *
 * Provides type-safe API methods for interacting with the review queue backend.
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
  QueueItem,
  QueueFilters,
  PaginationParams,
  PaginatedResponse,
  ResolutionSubmission,
  Resolution,
  DashboardStats,
  APIError,
  User,
} from './types';

// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';
const API_TIMEOUT = 30000;

/**
 * Create configured axios instance
 */
function createAPIClient(): AxiosInstance {
  const client = axios.create({
    baseURL: API_BASE_URL,
    timeout: API_TIMEOUT,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Request interceptor for auth token
  client.interceptors.request.use((config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  });

  // Response interceptor for error handling
  client.interceptors.response.use(
    (response) => response,
    (error: AxiosError<APIError>) => {
      if (error.response?.status === 401) {
        // Handle unauthorized - redirect to login
        window.location.href = '/login';
      }
      return Promise.reject(error);
    }
  );

  return client;
}

const apiClient = createAPIClient();

/**
 * Review Queue API
 */
export const queueAPI = {
  /**
   * Get paginated queue items with optional filters
   */
  async getQueue(
    filters: QueueFilters = {},
    pagination: PaginationParams = { page: 1, perPage: 25 }
  ): Promise<PaginatedResponse<QueueItem>> {
    const params = {
      ...filters,
      ...pagination,
    };
    const response = await apiClient.get<PaginatedResponse<QueueItem>>('/review/queue', { params });
    return response.data;
  },

  /**
   * Get a single queue item by ID
   */
  async getQueueItem(id: string): Promise<QueueItem> {
    const response = await apiClient.get<QueueItem>(`/review/queue/${id}`);
    return response.data;
  },

  /**
   * Claim a queue item for review
   */
  async claimItem(id: string): Promise<QueueItem> {
    const response = await apiClient.post<QueueItem>(`/review/queue/${id}/claim`);
    return response.data;
  },

  /**
   * Release a claimed queue item
   */
  async releaseItem(id: string): Promise<QueueItem> {
    const response = await apiClient.post<QueueItem>(`/review/queue/${id}/release`);
    return response.data;
  },

  /**
   * Submit resolution for a queue item
   */
  async submitResolution(
    id: string,
    resolution: ResolutionSubmission
  ): Promise<Resolution> {
    const response = await apiClient.post<Resolution>(
      `/review/queue/${id}/resolve`,
      resolution
    );
    return response.data;
  },

  /**
   * Get next item from queue (auto-claim)
   */
  async getNextItem(filters?: QueueFilters): Promise<QueueItem | null> {
    try {
      const response = await apiClient.post<QueueItem>('/review/queue/next', filters);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.status === 404) {
        return null;
      }
      throw error;
    }
  },

  /**
   * Skip current item (defer for later)
   */
  async skipItem(id: string, reason?: string): Promise<QueueItem> {
    const response = await apiClient.post<QueueItem>(`/review/queue/${id}/skip`, { reason });
    return response.data;
  },

  /**
   * Escalate item to senior reviewer
   */
  async escalateItem(id: string, reason: string): Promise<QueueItem> {
    const response = await apiClient.post<QueueItem>(`/review/queue/${id}/escalate`, { reason });
    return response.data;
  },
};

/**
 * Dashboard API
 */
export const dashboardAPI = {
  /**
   * Get dashboard statistics
   */
  async getStats(): Promise<DashboardStats> {
    const response = await apiClient.get<DashboardStats>('/review/dashboard/stats');
    return response.data;
  },

  /**
   * Get reviewer performance metrics
   */
  async getPerformance(reviewerId?: string): Promise<{
    resolvedToday: number;
    resolvedThisWeek: number;
    averageTime: number;
    accuracy: number;
  }> {
    const params = reviewerId ? { reviewerId } : {};
    const response = await apiClient.get('/review/dashboard/performance', { params });
    return response.data;
  },
};

/**
 * User API
 */
export const userAPI = {
  /**
   * Get current user info
   */
  async getCurrentUser(): Promise<User> {
    const response = await apiClient.get<User>('/auth/me');
    return response.data;
  },

  /**
   * Login
   */
  async login(email: string, password: string): Promise<{ token: string; user: User }> {
    const response = await apiClient.post('/auth/login', { email, password });
    localStorage.setItem('auth_token', response.data.token);
    return response.data;
  },

  /**
   * Logout
   */
  async logout(): Promise<void> {
    await apiClient.post('/auth/logout');
    localStorage.removeItem('auth_token');
  },
};

/**
 * Export all APIs
 */
export const api = {
  queue: queueAPI,
  dashboard: dashboardAPI,
  user: userAPI,
};

export default api;
