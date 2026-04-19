/**
 * GreenLang Agent Factory - Marketplace API Client
 *
 * API methods for the Agent Marketplace and Discovery system.
 */

import axios, { AxiosInstance } from 'axios';
import type {
  MarketplaceAgent,
  AgentVersion,
  AgentReview,
  ReviewSummary,
  AgentDeployment,
  Workflow,
  WorkflowExecution,
  AgentListParams,
  AgentSearchResult,
  DeployAgentRequest,
  DeployAgentResponse,
  CreateReviewRequest,
  WorkflowCreateRequest,
  WorkflowTestRequest,
  WorkflowTestResult,
  AgentComparison,
} from './types/marketplace';
import type { PaginatedResponse } from './types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

// Token storage key
const ACCESS_TOKEN_KEY = 'gl_access_token';

/**
 * Marketplace API Client
 *
 * Handles all marketplace-related API calls including agent discovery,
 * deployments, reviews, and workflow management.
 */
class MarketplaceAPIClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: `${API_BASE_URL}/marketplace`,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem(ACCESS_TOKEN_KEY);
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );
  }

  // ============================================================================
  // Agent Discovery Endpoints
  // ============================================================================

  /**
   * Get a paginated list of marketplace agents with filtering and sorting
   */
  async getAgents(params?: AgentListParams): Promise<AgentSearchResult> {
    const response = await this.client.get<AgentSearchResult>('/agents', { params });
    return response.data;
  }

  /**
   * Get a single agent by ID or slug
   */
  async getAgent(idOrSlug: string): Promise<MarketplaceAgent> {
    const response = await this.client.get<MarketplaceAgent>(`/agents/${idOrSlug}`);
    return response.data;
  }

  /**
   * Get featured agents for the marketplace homepage
   */
  async getFeaturedAgents(): Promise<MarketplaceAgent[]> {
    const response = await this.client.get<MarketplaceAgent[]>('/agents/featured');
    return response.data;
  }

  /**
   * Get trending agents based on recent deployments
   */
  async getTrendingAgents(limit?: number): Promise<MarketplaceAgent[]> {
    const response = await this.client.get<MarketplaceAgent[]>('/agents/trending', {
      params: { limit: limit || 10 },
    });
    return response.data;
  }

  /**
   * Get newly added agents
   */
  async getNewAgents(limit?: number): Promise<MarketplaceAgent[]> {
    const response = await this.client.get<MarketplaceAgent[]>('/agents/new', {
      params: { limit: limit || 10 },
    });
    return response.data;
  }

  /**
   * Get agents by regulatory framework
   */
  async getAgentsByRegulation(
    regulation: string,
    params?: Omit<AgentListParams, 'regulatoryFramework'>
  ): Promise<AgentSearchResult> {
    const response = await this.client.get<AgentSearchResult>('/agents', {
      params: { ...params, regulatoryFramework: regulation },
    });
    return response.data;
  }

  /**
   * Get related/similar agents
   */
  async getRelatedAgents(agentId: string, limit?: number): Promise<MarketplaceAgent[]> {
    const response = await this.client.get<MarketplaceAgent[]>(
      `/agents/${agentId}/related`,
      { params: { limit: limit || 5 } }
    );
    return response.data;
  }

  /**
   * Search agents with autocomplete
   */
  async searchAgents(query: string, limit?: number): Promise<{
    agents: MarketplaceAgent[];
    suggestions: string[];
  }> {
    const response = await this.client.get('/agents/search', {
      params: { q: query, limit: limit || 10 },
    });
    return response.data;
  }

  // ============================================================================
  // Agent Version Endpoints
  // ============================================================================

  /**
   * Get version history for an agent
   */
  async getAgentVersions(
    agentId: string,
    params?: { page?: number; perPage?: number }
  ): Promise<PaginatedResponse<AgentVersion>> {
    const response = await this.client.get<PaginatedResponse<AgentVersion>>(
      `/agents/${agentId}/versions`,
      { params }
    );
    return response.data;
  }

  /**
   * Get a specific version of an agent
   */
  async getAgentVersion(agentId: string, version: string): Promise<AgentVersion> {
    const response = await this.client.get<AgentVersion>(
      `/agents/${agentId}/versions/${version}`
    );
    return response.data;
  }

  // ============================================================================
  // Review Endpoints
  // ============================================================================

  /**
   * Get reviews for an agent
   */
  async getAgentReviews(
    agentId: string,
    params?: {
      page?: number;
      perPage?: number;
      sortBy?: 'newest' | 'oldest' | 'highest' | 'lowest' | 'helpful';
    }
  ): Promise<PaginatedResponse<AgentReview>> {
    const response = await this.client.get<PaginatedResponse<AgentReview>>(
      `/agents/${agentId}/reviews`,
      { params }
    );
    return response.data;
  }

  /**
   * Get review summary for an agent
   */
  async getReviewSummary(agentId: string): Promise<ReviewSummary> {
    const response = await this.client.get<ReviewSummary>(
      `/agents/${agentId}/reviews/summary`
    );
    return response.data;
  }

  /**
   * Create a new review
   */
  async createReview(data: CreateReviewRequest): Promise<AgentReview> {
    const response = await this.client.post<AgentReview>(
      `/agents/${data.agentId}/reviews`,
      data
    );
    return response.data;
  }

  /**
   * Update an existing review
   */
  async updateReview(
    agentId: string,
    reviewId: string,
    data: Partial<CreateReviewRequest>
  ): Promise<AgentReview> {
    const response = await this.client.patch<AgentReview>(
      `/agents/${agentId}/reviews/${reviewId}`,
      data
    );
    return response.data;
  }

  /**
   * Delete a review
   */
  async deleteReview(agentId: string, reviewId: string): Promise<void> {
    await this.client.delete(`/agents/${agentId}/reviews/${reviewId}`);
  }

  /**
   * Mark a review as helpful
   */
  async markReviewHelpful(agentId: string, reviewId: string): Promise<AgentReview> {
    const response = await this.client.post<AgentReview>(
      `/agents/${agentId}/reviews/${reviewId}/helpful`
    );
    return response.data;
  }

  // ============================================================================
  // Deployment Endpoints
  // ============================================================================

  /**
   * Deploy an agent
   */
  async deployAgent(data: DeployAgentRequest): Promise<DeployAgentResponse> {
    const response = await this.client.post<DeployAgentResponse>(
      `/agents/${data.agentId}/deploy`,
      data
    );
    return response.data;
  }

  /**
   * Get user's deployments
   */
  async getMyDeployments(params?: {
    page?: number;
    perPage?: number;
    status?: string;
  }): Promise<PaginatedResponse<AgentDeployment>> {
    const response = await this.client.get<PaginatedResponse<AgentDeployment>>(
      '/deployments',
      { params }
    );
    return response.data;
  }

  /**
   * Get a specific deployment
   */
  async getDeployment(deploymentId: string): Promise<AgentDeployment> {
    const response = await this.client.get<AgentDeployment>(
      `/deployments/${deploymentId}`
    );
    return response.data;
  }

  /**
   * Update deployment configuration
   */
  async updateDeployment(
    deploymentId: string,
    config: Partial<AgentDeployment['configuration']>
  ): Promise<AgentDeployment> {
    const response = await this.client.patch<AgentDeployment>(
      `/deployments/${deploymentId}`,
      { configuration: config }
    );
    return response.data;
  }

  /**
   * Undeploy/remove an agent deployment
   */
  async undeployAgent(deploymentId: string): Promise<void> {
    await this.client.delete(`/deployments/${deploymentId}`);
  }

  /**
   * Get deployment metrics
   */
  async getDeploymentMetrics(
    deploymentId: string,
    params?: { startDate?: string; endDate?: string }
  ): Promise<{
    requests: { date: string; count: number }[];
    responseTime: { date: string; avg: number }[];
    errors: { date: string; count: number }[];
  }> {
    const response = await this.client.get(`/deployments/${deploymentId}/metrics`, {
      params,
    });
    return response.data;
  }

  // ============================================================================
  // Comparison Endpoints
  // ============================================================================

  /**
   * Compare multiple agents
   */
  async compareAgents(agentIds: string[]): Promise<AgentComparison> {
    const response = await this.client.post<AgentComparison>('/agents/compare', {
      agentIds,
    });
    return response.data;
  }

  // ============================================================================
  // Workflow Endpoints
  // ============================================================================

  /**
   * Get user's workflows
   */
  async getWorkflows(params?: {
    page?: number;
    perPage?: number;
    status?: string;
  }): Promise<PaginatedResponse<Workflow>> {
    const response = await this.client.get<PaginatedResponse<Workflow>>('/workflows', {
      params,
    });
    return response.data;
  }

  /**
   * Get a specific workflow
   */
  async getWorkflow(workflowId: string): Promise<Workflow> {
    const response = await this.client.get<Workflow>(`/workflows/${workflowId}`);
    return response.data;
  }

  /**
   * Create a new workflow
   */
  async createWorkflow(data: WorkflowCreateRequest): Promise<Workflow> {
    const response = await this.client.post<Workflow>('/workflows', data);
    return response.data;
  }

  /**
   * Update a workflow
   */
  async updateWorkflow(
    workflowId: string,
    data: Partial<WorkflowCreateRequest>
  ): Promise<Workflow> {
    const response = await this.client.patch<Workflow>(
      `/workflows/${workflowId}`,
      data
    );
    return response.data;
  }

  /**
   * Delete a workflow
   */
  async deleteWorkflow(workflowId: string): Promise<void> {
    await this.client.delete(`/workflows/${workflowId}`);
  }

  /**
   * Test a workflow with sample data
   */
  async testWorkflow(data: WorkflowTestRequest): Promise<WorkflowTestResult> {
    const response = await this.client.post<WorkflowTestResult>(
      `/workflows/${data.workflowId}/test`,
      data
    );
    return response.data;
  }

  /**
   * Execute a workflow
   */
  async executeWorkflow(
    workflowId: string,
    input: Record<string, unknown>
  ): Promise<WorkflowExecution> {
    const response = await this.client.post<WorkflowExecution>(
      `/workflows/${workflowId}/execute`,
      { input }
    );
    return response.data;
  }

  /**
   * Get workflow execution history
   */
  async getWorkflowExecutions(
    workflowId: string,
    params?: { page?: number; perPage?: number }
  ): Promise<PaginatedResponse<WorkflowExecution>> {
    const response = await this.client.get<PaginatedResponse<WorkflowExecution>>(
      `/workflows/${workflowId}/executions`,
      { params }
    );
    return response.data;
  }

  /**
   * Get a specific workflow execution
   */
  async getWorkflowExecution(
    workflowId: string,
    executionId: string
  ): Promise<WorkflowExecution> {
    const response = await this.client.get<WorkflowExecution>(
      `/workflows/${workflowId}/executions/${executionId}`
    );
    return response.data;
  }

  // ============================================================================
  // Favorites Endpoints
  // ============================================================================

  /**
   * Get user's favorite agents
   */
  async getFavorites(): Promise<MarketplaceAgent[]> {
    const response = await this.client.get<MarketplaceAgent[]>('/favorites');
    return response.data;
  }

  /**
   * Add agent to favorites
   */
  async addToFavorites(agentId: string): Promise<void> {
    await this.client.post(`/favorites/${agentId}`);
  }

  /**
   * Remove agent from favorites
   */
  async removeFromFavorites(agentId: string): Promise<void> {
    await this.client.delete(`/favorites/${agentId}`);
  }

  // ============================================================================
  // Category Endpoints
  // ============================================================================

  /**
   * Get all categories with agent counts
   */
  async getCategories(): Promise<{
    category: string;
    displayName: string;
    description: string;
    icon: string;
    agentCount: number;
  }[]> {
    const response = await this.client.get('/categories');
    return response.data;
  }

  /**
   * Get all available tags
   */
  async getTags(): Promise<{ tag: string; count: number }[]> {
    const response = await this.client.get('/tags');
    return response.data;
  }

  /**
   * Get regulatory frameworks
   */
  async getRegulatoryFrameworks(): Promise<{
    framework: string;
    displayName: string;
    description: string;
    deadline?: string;
    agentCount: number;
  }[]> {
    const response = await this.client.get('/regulatory-frameworks');
    return response.data;
  }
}

// Export singleton instance
export const marketplaceAPI = new MarketplaceAPIClient();

// Export class for testing
export { MarketplaceAPIClient };
