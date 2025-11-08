/**
 * GreenLang SDK Client
 *
 * Main client class for interacting with the GreenLang API.
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosError } from 'axios';
import {
  Workflow,
  WorkflowDefinition,
  Agent,
  ExecutionResult,
  Citation,
  PaginatedResponse,
  ListWorkflowsOptions,
  ListAgentsOptions,
  ListExecutionsOptions,
  StreamChunk,
} from './types';
import {
  GreenLangError,
  AuthenticationError,
  RateLimitError,
  NotFoundError,
  ValidationError,
  APIError,
} from './errors';

export interface ClientOptions {
  /** Your GreenLang API key */
  apiKey: string;
  /** Base URL for the API (default: https://api.greenlang.com) */
  baseURL?: string;
  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;
  /** Maximum number of retries for failed requests (default: 3) */
  maxRetries?: number;
}

/**
 * GreenLang API Client
 *
 * The main entry point for interacting with the GreenLang API.
 *
 * @example
 * ```typescript
 * const client = new GreenLangClient({ apiKey: 'gl_your_api_key' });
 *
 * // List workflows
 * const workflows = await client.listWorkflows();
 *
 * // Execute workflow
 * const result = await client.executeWorkflow('wf_123', { query: 'test' });
 * ```
 */
export class GreenLangClient {
  private client: AxiosInstance;
  private maxRetries: number;

  constructor(options: ClientOptions) {
    const {
      apiKey,
      baseURL = 'https://api.greenlang.com',
      timeout = 30000,
      maxRetries = 3,
    } = options;

    this.maxRetries = maxRetries;

    // Create axios instance with default config
    this.client = axios.create({
      baseURL: baseURL.replace(/\/$/, ''),
      timeout,
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'User-Agent': 'greenlang-js-sdk/1.0.0',
      },
    });

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        throw this.handleError(error);
      }
    );
  }

  /**
   * Handle API errors
   */
  private handleError(error: AxiosError): GreenLangError {
    if (!error.response) {
      return new APIError(error.message || 'Network error');
    }

    const { status, data } = error.response;
    const message = (data as any)?.detail || error.message;

    switch (status) {
      case 401:
        return new AuthenticationError(message);
      case 404:
        return new NotFoundError(message);
      case 422:
        return new ValidationError(message, (data as any)?.errors);
      case 429:
        const retryAfter = error.response.headers['x-ratelimit-reset'];
        return new RateLimitError(message, retryAfter);
      default:
        return new APIError(message, status);
    }
  }

  /**
   * Make HTTP request with retry logic
   */
  private async request<T>(
    config: AxiosRequestConfig,
    retries = 0
  ): Promise<T> {
    try {
      const response = await this.client.request<T>(config);
      return response.data;
    } catch (error) {
      // Retry on rate limit or server errors
      if (
        retries < this.maxRetries &&
        error instanceof APIError &&
        (error.statusCode === 429 || (error.statusCode && error.statusCode >= 500))
      ) {
        const delay = Math.pow(2, retries) * 1000; // Exponential backoff
        await new Promise((resolve) => setTimeout(resolve, delay));
        return this.request<T>(config, retries + 1);
      }
      throw error;
    }
  }

  // Workflow Methods

  /**
   * Create a new workflow
   *
   * @param workflowDef - Workflow definition
   * @returns Created workflow
   *
   * @example
   * ```typescript
   * const workflow = await client.createWorkflow({
   *   name: 'Carbon Analysis',
   *   description: 'Analyze carbon emissions',
   *   agents: [{ agent_id: 'carbon_analyzer', config: {} }],
   * });
   * ```
   */
  async createWorkflow(workflowDef: WorkflowDefinition): Promise<Workflow> {
    return this.request<Workflow>({
      method: 'POST',
      url: '/api/workflows',
      data: workflowDef,
    });
  }

  /**
   * Get workflow by ID
   *
   * @param workflowId - Workflow ID
   * @returns Workflow object
   */
  async getWorkflow(workflowId: string): Promise<Workflow> {
    return this.request<Workflow>({
      method: 'GET',
      url: `/api/workflows/${workflowId}`,
    });
  }

  /**
   * List workflows
   *
   * @param options - List options
   * @returns Array of workflows
   *
   * @example
   * ```typescript
   * const workflows = await client.listWorkflows({
   *   limit: 10,
   *   category: 'carbon',
   * });
   * ```
   */
  async listWorkflows(options: ListWorkflowsOptions = {}): Promise<Workflow[]> {
    const response = await this.request<{ items: Workflow[] }>({
      method: 'GET',
      url: '/api/workflows',
      params: options,
    });
    return response.items;
  }

  /**
   * Update workflow
   *
   * @param workflowId - Workflow ID
   * @param updates - Fields to update
   * @returns Updated workflow
   */
  async updateWorkflow(
    workflowId: string,
    updates: Partial<WorkflowDefinition>
  ): Promise<Workflow> {
    return this.request<Workflow>({
      method: 'PUT',
      url: `/api/workflows/${workflowId}`,
      data: updates,
    });
  }

  /**
   * Delete workflow
   *
   * @param workflowId - Workflow ID
   */
  async deleteWorkflow(workflowId: string): Promise<void> {
    await this.request<void>({
      method: 'DELETE',
      url: `/api/workflows/${workflowId}`,
    });
  }

  /**
   * Execute a workflow
   *
   * @param workflowId - Workflow ID
   * @param inputData - Input data for workflow
   * @param stream - Whether to stream results
   * @returns Execution result
   *
   * @example
   * ```typescript
   * const result = await client.executeWorkflow('wf_123', {
   *   query: 'What is carbon footprint?',
   * });
   *
   * console.log(result.output_data);
   * ```
   */
  async executeWorkflow(
    workflowId: string,
    inputData: Record<string, any>,
    stream = false
  ): Promise<ExecutionResult> {
    return this.request<ExecutionResult>({
      method: 'POST',
      url: `/api/workflows/${workflowId}/execute`,
      data: { input_data: inputData, stream },
    });
  }

  // Agent Methods

  /**
   * Get agent by ID
   *
   * @param agentId - Agent ID
   * @returns Agent object
   */
  async getAgent(agentId: string): Promise<Agent> {
    return this.request<Agent>({
      method: 'GET',
      url: `/api/agents/${agentId}`,
    });
  }

  /**
   * List agents
   *
   * @param options - List options
   * @returns Array of agents
   *
   * @example
   * ```typescript
   * const agents = await client.listAgents({ category: 'carbon' });
   * ```
   */
  async listAgents(options: ListAgentsOptions = {}): Promise<Agent[]> {
    const response = await this.request<{ items: Agent[] }>({
      method: 'GET',
      url: '/api/agents',
      params: options,
    });
    return response.items;
  }

  /**
   * Execute an agent directly
   *
   * @param agentId - Agent ID
   * @param inputData - Input data for agent
   * @param config - Optional agent configuration
   * @returns Execution result
   *
   * @example
   * ```typescript
   * const result = await client.executeAgent('carbon_analyzer', {
   *   query: 'Calculate emissions',
   * });
   * ```
   */
  async executeAgent(
    agentId: string,
    inputData: Record<string, any>,
    config?: Record<string, any>
  ): Promise<ExecutionResult> {
    return this.request<ExecutionResult>({
      method: 'POST',
      url: `/api/agents/${agentId}/execute`,
      data: { input_data: inputData, config },
    });
  }

  // Execution Methods

  /**
   * Get execution result by ID
   *
   * @param executionId - Execution ID
   * @returns Execution result
   */
  async getExecution(executionId: string): Promise<ExecutionResult> {
    return this.request<ExecutionResult>({
      method: 'GET',
      url: `/api/executions/${executionId}`,
    });
  }

  /**
   * List execution results
   *
   * @param options - List options
   * @returns Array of execution results
   */
  async listExecutions(
    options: ListExecutionsOptions = {}
  ): Promise<ExecutionResult[]> {
    const response = await this.request<{ items: ExecutionResult[] }>({
      method: 'GET',
      url: '/api/executions',
      params: options,
    });
    return response.items;
  }

  // Citation Methods

  /**
   * Get citations for an execution
   *
   * @param executionId - Execution ID
   * @returns Array of citations
   */
  async getCitations(executionId: string): Promise<Citation[]> {
    const response = await this.request<{ citations: Citation[] }>({
      method: 'GET',
      url: `/api/executions/${executionId}/citations`,
    });
    return response.citations;
  }

  // Streaming Methods

  /**
   * Stream workflow execution results
   *
   * @param workflowId - Workflow ID
   * @param inputData - Input data for workflow
   * @param onChunk - Callback for each chunk
   *
   * @example
   * ```typescript
   * await client.streamExecution('wf_123', { query: 'test' }, (chunk) => {
   *   if (chunk.type === 'progress') {
   *     console.log(`Progress: ${chunk.percentage}%`);
   *   }
   * });
   * ```
   */
  async streamExecution(
    workflowId: string,
    inputData: Record<string, any>,
    onChunk: (chunk: StreamChunk) => void
  ): Promise<void> {
    const response = await this.client.post(
      `/api/workflows/${workflowId}/execute`,
      { input_data: inputData, stream: true },
      {
        responseType: 'stream',
        timeout: 0, // No timeout for streaming
      }
    );

    return new Promise((resolve, reject) => {
      response.data.on('data', (data: Buffer) => {
        const lines = data.toString().split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const chunk = JSON.parse(line.slice(6));
              onChunk(chunk);

              if (chunk.type === 'complete' || chunk.type === 'error') {
                resolve();
              }
            } catch (error) {
              reject(new APIError('Failed to parse stream chunk'));
            }
          }
        }
      });

      response.data.on('error', reject);
      response.data.on('end', resolve);
    });
  }

  // Utility Methods

  /**
   * Check API health status
   *
   * @returns Health status
   */
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    return this.request({
      method: 'GET',
      url: '/health',
    });
  }
}

/**
 * Create a GreenLang client
 *
 * @param apiKey - Your GreenLang API key
 * @param options - Additional options
 * @returns GreenLangClient instance
 *
 * @example
 * ```typescript
 * import { createClient } from '@greenlang/sdk';
 *
 * const client = createClient('gl_your_api_key');
 * ```
 */
export function createClient(
  apiKey: string,
  options?: Omit<ClientOptions, 'apiKey'>
): GreenLangClient {
  return new GreenLangClient({ apiKey, ...options });
}
