/**
 * GreenLang Agent Factory - Marketplace React Query Hooks
 *
 * React Query hooks for the Agent Marketplace API.
 * Provides data fetching, caching, and mutations for marketplace operations.
 */

import {
  useQuery,
  useMutation,
  useQueryClient,
  useInfiniteQuery,
  type UseQueryOptions,
  type UseMutationOptions,
} from '@tanstack/react-query';
import { marketplaceAPI } from '../marketplace';
import type {
  MarketplaceAgent,
  AgentVersion,
  AgentReview,
  ReviewSummary,
  AgentDeployment,
  AgentSearchResult,
  AgentListParams,
  DeployAgentRequest,
  DeployAgentResponse,
  CreateReviewRequest,
  AgentComparison,
  AgentMarketplaceMetrics,
} from '../types/marketplace';
import type { PaginatedResponse } from '../types';

// ============================================================================
// Query Keys
// ============================================================================

export const marketplaceKeys = {
  all: ['marketplace'] as const,
  agents: () => [...marketplaceKeys.all, 'agents'] as const,
  agentsList: (params?: AgentListParams) => [...marketplaceKeys.agents(), 'list', params] as const,
  agentsSearch: (query: string) => [...marketplaceKeys.agents(), 'search', query] as const,
  agentsFeatured: () => [...marketplaceKeys.agents(), 'featured'] as const,
  agentsTrending: () => [...marketplaceKeys.agents(), 'trending'] as const,
  agentsNew: () => [...marketplaceKeys.agents(), 'new'] as const,
  agent: (id: string) => [...marketplaceKeys.agents(), 'detail', id] as const,
  agentRelated: (id: string) => [...marketplaceKeys.agent(id), 'related'] as const,
  agentVersions: (id: string) => [...marketplaceKeys.agent(id), 'versions'] as const,
  agentVersion: (id: string, version: string) => [...marketplaceKeys.agentVersions(id), version] as const,
  agentReviews: (id: string) => [...marketplaceKeys.agent(id), 'reviews'] as const,
  agentReviewSummary: (id: string) => [...marketplaceKeys.agent(id), 'reviewSummary'] as const,
  agentMetrics: (id: string) => [...marketplaceKeys.agent(id), 'metrics'] as const,
  deployments: () => [...marketplaceKeys.all, 'deployments'] as const,
  deployment: (id: string) => [...marketplaceKeys.deployments(), id] as const,
  deploymentMetrics: (id: string) => [...marketplaceKeys.deployment(id), 'metrics'] as const,
  categories: () => [...marketplaceKeys.all, 'categories'] as const,
  tags: () => [...marketplaceKeys.all, 'tags'] as const,
  frameworks: () => [...marketplaceKeys.all, 'frameworks'] as const,
  favorites: () => [...marketplaceKeys.all, 'favorites'] as const,
  comparison: (ids: string[]) => [...marketplaceKeys.all, 'comparison', ids.sort().join(',')] as const,
};

// ============================================================================
// Agent Discovery Hooks
// ============================================================================

/**
 * Hook for fetching paginated list of marketplace agents
 */
export function useAgents(
  params?: AgentListParams,
  options?: Omit<UseQueryOptions<AgentSearchResult>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: marketplaceKeys.agentsList(params),
    queryFn: () => marketplaceAPI.getAgents(params),
    staleTime: 1000 * 60 * 5, // 5 minutes
    ...options,
  });
}

/**
 * Hook for infinite scrolling agent list
 */
export function useInfiniteAgents(params?: Omit<AgentListParams, 'page'>) {
  return useInfiniteQuery({
    queryKey: [...marketplaceKeys.agentsList(params), 'infinite'],
    queryFn: ({ pageParam = 1 }) =>
      marketplaceAPI.getAgents({ ...params, page: pageParam }),
    getNextPageParam: (lastPage) => {
      const { page, totalPages } = lastPage.pagination;
      return page < totalPages ? page + 1 : undefined;
    },
    initialPageParam: 1,
    staleTime: 1000 * 60 * 5,
  });
}

/**
 * Hook for fetching a single agent by ID or slug
 */
export function useAgent(
  idOrSlug: string,
  options?: Omit<UseQueryOptions<MarketplaceAgent>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: marketplaceKeys.agent(idOrSlug),
    queryFn: () => marketplaceAPI.getAgent(idOrSlug),
    staleTime: 1000 * 60 * 10, // 10 minutes
    enabled: !!idOrSlug,
    ...options,
  });
}

/**
 * Hook for searching agents with autocomplete
 */
export function useAgentSearch(
  query: string,
  options?: Omit<
    UseQueryOptions<{ agents: MarketplaceAgent[]; suggestions: string[] }>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery({
    queryKey: marketplaceKeys.agentsSearch(query),
    queryFn: () => marketplaceAPI.searchAgents(query),
    enabled: query.length >= 2,
    staleTime: 1000 * 60 * 2, // 2 minutes
    ...options,
  });
}

/**
 * Hook for fetching featured agents
 */
export function useFeaturedAgents(
  options?: Omit<UseQueryOptions<MarketplaceAgent[]>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: marketplaceKeys.agentsFeatured(),
    queryFn: () => marketplaceAPI.getFeaturedAgents(),
    staleTime: 1000 * 60 * 15, // 15 minutes
    ...options,
  });
}

/**
 * Hook for fetching trending agents
 */
export function useTrendingAgents(
  limit?: number,
  options?: Omit<UseQueryOptions<MarketplaceAgent[]>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: marketplaceKeys.agentsTrending(),
    queryFn: () => marketplaceAPI.getTrendingAgents(limit),
    staleTime: 1000 * 60 * 10, // 10 minutes
    ...options,
  });
}

/**
 * Hook for fetching new agents
 */
export function useNewAgents(
  limit?: number,
  options?: Omit<UseQueryOptions<MarketplaceAgent[]>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: marketplaceKeys.agentsNew(),
    queryFn: () => marketplaceAPI.getNewAgents(limit),
    staleTime: 1000 * 60 * 10,
    ...options,
  });
}

/**
 * Hook for fetching related agents
 */
export function useRelatedAgents(
  agentId: string,
  limit?: number,
  options?: Omit<UseQueryOptions<MarketplaceAgent[]>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: marketplaceKeys.agentRelated(agentId),
    queryFn: () => marketplaceAPI.getRelatedAgents(agentId, limit),
    enabled: !!agentId,
    staleTime: 1000 * 60 * 10,
    ...options,
  });
}

// ============================================================================
// Agent Version Hooks
// ============================================================================

/**
 * Hook for fetching agent version history
 */
export function useAgentVersions(
  agentId: string,
  params?: { page?: number; perPage?: number },
  options?: Omit<UseQueryOptions<PaginatedResponse<AgentVersion>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: marketplaceKeys.agentVersions(agentId),
    queryFn: () => marketplaceAPI.getAgentVersions(agentId, params),
    enabled: !!agentId,
    staleTime: 1000 * 60 * 15,
    ...options,
  });
}

/**
 * Hook for fetching a specific agent version
 */
export function useAgentVersion(
  agentId: string,
  version: string,
  options?: Omit<UseQueryOptions<AgentVersion>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: marketplaceKeys.agentVersion(agentId, version),
    queryFn: () => marketplaceAPI.getAgentVersion(agentId, version),
    enabled: !!agentId && !!version,
    staleTime: 1000 * 60 * 30, // 30 minutes - versions don't change
    ...options,
  });
}

// ============================================================================
// Review Hooks
// ============================================================================

/**
 * Hook for fetching agent reviews
 */
export function useAgentReviews(
  agentId: string,
  params?: {
    page?: number;
    perPage?: number;
    sortBy?: 'newest' | 'oldest' | 'highest' | 'lowest' | 'helpful';
  },
  options?: Omit<UseQueryOptions<PaginatedResponse<AgentReview>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: [...marketplaceKeys.agentReviews(agentId), params],
    queryFn: () => marketplaceAPI.getAgentReviews(agentId, params),
    enabled: !!agentId,
    staleTime: 1000 * 60 * 5,
    ...options,
  });
}

/**
 * Hook for fetching review summary
 */
export function useReviewSummary(
  agentId: string,
  options?: Omit<UseQueryOptions<ReviewSummary>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: marketplaceKeys.agentReviewSummary(agentId),
    queryFn: () => marketplaceAPI.getReviewSummary(agentId),
    enabled: !!agentId,
    staleTime: 1000 * 60 * 10,
    ...options,
  });
}

/**
 * Hook for creating a review
 */
export function useCreateReview(
  options?: UseMutationOptions<AgentReview, Error, CreateReviewRequest>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreateReviewRequest) => marketplaceAPI.createReview(data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: marketplaceKeys.agentReviews(variables.agentId),
      });
      queryClient.invalidateQueries({
        queryKey: marketplaceKeys.agentReviewSummary(variables.agentId),
      });
      queryClient.invalidateQueries({
        queryKey: marketplaceKeys.agent(variables.agentId),
      });
    },
    ...options,
  });
}

/**
 * Hook for marking a review as helpful
 */
export function useMarkReviewHelpful(
  options?: UseMutationOptions<AgentReview, Error, { agentId: string; reviewId: string }>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ agentId, reviewId }) =>
      marketplaceAPI.markReviewHelpful(agentId, reviewId),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: marketplaceKeys.agentReviews(variables.agentId),
      });
    },
    ...options,
  });
}

// ============================================================================
// Deployment Hooks
// ============================================================================

/**
 * Hook for deploying an agent
 */
export function useDeployAgent(
  options?: UseMutationOptions<DeployAgentResponse, Error, DeployAgentRequest>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: DeployAgentRequest) => marketplaceAPI.deployAgent(data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: marketplaceKeys.deployments() });
      queryClient.invalidateQueries({
        queryKey: marketplaceKeys.agent(variables.agentId),
      });
    },
    ...options,
  });
}

/**
 * Hook for fetching user's deployments
 */
export function useMyDeployments(
  params?: { page?: number; perPage?: number; status?: string },
  options?: Omit<UseQueryOptions<PaginatedResponse<AgentDeployment>>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: [...marketplaceKeys.deployments(), params],
    queryFn: () => marketplaceAPI.getMyDeployments(params),
    staleTime: 1000 * 60 * 2,
    ...options,
  });
}

/**
 * Hook for fetching a specific deployment
 */
export function useDeployment(
  deploymentId: string,
  options?: Omit<UseQueryOptions<AgentDeployment>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: marketplaceKeys.deployment(deploymentId),
    queryFn: () => marketplaceAPI.getDeployment(deploymentId),
    enabled: !!deploymentId,
    staleTime: 1000 * 60 * 2,
    ...options,
  });
}

/**
 * Hook for fetching deployment metrics
 */
export function useDeploymentMetrics(
  deploymentId: string,
  params?: { startDate?: string; endDate?: string },
  options?: Omit<
    UseQueryOptions<{
      requests: { date: string; count: number }[];
      responseTime: { date: string; avg: number }[];
      errors: { date: string; count: number }[];
    }>,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery({
    queryKey: [...marketplaceKeys.deploymentMetrics(deploymentId), params],
    queryFn: () => marketplaceAPI.getDeploymentMetrics(deploymentId, params),
    enabled: !!deploymentId,
    staleTime: 1000 * 60 * 1, // 1 minute for metrics
    refetchInterval: 1000 * 60, // Refetch every minute
    ...options,
  });
}

/**
 * Hook for fetching agent marketplace metrics (performance data)
 */
export function useAgentMetrics(
  agentId: string,
  options?: Omit<UseQueryOptions<AgentMarketplaceMetrics>, 'queryKey' | 'queryFn'>
) {
  const { data: agent } = useAgent(agentId);

  return useQuery({
    queryKey: marketplaceKeys.agentMetrics(agentId),
    queryFn: async () => {
      // Agent metrics are embedded in the agent data
      // If we need separate endpoint, we can add it here
      const agentData = await marketplaceAPI.getAgent(agentId);
      return agentData.metrics;
    },
    enabled: !!agentId,
    staleTime: 1000 * 60 * 5,
    ...options,
  });
}

/**
 * Hook for undeploying an agent
 */
export function useUndeployAgent(
  options?: UseMutationOptions<void, Error, string>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (deploymentId: string) => marketplaceAPI.undeployAgent(deploymentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: marketplaceKeys.deployments() });
    },
    ...options,
  });
}

// ============================================================================
// Comparison Hook
// ============================================================================

/**
 * Hook for comparing multiple agents
 */
export function useAgentComparison(
  agentIds: string[],
  options?: Omit<UseQueryOptions<AgentComparison>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: marketplaceKeys.comparison(agentIds),
    queryFn: () => marketplaceAPI.compareAgents(agentIds),
    enabled: agentIds.length >= 2,
    staleTime: 1000 * 60 * 10,
    ...options,
  });
}

// ============================================================================
// Category & Tag Hooks
// ============================================================================

/**
 * Hook for fetching categories
 */
export function useCategories(
  options?: Omit<
    UseQueryOptions<
      {
        category: string;
        displayName: string;
        description: string;
        icon: string;
        agentCount: number;
      }[]
    >,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery({
    queryKey: marketplaceKeys.categories(),
    queryFn: () => marketplaceAPI.getCategories(),
    staleTime: 1000 * 60 * 30, // 30 minutes - categories rarely change
    ...options,
  });
}

/**
 * Hook for fetching tags
 */
export function useTags(
  options?: Omit<UseQueryOptions<{ tag: string; count: number }[]>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: marketplaceKeys.tags(),
    queryFn: () => marketplaceAPI.getTags(),
    staleTime: 1000 * 60 * 15,
    ...options,
  });
}

/**
 * Hook for fetching regulatory frameworks
 */
export function useRegulatoryFrameworks(
  options?: Omit<
    UseQueryOptions<
      {
        framework: string;
        displayName: string;
        description: string;
        deadline?: string;
        agentCount: number;
      }[]
    >,
    'queryKey' | 'queryFn'
  >
) {
  return useQuery({
    queryKey: marketplaceKeys.frameworks(),
    queryFn: () => marketplaceAPI.getRegulatoryFrameworks(),
    staleTime: 1000 * 60 * 30,
    ...options,
  });
}

// ============================================================================
// Favorites Hooks
// ============================================================================

/**
 * Hook for fetching user's favorites
 */
export function useFavorites(
  options?: Omit<UseQueryOptions<MarketplaceAgent[]>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: marketplaceKeys.favorites(),
    queryFn: () => marketplaceAPI.getFavorites(),
    staleTime: 1000 * 60 * 5,
    ...options,
  });
}

/**
 * Hook for adding agent to favorites
 */
export function useAddToFavorites(
  options?: UseMutationOptions<void, Error, string>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (agentId: string) => marketplaceAPI.addToFavorites(agentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: marketplaceKeys.favorites() });
    },
    ...options,
  });
}

/**
 * Hook for removing agent from favorites
 */
export function useRemoveFromFavorites(
  options?: UseMutationOptions<void, Error, string>
) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (agentId: string) => marketplaceAPI.removeFromFavorites(agentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: marketplaceKeys.favorites() });
    },
    ...options,
  });
}
