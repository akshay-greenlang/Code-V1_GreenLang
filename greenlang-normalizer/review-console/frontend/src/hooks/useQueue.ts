/**
 * useQueue Hook
 *
 * React Query hook for managing review queue data fetching and state.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useState, useCallback, useMemo } from 'react';
import { queueAPI } from '../api/client';
import type {
  QueueItem,
  QueueFilters,
  PaginationParams,
  PaginatedResponse,
} from '../api/types';

// Query keys for cache management
export const queueKeys = {
  all: ['queue'] as const,
  lists: () => [...queueKeys.all, 'list'] as const,
  list: (filters: QueueFilters, pagination: PaginationParams) =>
    [...queueKeys.lists(), { filters, pagination }] as const,
  details: () => [...queueKeys.all, 'detail'] as const,
  detail: (id: string) => [...queueKeys.details(), id] as const,
};

/**
 * Default pagination settings
 */
const DEFAULT_PAGINATION: PaginationParams = {
  page: 1,
  perPage: 25,
  sortBy: 'createdAt',
  sortDirection: 'desc',
};

/**
 * Hook for fetching and managing the review queue list
 */
export function useQueueList(initialFilters: QueueFilters = {}) {
  const queryClient = useQueryClient();
  const [filters, setFilters] = useState<QueueFilters>(initialFilters);
  const [pagination, setPagination] = useState<PaginationParams>(DEFAULT_PAGINATION);

  // Fetch queue items
  const query = useQuery<PaginatedResponse<QueueItem>>({
    queryKey: queueKeys.list(filters, pagination),
    queryFn: () => queueAPI.getQueue(filters, pagination),
    staleTime: 30000, // 30 seconds
    refetchInterval: 60000, // Refetch every minute
  });

  // Update filters
  const updateFilters = useCallback((newFilters: Partial<QueueFilters>) => {
    setFilters((prev) => ({ ...prev, ...newFilters }));
    setPagination((prev) => ({ ...prev, page: 1 })); // Reset to first page
  }, []);

  // Clear filters
  const clearFilters = useCallback(() => {
    setFilters({});
    setPagination(DEFAULT_PAGINATION);
  }, []);

  // Change page
  const setPage = useCallback((page: number) => {
    setPagination((prev) => ({ ...prev, page }));
  }, []);

  // Change items per page
  const setPerPage = useCallback((perPage: number) => {
    setPagination((prev) => ({ ...prev, perPage, page: 1 }));
  }, []);

  // Change sort
  const setSort = useCallback((sortBy: string, sortDirection: 'asc' | 'desc') => {
    setPagination((prev) => ({ ...prev, sortBy, sortDirection }));
  }, []);

  // Refresh data
  const refresh = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: queueKeys.lists() });
  }, [queryClient]);

  return {
    // Data
    items: query.data?.items ?? [],
    pagination: query.data?.pagination ?? {
      page: 1,
      perPage: 25,
      totalItems: 0,
      totalPages: 0,
      hasMore: false,
    },
    filters,

    // State
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    error: query.error,

    // Actions
    updateFilters,
    clearFilters,
    setPage,
    setPerPage,
    setSort,
    refresh,
  };
}

/**
 * Hook for fetching a single queue item
 */
export function useQueueItem(id: string | undefined) {
  const query = useQuery<QueueItem>({
    queryKey: queueKeys.detail(id ?? ''),
    queryFn: () => queueAPI.getQueueItem(id!),
    enabled: !!id,
    staleTime: 10000, // 10 seconds
  });

  return {
    item: query.data,
    isLoading: query.isLoading,
    error: query.error,
    refetch: query.refetch,
  };
}

/**
 * Hook for claiming queue items
 */
export function useClaimItem() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => queueAPI.claimItem(id),
    onSuccess: (data) => {
      // Update the item in cache
      queryClient.setQueryData(queueKeys.detail(data.id), data);
      // Invalidate list to refresh
      queryClient.invalidateQueries({ queryKey: queueKeys.lists() });
    },
  });
}

/**
 * Hook for releasing queue items
 */
export function useReleaseItem() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => queueAPI.releaseItem(id),
    onSuccess: (data) => {
      queryClient.setQueryData(queueKeys.detail(data.id), data);
      queryClient.invalidateQueries({ queryKey: queueKeys.lists() });
    },
  });
}

/**
 * Hook for getting the next item in queue
 */
export function useNextItem() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (filters?: QueueFilters) => queueAPI.getNextItem(filters),
    onSuccess: (data) => {
      if (data) {
        queryClient.setQueryData(queueKeys.detail(data.id), data);
      }
      queryClient.invalidateQueries({ queryKey: queueKeys.lists() });
    },
  });
}

/**
 * Hook for skipping queue items
 */
export function useSkipItem() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, reason }: { id: string; reason?: string }) =>
      queueAPI.skipItem(id, reason),
    onSuccess: (data) => {
      queryClient.setQueryData(queueKeys.detail(data.id), data);
      queryClient.invalidateQueries({ queryKey: queueKeys.lists() });
    },
  });
}

/**
 * Hook for escalating queue items
 */
export function useEscalateItem() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, reason }: { id: string; reason: string }) =>
      queueAPI.escalateItem(id, reason),
    onSuccess: (data) => {
      queryClient.setQueryData(queueKeys.detail(data.id), data);
      queryClient.invalidateQueries({ queryKey: queueKeys.lists() });
    },
  });
}

/**
 * Hook that combines queue functionality for power-user workflow
 */
export function useReviewWorkflow() {
  const queryClient = useQueryClient();
  const [currentItem, setCurrentItem] = useState<QueueItem | null>(null);
  const [filters, setFilters] = useState<QueueFilters>({});

  const nextItemMutation = useNextItem();
  const skipMutation = useSkipItem();

  // Get next item and set as current
  const getNext = useCallback(async () => {
    const item = await nextItemMutation.mutateAsync(filters);
    setCurrentItem(item);
    return item;
  }, [nextItemMutation, filters]);

  // Skip current and get next
  const skipAndNext = useCallback(async (reason?: string) => {
    if (currentItem) {
      await skipMutation.mutateAsync({ id: currentItem.id, reason });
    }
    return getNext();
  }, [currentItem, skipMutation, getNext]);

  // Clear current item
  const clearCurrent = useCallback(() => {
    setCurrentItem(null);
  }, []);

  // After resolution, get next
  const afterResolution = useCallback(async () => {
    queryClient.invalidateQueries({ queryKey: queueKeys.lists() });
    return getNext();
  }, [queryClient, getNext]);

  return {
    currentItem,
    setCurrentItem,
    filters,
    setFilters,
    getNext,
    skipAndNext,
    clearCurrent,
    afterResolution,
    isLoadingNext: nextItemMutation.isPending,
    isSkipping: skipMutation.isPending,
  };
}
