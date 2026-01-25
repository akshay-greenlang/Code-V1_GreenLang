/**
 * GreenLang Agent Factory - Marketplace Store
 *
 * Zustand store for marketplace state management.
 * Handles filters, view preferences, comparison list, and user interactions.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type {
  AgentCategory,
  AgentPricingTier,
  SortOption,
  MarketplaceFilters,
} from '@/api/types/marketplace';

// ============================================================================
// Types
// ============================================================================

export type ViewMode = 'grid' | 'list';

export interface MarketplaceState {
  // Filters
  filters: MarketplaceFilters;

  // View preferences
  viewMode: ViewMode;

  // Comparison
  compareList: string[];
  maxCompareItems: number;

  // User history
  recentlyViewed: string[];
  maxRecentlyViewed: number;

  // Favorites (synced with server, but cached locally)
  favorites: string[];

  // Search
  searchSuggestions: string[];
  isSearchOpen: boolean;

  // UI state
  isCategoryNavCollapsed: boolean;
  activeTab: string;
}

export interface MarketplaceActions {
  // Filter actions
  setCategory: (category: AgentCategory | undefined) => void;
  setSearchQuery: (query: string) => void;
  setSortBy: (sortBy: SortOption) => void;
  setTags: (tags: string[]) => void;
  addTag: (tag: string) => void;
  removeTag: (tag: string) => void;
  setPricingTier: (tier: AgentPricingTier | undefined) => void;
  setVerifiedOnly: (verified: boolean | undefined) => void;
  setRegulatoryFramework: (framework: string | undefined) => void;
  resetFilters: () => void;

  // View actions
  setViewMode: (mode: ViewMode) => void;
  toggleViewMode: () => void;

  // Comparison actions
  addToCompare: (agentId: string) => boolean;
  removeFromCompare: (agentId: string) => void;
  clearCompare: () => void;
  isInCompareList: (agentId: string) => boolean;

  // Recently viewed actions
  addToRecentlyViewed: (agentId: string) => void;
  clearRecentlyViewed: () => void;

  // Favorites actions (local cache - syncs with server via React Query)
  addToFavorites: (agentId: string) => void;
  removeFromFavorites: (agentId: string) => void;
  toggleFavorite: (agentId: string) => void;
  isFavorite: (agentId: string) => boolean;
  setFavorites: (favorites: string[]) => void;

  // Search actions
  setSearchSuggestions: (suggestions: string[]) => void;
  setSearchOpen: (open: boolean) => void;

  // UI actions
  toggleCategoryNav: () => void;
  setCategoryNavCollapsed: (collapsed: boolean) => void;
  setActiveTab: (tab: string) => void;
}

// ============================================================================
// Default State
// ============================================================================

const defaultFilters: MarketplaceFilters = {
  search: '',
  sortBy: 'popularity',
  tags: [],
  category: undefined,
  pricingTier: undefined,
  isVerified: undefined,
  regulatoryFramework: undefined,
};

const defaultState: MarketplaceState = {
  filters: defaultFilters,
  viewMode: 'grid',
  compareList: [],
  maxCompareItems: 4,
  recentlyViewed: [],
  maxRecentlyViewed: 10,
  favorites: [],
  searchSuggestions: [],
  isSearchOpen: false,
  isCategoryNavCollapsed: false,
  activeTab: 'overview',
};

// ============================================================================
// Store
// ============================================================================

export const useMarketplaceStore = create<MarketplaceState & MarketplaceActions>()(
  persist(
    (set, get) => ({
      ...defaultState,

      // ========================================================================
      // Filter Actions
      // ========================================================================

      setCategory: (category) => {
        set((state) => ({
          filters: { ...state.filters, category },
        }));
      },

      setSearchQuery: (search) => {
        set((state) => ({
          filters: { ...state.filters, search },
        }));
      },

      setSortBy: (sortBy) => {
        set((state) => ({
          filters: { ...state.filters, sortBy },
        }));
      },

      setTags: (tags) => {
        set((state) => ({
          filters: { ...state.filters, tags },
        }));
      },

      addTag: (tag) => {
        const currentTags = get().filters.tags;
        if (!currentTags.includes(tag)) {
          set((state) => ({
            filters: { ...state.filters, tags: [...currentTags, tag] },
          }));
        }
      },

      removeTag: (tag) => {
        set((state) => ({
          filters: {
            ...state.filters,
            tags: state.filters.tags.filter((t) => t !== tag),
          },
        }));
      },

      setPricingTier: (pricingTier) => {
        set((state) => ({
          filters: { ...state.filters, pricingTier },
        }));
      },

      setVerifiedOnly: (isVerified) => {
        set((state) => ({
          filters: { ...state.filters, isVerified },
        }));
      },

      setRegulatoryFramework: (regulatoryFramework) => {
        set((state) => ({
          filters: { ...state.filters, regulatoryFramework },
        }));
      },

      resetFilters: () => {
        set({ filters: defaultFilters });
      },

      // ========================================================================
      // View Actions
      // ========================================================================

      setViewMode: (viewMode) => {
        set({ viewMode });
      },

      toggleViewMode: () => {
        set((state) => ({
          viewMode: state.viewMode === 'grid' ? 'list' : 'grid',
        }));
      },

      // ========================================================================
      // Comparison Actions
      // ========================================================================

      addToCompare: (agentId) => {
        const { compareList, maxCompareItems } = get();
        if (compareList.includes(agentId)) {
          return false;
        }
        if (compareList.length >= maxCompareItems) {
          return false;
        }
        set({ compareList: [...compareList, agentId] });
        return true;
      },

      removeFromCompare: (agentId) => {
        set((state) => ({
          compareList: state.compareList.filter((id) => id !== agentId),
        }));
      },

      clearCompare: () => {
        set({ compareList: [] });
      },

      isInCompareList: (agentId) => {
        return get().compareList.includes(agentId);
      },

      // ========================================================================
      // Recently Viewed Actions
      // ========================================================================

      addToRecentlyViewed: (agentId) => {
        const { recentlyViewed, maxRecentlyViewed } = get();

        // Remove if already exists (will be added to front)
        const filtered = recentlyViewed.filter((id) => id !== agentId);

        // Add to front and limit size
        const updated = [agentId, ...filtered].slice(0, maxRecentlyViewed);

        set({ recentlyViewed: updated });
      },

      clearRecentlyViewed: () => {
        set({ recentlyViewed: [] });
      },

      // ========================================================================
      // Favorites Actions
      // ========================================================================

      addToFavorites: (agentId) => {
        const { favorites } = get();
        if (!favorites.includes(agentId)) {
          set({ favorites: [...favorites, agentId] });
        }
      },

      removeFromFavorites: (agentId) => {
        set((state) => ({
          favorites: state.favorites.filter((id) => id !== agentId),
        }));
      },

      toggleFavorite: (agentId) => {
        const { favorites } = get();
        if (favorites.includes(agentId)) {
          get().removeFromFavorites(agentId);
        } else {
          get().addToFavorites(agentId);
        }
      },

      isFavorite: (agentId) => {
        return get().favorites.includes(agentId);
      },

      setFavorites: (favorites) => {
        set({ favorites });
      },

      // ========================================================================
      // Search Actions
      // ========================================================================

      setSearchSuggestions: (searchSuggestions) => {
        set({ searchSuggestions });
      },

      setSearchOpen: (isSearchOpen) => {
        set({ isSearchOpen });
      },

      // ========================================================================
      // UI Actions
      // ========================================================================

      toggleCategoryNav: () => {
        set((state) => ({
          isCategoryNavCollapsed: !state.isCategoryNavCollapsed,
        }));
      },

      setCategoryNavCollapsed: (isCategoryNavCollapsed) => {
        set({ isCategoryNavCollapsed });
      },

      setActiveTab: (activeTab) => {
        set({ activeTab });
      },
    }),
    {
      name: 'gl-marketplace-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        // Only persist these fields
        viewMode: state.viewMode,
        recentlyViewed: state.recentlyViewed,
        favorites: state.favorites,
        isCategoryNavCollapsed: state.isCategoryNavCollapsed,
        // Don't persist: filters, compareList, searchSuggestions, isSearchOpen
      }),
    }
  )
);

// ============================================================================
// Selectors
// ============================================================================

export const selectFilters = (state: MarketplaceState) => state.filters;
export const selectViewMode = (state: MarketplaceState) => state.viewMode;
export const selectCompareList = (state: MarketplaceState) => state.compareList;
export const selectCompareCount = (state: MarketplaceState) => state.compareList.length;
export const selectCanAddToCompare = (state: MarketplaceState) =>
  state.compareList.length < state.maxCompareItems;
export const selectRecentlyViewed = (state: MarketplaceState) => state.recentlyViewed;
export const selectFavorites = (state: MarketplaceState) => state.favorites;
export const selectSearchQuery = (state: MarketplaceState) => state.filters.search;
export const selectCategory = (state: MarketplaceState) => state.filters.category;
export const selectSortBy = (state: MarketplaceState) => state.filters.sortBy;
export const selectTags = (state: MarketplaceState) => state.filters.tags;
export const selectIsSearchOpen = (state: MarketplaceState) => state.isSearchOpen;
export const selectIsCategoryNavCollapsed = (state: MarketplaceState) =>
  state.isCategoryNavCollapsed;
