/**
 * AgentCatalog Page
 *
 * Main marketplace page showing all available agents with filtering,
 * search, sorting, and pagination. Supports grid and list view modes.
 */

import * as React from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  Search,
  Grid3X3,
  List,
  SlidersHorizontal,
  X,
  ChevronDown,
  GitCompare,
  Sparkles,
  TrendingUp,
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { Input } from '@/components/ui/Input';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/Select';
import { Pagination, PaginationInfo } from '@/components/ui/Pagination';
import { AgentCard, AgentCardSkeleton } from '@/components/marketplace/AgentCard';
import { CategoryNav } from '@/components/marketplace/CategoryNav';
import { useMarketplaceStore } from '@/stores/marketplaceStore';
import {
  useAgents,
  useAgentSearch,
  useFeaturedAgents,
  useTrendingAgents,
} from '@/api/hooks/marketplace';
import type { MarketplaceAgent, SortOption, AgentListParams } from '@/api/types/marketplace';

// ============================================================================
// Types
// ============================================================================

const sortOptions: { value: SortOption; label: string }[] = [
  { value: 'popularity', label: 'Most Popular' },
  { value: 'rating', label: 'Highest Rated' },
  { value: 'newest', label: 'Newest First' },
  { value: 'name_asc', label: 'Name (A-Z)' },
  { value: 'name_desc', label: 'Name (Z-A)' },
  { value: 'regulatory_deadline', label: 'Regulatory Deadline' },
];

// ============================================================================
// Search Autocomplete Component
// ============================================================================

interface SearchAutocompleteProps {
  value: string;
  onChange: (value: string) => void;
  onSelect: (agent: MarketplaceAgent) => void;
  className?: string;
}

function SearchAutocomplete({
  value,
  onChange,
  onSelect,
  className,
}: SearchAutocompleteProps) {
  const [isFocused, setIsFocused] = React.useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);
  const { data: searchResults, isLoading } = useAgentSearch(value);

  const showDropdown = isFocused && value.length >= 2 && (searchResults?.agents?.length || searchResults?.suggestions?.length);

  return (
    <div className={cn('relative', className)}>
      <Input
        ref={inputRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setTimeout(() => setIsFocused(false), 200)}
        placeholder="Search agents..."
        leftIcon={<Search className="h-4 w-4" />}
        rightIcon={
          value && (
            <button
              onClick={() => onChange('')}
              className="rounded-full p-0.5 hover:bg-muted"
            >
              <X className="h-3 w-3" />
            </button>
          )
        }
        className="pr-10"
      />

      {/* Autocomplete Dropdown */}
      {showDropdown && (
        <div className="absolute left-0 right-0 top-full z-50 mt-1 overflow-hidden rounded-lg border bg-popover shadow-lg">
          {isLoading ? (
            <div className="p-4 text-center text-sm text-muted-foreground">
              Searching...
            </div>
          ) : (
            <>
              {/* Agent Results */}
              {searchResults?.agents && searchResults.agents.length > 0 && (
                <div className="border-b p-2">
                  <p className="mb-2 px-2 text-xs font-medium text-muted-foreground">
                    Agents
                  </p>
                  {searchResults.agents.slice(0, 5).map((agent) => (
                    <button
                      key={agent.id}
                      onClick={() => onSelect(agent)}
                      className="flex w-full items-center gap-3 rounded-md px-2 py-2 text-left hover:bg-muted"
                    >
                      <span className="text-xl">{agent.icon}</span>
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-medium">
                          {agent.name}
                        </p>
                        <p className="truncate text-xs text-muted-foreground">
                          {agent.shortDescription}
                        </p>
                      </div>
                      <Badge variant="outline" size="sm">
                        {agent.category}
                      </Badge>
                    </button>
                  ))}
                </div>
              )}

              {/* Suggestions */}
              {searchResults?.suggestions && searchResults.suggestions.length > 0 && (
                <div className="p-2">
                  <p className="mb-2 px-2 text-xs font-medium text-muted-foreground">
                    Suggestions
                  </p>
                  {searchResults.suggestions.map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => onChange(suggestion)}
                      className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-muted"
                    >
                      <Search className="h-3 w-3 text-muted-foreground" />
                      {suggestion}
                    </button>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Compare Bar Component
// ============================================================================

function CompareBar() {
  const navigate = useNavigate();
  const { compareList, clearCompare, removeFromCompare } = useMarketplaceStore();
  const { data: agentsData } = useAgents({
    // We'd need to filter by IDs, but for now this is a placeholder
  });

  if (compareList.length === 0) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-40 border-t bg-card shadow-lg">
      <div className="container mx-auto flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-4">
          <GitCompare className="h-5 w-5 text-primary" />
          <span className="font-medium">
            Compare ({compareList.length}/4 agents selected)
          </span>
          <div className="flex gap-2">
            {compareList.map((id) => (
              <Badge
                key={id}
                variant="secondary"
                className="cursor-pointer gap-1"
                onClick={() => removeFromCompare(id)}
              >
                {id.slice(0, 8)}...
                <X className="h-3 w-3" />
              </Badge>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={clearCompare}>
            Clear
          </Button>
          <Button
            variant="primary"
            size="sm"
            disabled={compareList.length < 2}
            onClick={() => navigate(`/marketplace/compare?agents=${compareList.join(',')}`)}
          >
            Compare Now
          </Button>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Featured Agents Section
// ============================================================================

function FeaturedAgentsSection() {
  const { data: featured, isLoading } = useFeaturedAgents();
  const { viewMode } = useMarketplaceStore();

  if (isLoading) {
    return (
      <div className="mb-8">
        <div className="mb-4 flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-amber-500" />
          <h2 className="text-lg font-semibold">Featured Agents</h2>
        </div>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <AgentCardSkeleton key={i} viewMode="grid" />
          ))}
        </div>
      </div>
    );
  }

  if (!featured || featured.length === 0) return null;

  return (
    <div className="mb-8">
      <div className="mb-4 flex items-center gap-2">
        <Sparkles className="h-5 w-5 text-amber-500" />
        <h2 className="text-lg font-semibold">Featured Agents</h2>
      </div>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
        {featured.slice(0, 3).map((agent) => (
          <AgentCard key={agent.id} agent={agent} viewMode="grid" />
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Trending Agents Section
// ============================================================================

function TrendingAgentsSection() {
  const { data: trending, isLoading } = useTrendingAgents(6);

  if (isLoading || !trending || trending.length === 0) return null;

  return (
    <div className="mb-8 rounded-lg border bg-gradient-to-r from-emerald-50 to-cyan-50 p-4 dark:from-emerald-950/20 dark:to-cyan-950/20">
      <div className="mb-3 flex items-center gap-2">
        <TrendingUp className="h-5 w-5 text-emerald-600" />
        <h2 className="text-sm font-semibold">Trending This Week</h2>
      </div>
      <div className="flex gap-3 overflow-x-auto pb-2">
        {trending.map((agent) => (
          <a
            key={agent.id}
            href={`/marketplace/agents/${agent.slug}`}
            className="flex flex-shrink-0 items-center gap-2 rounded-full border bg-white px-3 py-1.5 text-sm hover:border-primary/50 dark:bg-card"
          >
            <span>{agent.icon}</span>
            <span className="font-medium">{agent.name}</span>
          </a>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Agent Catalog Page
// ============================================================================

export function AgentCatalog() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [isMobileFilterOpen, setIsMobileFilterOpen] = React.useState(false);

  const {
    filters,
    viewMode,
    setViewMode,
    setSearchQuery,
    setSortBy,
    compareList,
  } = useMarketplaceStore();

  // Parse URL params
  const page = parseInt(searchParams.get('page') || '1', 10);
  const perPage = parseInt(searchParams.get('perPage') || '12', 10);

  // Build query params
  const queryParams: AgentListParams = React.useMemo(
    () => ({
      page,
      perPage,
      category: filters.category,
      search: filters.search || undefined,
      sortBy: filters.sortBy,
      tags: filters.tags.length > 0 ? filters.tags : undefined,
      pricingTier: filters.pricingTier,
      isVerified: filters.isVerified,
      regulatoryFramework: filters.regulatoryFramework,
    }),
    [page, perPage, filters]
  );

  // Fetch agents
  const {
    data: agentsData,
    isLoading,
    isError,
    error,
  } = useAgents(queryParams);

  const agents = agentsData?.agents || [];
  const pagination = agentsData?.pagination;
  const totalPages = pagination?.totalPages || 1;

  // Handle page change
  const handlePageChange = (newPage: number) => {
    setSearchParams((prev) => {
      prev.set('page', newPage.toString());
      return prev;
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Handle agent selection
  const handleAgentSelect = (agent: MarketplaceAgent) => {
    navigate(`/marketplace/agents/${agent.slug}`);
  };

  // Handle deploy
  const handleDeploy = (agent: MarketplaceAgent) => {
    navigate(`/marketplace/agents/${agent.slug}?deploy=true`);
  };

  // Show featured/trending only on first page with no filters
  const showFeaturedSections =
    page === 1 &&
    !filters.category &&
    !filters.search &&
    filters.tags.length === 0;

  return (
    <div className="flex min-h-screen">
      {/* Desktop Sidebar */}
      <div className="hidden w-72 flex-shrink-0 border-r lg:block">
        <CategoryNav className="sticky top-0 h-screen" />
      </div>

      {/* Mobile Filter Overlay */}
      {isMobileFilterOpen && (
        <>
          <div
            className="fixed inset-0 z-40 bg-black/50 lg:hidden"
            onClick={() => setIsMobileFilterOpen(false)}
          />
          <CategoryNav
            isMobile
            onClose={() => setIsMobileFilterOpen(false)}
          />
        </>
      )}

      {/* Main Content */}
      <div className="flex-1">
        {/* Header */}
        <div className="sticky top-0 z-30 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="container mx-auto px-4 py-4">
            {/* Title Row */}
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold">Agent Marketplace</h1>
                <p className="text-sm text-muted-foreground">
                  Discover and deploy AI agents for sustainability and compliance
                </p>
              </div>
            </div>

            {/* Search and Controls Row */}
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center">
              {/* Search */}
              <SearchAutocomplete
                value={filters.search}
                onChange={setSearchQuery}
                onSelect={handleAgentSelect}
                className="flex-1 sm:max-w-md"
              />

              {/* Controls */}
              <div className="flex items-center gap-2">
                {/* Mobile Filter Toggle */}
                <Button
                  variant="outline"
                  size="sm"
                  className="lg:hidden"
                  onClick={() => setIsMobileFilterOpen(true)}
                  leftIcon={<SlidersHorizontal className="h-4 w-4" />}
                >
                  Filters
                </Button>

                {/* Sort Dropdown */}
                <Select value={filters.sortBy} onValueChange={(v) => setSortBy(v as SortOption)}>
                  <SelectTrigger className="w-40">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {sortOptions.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                {/* View Mode Toggle */}
                <div className="flex rounded-lg border p-1">
                  <Button
                    variant={viewMode === 'grid' ? 'primary' : 'ghost'}
                    size="icon-sm"
                    onClick={() => setViewMode('grid')}
                    aria-label="Grid view"
                  >
                    <Grid3X3 className="h-4 w-4" />
                  </Button>
                  <Button
                    variant={viewMode === 'list' ? 'primary' : 'ghost'}
                    size="icon-sm"
                    onClick={() => setViewMode('list')}
                    aria-label="List view"
                  >
                    <List className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>

            {/* Active Filters Tags */}
            {(filters.category || filters.tags.length > 0 || filters.search) && (
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <span className="text-sm text-muted-foreground">Filtered by:</span>
                {filters.category && (
                  <Badge variant="secondary">
                    {filters.category}
                  </Badge>
                )}
                {filters.tags.map((tag) => (
                  <Badge key={tag} variant="secondary">
                    {tag}
                  </Badge>
                ))}
                {filters.search && (
                  <Badge variant="secondary">
                    "{filters.search}"
                  </Badge>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Content */}
        <div className="container mx-auto px-4 py-6">
          {/* Featured & Trending Sections */}
          {showFeaturedSections && (
            <>
              <TrendingAgentsSection />
              <FeaturedAgentsSection />
            </>
          )}

          {/* Results Info */}
          {pagination && (
            <div className="mb-4 flex items-center justify-between">
              <PaginationInfo
                currentPage={page}
                pageSize={perPage}
                totalItems={pagination.totalItems}
              />
              {!showFeaturedSections && (
                <p className="text-sm text-muted-foreground">
                  {filters.category
                    ? `Showing ${filters.category} agents`
                    : 'All agents'}
                </p>
              )}
            </div>
          )}

          {/* Loading State */}
          {isLoading && (
            <div
              className={cn(
                'gap-4',
                viewMode === 'grid'
                  ? 'grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3'
                  : 'flex flex-col'
              )}
            >
              {Array.from({ length: perPage }).map((_, i) => (
                <AgentCardSkeleton key={i} viewMode={viewMode} />
              ))}
            </div>
          )}

          {/* Error State */}
          {isError && (
            <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-8 text-center">
              <p className="text-lg font-medium text-destructive">
                Failed to load agents
              </p>
              <p className="text-sm text-muted-foreground">
                {error instanceof Error ? error.message : 'An error occurred'}
              </p>
              <Button
                variant="outline"
                className="mt-4"
                onClick={() => window.location.reload()}
              >
                Try Again
              </Button>
            </div>
          )}

          {/* Empty State */}
          {!isLoading && !isError && agents.length === 0 && (
            <div className="rounded-lg border bg-muted/30 p-12 text-center">
              <Search className="mx-auto mb-4 h-12 w-12 text-muted-foreground" />
              <h3 className="text-lg font-medium">No agents found</h3>
              <p className="text-sm text-muted-foreground">
                Try adjusting your search or filters to find what you're looking for.
              </p>
              <Button
                variant="outline"
                className="mt-4"
                onClick={() => {
                  useMarketplaceStore.getState().resetFilters();
                  setSearchParams({});
                }}
              >
                Clear Filters
              </Button>
            </div>
          )}

          {/* Agent Grid/List */}
          {!isLoading && !isError && agents.length > 0 && (
            <div
              className={cn(
                'gap-4',
                viewMode === 'grid'
                  ? 'grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3'
                  : 'flex flex-col'
              )}
            >
              {agents.map((agent) => (
                <AgentCard
                  key={agent.id}
                  agent={agent}
                  viewMode={viewMode}
                  onDeploy={handleDeploy}
                />
              ))}
            </div>
          )}

          {/* Pagination */}
          {pagination && totalPages > 1 && (
            <div className="mt-8 flex justify-center">
              <Pagination
                currentPage={page}
                totalPages={totalPages}
                onPageChange={handlePageChange}
              />
            </div>
          )}
        </div>
      </div>

      {/* Compare Bar */}
      {compareList.length > 0 && <CompareBar />}
    </div>
  );
}

export default AgentCatalog;
