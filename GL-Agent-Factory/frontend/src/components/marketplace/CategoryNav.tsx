/**
 * CategoryNav Component
 *
 * Vertical navigation sidebar for marketplace categories.
 * Shows agent count per category and supports collapsible subcategories.
 */

import * as React from 'react';
import {
  Cloud,
  Shield,
  FileText,
  Factory,
  BarChart3,
  Link2,
  ChevronDown,
  ChevronRight,
  Layers,
  X,
  Filter,
  Tag,
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Skeleton } from '@/components/ui/Skeleton';
import { useMarketplaceStore } from '@/stores/marketplaceStore';
import { useCategories, useRegulatoryFrameworks, useTags } from '@/api/hooks/marketplace';
import type { AgentCategory, AgentPricingTier } from '@/api/types/marketplace';

// ============================================================================
// Types
// ============================================================================

export interface CategoryNavProps {
  className?: string;
  onClose?: () => void;
  isMobile?: boolean;
}

interface CategoryItem {
  id: AgentCategory;
  name: string;
  icon: React.ElementType;
  count: number;
  subcategories?: { id: string; name: string; count: number }[];
}

// ============================================================================
// Category Config
// ============================================================================

const categoryIcons: Record<AgentCategory, React.ElementType> = {
  emissions: Cloud,
  compliance: Shield,
  reporting: FileText,
  industry: Factory,
  analytics: BarChart3,
  integration: Link2,
};

const categoryLabels: Record<AgentCategory, string> = {
  emissions: 'Emissions Tracking',
  compliance: 'Compliance',
  reporting: 'Reporting',
  industry: 'Industry Specific',
  analytics: 'Analytics',
  integration: 'Integration',
};

const pricingTiers: { id: AgentPricingTier; label: string }[] = [
  { id: 'free', label: 'Free' },
  { id: 'starter', label: 'Starter' },
  { id: 'professional', label: 'Professional' },
  { id: 'enterprise', label: 'Enterprise' },
];

// ============================================================================
// Category Nav Item
// ============================================================================

interface CategoryNavItemProps {
  category: CategoryItem;
  isActive: boolean;
  onClick: () => void;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
}

function CategoryNavItem({
  category,
  isActive,
  onClick,
  isExpanded,
  onToggleExpand,
}: CategoryNavItemProps) {
  const Icon = category.icon;
  const hasSubcategories = category.subcategories && category.subcategories.length > 0;

  return (
    <div>
      <button
        onClick={onClick}
        className={cn(
          'group flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm font-medium transition-colors',
          isActive
            ? 'bg-primary/10 text-primary'
            : 'text-muted-foreground hover:bg-muted hover:text-foreground'
        )}
      >
        <Icon
          className={cn(
            'h-5 w-5 flex-shrink-0',
            isActive ? 'text-primary' : 'text-muted-foreground group-hover:text-foreground'
          )}
        />
        <span className="flex-1 truncate">{category.name}</span>
        <Badge
          variant={isActive ? 'default' : 'secondary'}
          size="sm"
          className="ml-auto"
        >
          {category.count}
        </Badge>
        {hasSubcategories && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onToggleExpand?.();
            }}
            className="ml-1 rounded p-0.5 hover:bg-muted"
          >
            {isExpanded ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </button>
        )}
      </button>

      {/* Subcategories */}
      {hasSubcategories && isExpanded && (
        <div className="ml-8 mt-1 space-y-1 border-l pl-3">
          {category.subcategories!.map((sub) => (
            <button
              key={sub.id}
              className="flex w-full items-center justify-between rounded-md px-2 py-1.5 text-sm text-muted-foreground hover:bg-muted hover:text-foreground"
            >
              <span>{sub.name}</span>
              <span className="text-xs">{sub.count}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Filter Section
// ============================================================================

interface FilterSectionProps {
  title: string;
  icon: React.ElementType;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

function FilterSection({ title, icon: Icon, children, defaultOpen = true }: FilterSectionProps) {
  const [isOpen, setIsOpen] = React.useState(defaultOpen);

  return (
    <div className="border-t py-4">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between text-sm font-semibold text-foreground"
      >
        <div className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-muted-foreground" />
          <span>{title}</span>
        </div>
        {isOpen ? (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-4 w-4 text-muted-foreground" />
        )}
      </button>
      {isOpen && <div className="mt-3">{children}</div>}
    </div>
  );
}

// ============================================================================
// CategoryNav Component
// ============================================================================

export function CategoryNav({ className, onClose, isMobile = false }: CategoryNavProps) {
  const {
    filters,
    setCategory,
    setPricingTier,
    setVerifiedOnly,
    setRegulatoryFramework,
    addTag,
    removeTag,
    resetFilters,
  } = useMarketplaceStore();

  const [expandedCategories, setExpandedCategories] = React.useState<Set<string>>(new Set());

  // Fetch data
  const { data: categoriesData, isLoading: isLoadingCategories } = useCategories();
  const { data: frameworksData, isLoading: isLoadingFrameworks } = useRegulatoryFrameworks();
  const { data: tagsData, isLoading: isLoadingTags } = useTags();

  // Transform categories data
  const categories: CategoryItem[] = React.useMemo(() => {
    if (!categoriesData) return [];
    return categoriesData.map((cat) => ({
      id: cat.category as AgentCategory,
      name: categoryLabels[cat.category as AgentCategory] || cat.displayName,
      icon: categoryIcons[cat.category as AgentCategory] || Layers,
      count: cat.agentCount,
    }));
  }, [categoriesData]);

  const toggleCategoryExpand = (categoryId: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(categoryId)) {
        next.delete(categoryId);
      } else {
        next.add(categoryId);
      }
      return next;
    });
  };

  const handleCategoryClick = (categoryId: AgentCategory) => {
    if (filters.category === categoryId) {
      setCategory(undefined);
    } else {
      setCategory(categoryId);
    }
    if (isMobile) {
      onClose?.();
    }
  };

  const hasActiveFilters =
    filters.category ||
    filters.pricingTier ||
    filters.isVerified ||
    filters.regulatoryFramework ||
    filters.tags.length > 0;

  return (
    <aside
      className={cn(
        'flex h-full flex-col overflow-hidden bg-card',
        isMobile && 'fixed inset-y-0 left-0 z-50 w-80 shadow-xl',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b p-4">
        <div className="flex items-center gap-2">
          <Filter className="h-5 w-5 text-muted-foreground" />
          <h2 className="font-semibold">Filters</h2>
        </div>
        <div className="flex items-center gap-2">
          {hasActiveFilters && (
            <Button variant="ghost" size="sm" onClick={resetFilters}>
              Clear all
            </Button>
          )}
          {isMobile && onClose && (
            <Button variant="ghost" size="icon-sm" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {/* Categories */}
        <div className="mb-2">
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Categories
          </h3>

          {/* All Agents Option */}
          <button
            onClick={() => {
              setCategory(undefined);
              if (isMobile) onClose?.();
            }}
            className={cn(
              'group mb-1 flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm font-medium transition-colors',
              !filters.category
                ? 'bg-primary/10 text-primary'
                : 'text-muted-foreground hover:bg-muted hover:text-foreground'
            )}
          >
            <Layers
              className={cn(
                'h-5 w-5 flex-shrink-0',
                !filters.category
                  ? 'text-primary'
                  : 'text-muted-foreground group-hover:text-foreground'
              )}
            />
            <span className="flex-1">All Agents</span>
          </button>

          {/* Category List */}
          {isLoadingCategories ? (
            <div className="space-y-2">
              {[1, 2, 3, 4, 5].map((i) => (
                <Skeleton key={i} className="h-10 w-full rounded-lg" />
              ))}
            </div>
          ) : (
            <div className="space-y-1">
              {categories.map((category) => (
                <CategoryNavItem
                  key={category.id}
                  category={category}
                  isActive={filters.category === category.id}
                  onClick={() => handleCategoryClick(category.id)}
                  isExpanded={expandedCategories.has(category.id)}
                  onToggleExpand={() => toggleCategoryExpand(category.id)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Pricing Tier Filter */}
        <FilterSection title="Pricing" icon={Tag}>
          <div className="space-y-2">
            {pricingTiers.map((tier) => (
              <label
                key={tier.id}
                className="flex cursor-pointer items-center gap-2 text-sm"
              >
                <input
                  type="radio"
                  name="pricing"
                  checked={filters.pricingTier === tier.id}
                  onChange={() =>
                    setPricingTier(filters.pricingTier === tier.id ? undefined : tier.id)
                  }
                  className="h-4 w-4 border-gray-300 text-primary focus:ring-primary"
                />
                <span
                  className={cn(
                    filters.pricingTier === tier.id
                      ? 'text-foreground'
                      : 'text-muted-foreground'
                  )}
                >
                  {tier.label}
                </span>
              </label>
            ))}
          </div>
        </FilterSection>

        {/* Regulatory Frameworks */}
        <FilterSection title="Regulatory Framework" icon={Shield}>
          {isLoadingFrameworks ? (
            <div className="space-y-2">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-6 w-full rounded" />
              ))}
            </div>
          ) : (
            <div className="space-y-2">
              {frameworksData?.map((framework) => (
                <label
                  key={framework.framework}
                  className="flex cursor-pointer items-center justify-between gap-2 text-sm"
                >
                  <div className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="framework"
                      checked={filters.regulatoryFramework === framework.framework}
                      onChange={() =>
                        setRegulatoryFramework(
                          filters.regulatoryFramework === framework.framework
                            ? undefined
                            : framework.framework
                        )
                      }
                      className="h-4 w-4 border-gray-300 text-primary focus:ring-primary"
                    />
                    <span
                      className={cn(
                        filters.regulatoryFramework === framework.framework
                          ? 'text-foreground'
                          : 'text-muted-foreground'
                      )}
                    >
                      {framework.displayName}
                    </span>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {framework.agentCount}
                  </span>
                </label>
              ))}
            </div>
          )}
        </FilterSection>

        {/* Verified Only */}
        <FilterSection title="Verification" icon={Shield} defaultOpen={false}>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={filters.isVerified === true}
              onChange={() => setVerifiedOnly(filters.isVerified ? undefined : true)}
              className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
            />
            <span
              className={cn(
                filters.isVerified ? 'text-foreground' : 'text-muted-foreground'
              )}
            >
              Verified agents only
            </span>
          </label>
        </FilterSection>

        {/* Popular Tags */}
        <FilterSection title="Popular Tags" icon={Tag} defaultOpen={false}>
          {isLoadingTags ? (
            <div className="flex flex-wrap gap-2">
              {[1, 2, 3, 4, 5].map((i) => (
                <Skeleton key={i} className="h-6 w-16 rounded-full" />
              ))}
            </div>
          ) : (
            <div className="flex flex-wrap gap-2">
              {tagsData?.slice(0, 12).map((tag) => {
                const isSelected = filters.tags.includes(tag.tag);
                return (
                  <button
                    key={tag.tag}
                    onClick={() =>
                      isSelected ? removeTag(tag.tag) : addTag(tag.tag)
                    }
                    className={cn(
                      'rounded-full border px-2.5 py-0.5 text-xs font-medium transition-colors',
                      isSelected
                        ? 'border-primary bg-primary/10 text-primary'
                        : 'border-border text-muted-foreground hover:border-primary/50 hover:text-foreground'
                    )}
                  >
                    {tag.tag}
                    {isSelected && (
                      <X className="ml-1 inline-block h-3 w-3" />
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </FilterSection>
      </div>

      {/* Footer - Active Filters Summary */}
      {hasActiveFilters && (
        <div className="border-t bg-muted/30 p-4">
          <p className="mb-2 text-xs font-medium text-muted-foreground">
            Active Filters
          </p>
          <div className="flex flex-wrap gap-1.5">
            {filters.category && (
              <Badge
                variant="secondary"
                className="cursor-pointer gap-1"
                onClick={() => setCategory(undefined)}
              >
                {categoryLabels[filters.category]}
                <X className="h-3 w-3" />
              </Badge>
            )}
            {filters.pricingTier && (
              <Badge
                variant="secondary"
                className="cursor-pointer gap-1"
                onClick={() => setPricingTier(undefined)}
              >
                {filters.pricingTier}
                <X className="h-3 w-3" />
              </Badge>
            )}
            {filters.isVerified && (
              <Badge
                variant="secondary"
                className="cursor-pointer gap-1"
                onClick={() => setVerifiedOnly(undefined)}
              >
                Verified
                <X className="h-3 w-3" />
              </Badge>
            )}
            {filters.regulatoryFramework && (
              <Badge
                variant="secondary"
                className="cursor-pointer gap-1"
                onClick={() => setRegulatoryFramework(undefined)}
              >
                {filters.regulatoryFramework}
                <X className="h-3 w-3" />
              </Badge>
            )}
            {filters.tags.map((tag) => (
              <Badge
                key={tag}
                variant="secondary"
                className="cursor-pointer gap-1"
                onClick={() => removeTag(tag)}
              >
                {tag}
                <X className="h-3 w-3" />
              </Badge>
            ))}
          </div>
        </div>
      )}
    </aside>
  );
}

// ============================================================================
// CategoryNav Skeleton
// ============================================================================

export function CategoryNavSkeleton() {
  return (
    <aside className="flex h-full flex-col bg-card">
      <div className="flex items-center justify-between border-b p-4">
        <Skeleton className="h-5 w-20" />
      </div>
      <div className="flex-1 p-4">
        <Skeleton className="mb-4 h-4 w-24" />
        <div className="space-y-2">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <Skeleton key={i} className="h-10 w-full rounded-lg" />
          ))}
        </div>
        <Skeleton className="mb-4 mt-6 h-4 w-20" />
        <div className="space-y-2">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-6 w-full rounded" />
          ))}
        </div>
      </div>
    </aside>
  );
}

export default CategoryNav;
