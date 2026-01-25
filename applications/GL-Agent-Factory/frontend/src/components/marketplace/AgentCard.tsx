/**
 * AgentCard Component
 *
 * Displays an agent card in the marketplace with icon, stats, rating,
 * and action buttons. Supports both grid and list view modes.
 */

import * as React from 'react';
import { Link } from 'react-router-dom';
import {
  Star,
  Clock,
  Target,
  Zap,
  Heart,
  GitCompare,
  ChevronRight,
  CheckCircle2,
  Sparkles,
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { Card, CardContent, CardFooter } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { useMarketplaceStore } from '@/stores/marketplaceStore';
import type { MarketplaceAgent, AgentCategory } from '@/api/types/marketplace';

// ============================================================================
// Types
// ============================================================================

export interface AgentCardProps {
  agent: MarketplaceAgent;
  viewMode?: 'grid' | 'list';
  onDeploy?: (agent: MarketplaceAgent) => void;
  onCompare?: (agent: MarketplaceAgent) => void;
  className?: string;
}

// ============================================================================
// Category Colors & Icons
// ============================================================================

const categoryConfig: Record<
  AgentCategory,
  { color: string; bgColor: string; borderColor: string; icon: string }
> = {
  emissions: {
    color: 'text-emerald-700',
    bgColor: 'bg-emerald-50',
    borderColor: 'border-emerald-200',
    icon: 'CO2',
  },
  compliance: {
    color: 'text-blue-700',
    bgColor: 'bg-blue-50',
    borderColor: 'border-blue-200',
    icon: 'Shield',
  },
  reporting: {
    color: 'text-purple-700',
    bgColor: 'bg-purple-50',
    borderColor: 'border-purple-200',
    icon: 'FileText',
  },
  industry: {
    color: 'text-amber-700',
    bgColor: 'bg-amber-50',
    borderColor: 'border-amber-200',
    icon: 'Factory',
  },
  analytics: {
    color: 'text-cyan-700',
    bgColor: 'bg-cyan-50',
    borderColor: 'border-cyan-200',
    icon: 'BarChart',
  },
  integration: {
    color: 'text-rose-700',
    bgColor: 'bg-rose-50',
    borderColor: 'border-rose-200',
    icon: 'Link',
  },
};

// ============================================================================
// Star Rating Component
// ============================================================================

function StarRating({ rating, showValue = true }: { rating: number; showValue?: boolean }) {
  const fullStars = Math.floor(rating);
  const hasHalfStar = rating % 1 >= 0.5;

  return (
    <div className="flex items-center gap-1">
      <div className="flex">
        {[1, 2, 3, 4, 5].map((star) => (
          <Star
            key={star}
            className={cn(
              'h-4 w-4',
              star <= fullStars
                ? 'fill-amber-400 text-amber-400'
                : star === fullStars + 1 && hasHalfStar
                ? 'fill-amber-400/50 text-amber-400'
                : 'fill-muted text-muted'
            )}
          />
        ))}
      </div>
      {showValue && (
        <span className="text-sm font-medium text-muted-foreground">
          {rating.toFixed(1)}
        </span>
      )}
    </div>
  );
}

// ============================================================================
// Agent Card - Grid View
// ============================================================================

function AgentCardGrid({
  agent,
  onDeploy,
  className,
}: Omit<AgentCardProps, 'viewMode'>) {
  const {
    addToCompare,
    removeFromCompare,
    isInCompareList,
    addToFavorites,
    removeFromFavorites,
    isFavorite,
  } = useMarketplaceStore();

  const inCompare = isInCompareList(agent.id);
  const favorited = isFavorite(agent.id);
  const config = categoryConfig[agent.category];

  const handleFavoriteClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (favorited) {
      removeFromFavorites(agent.id);
    } else {
      addToFavorites(agent.id);
    }
  };

  const handleCompareClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (inCompare) {
      removeFromCompare(agent.id);
    } else {
      addToCompare(agent.id);
    }
  };

  const handleDeployClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onDeploy?.(agent);
  };

  return (
    <Card
      variant="interactive"
      padding="none"
      className={cn('group relative overflow-hidden', className)}
    >
      {/* Badges */}
      <div className="absolute right-3 top-3 z-10 flex gap-1.5">
        {agent.isNew && (
          <Badge variant="info" size="sm" className="gap-1">
            <Sparkles className="h-3 w-3" />
            New
          </Badge>
        )}
        {agent.isVerified && (
          <Badge variant="success" size="sm" className="gap-1">
            <CheckCircle2 className="h-3 w-3" />
            Verified
          </Badge>
        )}
      </div>

      {/* Favorite Button */}
      <button
        onClick={handleFavoriteClick}
        className="absolute left-3 top-3 z-10 rounded-full bg-white/80 p-1.5 opacity-0 shadow-sm backdrop-blur-sm transition-all hover:bg-white group-hover:opacity-100"
        aria-label={favorited ? 'Remove from favorites' : 'Add to favorites'}
      >
        <Heart
          className={cn(
            'h-4 w-4 transition-colors',
            favorited ? 'fill-red-500 text-red-500' : 'text-muted-foreground'
          )}
        />
      </button>

      <Link to={`/marketplace/agents/${agent.slug}`} className="block">
        <CardContent className="p-6">
          {/* Header */}
          <div className="mb-4 flex items-start gap-4">
            {/* Icon */}
            <div
              className={cn(
                'flex h-14 w-14 items-center justify-center rounded-xl text-2xl',
                config.bgColor
              )}
              style={{ color: agent.iconColor }}
            >
              {agent.icon}
            </div>

            {/* Title & Category */}
            <div className="min-w-0 flex-1">
              <h3 className="truncate text-lg font-semibold text-foreground group-hover:text-primary">
                {agent.name}
              </h3>
              <Badge
                variant="outline"
                size="sm"
                className={cn(config.color, config.bgColor, config.borderColor)}
              >
                {agent.category.charAt(0).toUpperCase() + agent.category.slice(1)}
              </Badge>
            </div>
          </div>

          {/* Description */}
          <p className="mb-4 line-clamp-2 text-sm text-muted-foreground">
            {agent.shortDescription}
          </p>

          {/* Stats */}
          <div className="mb-4 grid grid-cols-2 gap-3">
            <div className="flex items-center gap-2 text-sm">
              <Target className="h-4 w-4 text-emerald-500" />
              <span className="font-medium">{agent.metrics.accuracy}%</span>
              <span className="text-muted-foreground">accuracy</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <Clock className="h-4 w-4 text-blue-500" />
              <span className="font-medium">{agent.metrics.avgResponseTime}ms</span>
              <span className="text-muted-foreground">avg</span>
            </div>
          </div>

          {/* Rating & Reviews */}
          <div className="flex items-center justify-between">
            <StarRating rating={agent.metrics.averageRating} />
            <span className="text-sm text-muted-foreground">
              {agent.metrics.totalReviews.toLocaleString()} reviews
            </span>
          </div>

          {/* Feature Preview on Hover */}
          <div className="mt-4 max-h-0 overflow-hidden opacity-0 transition-all duration-300 group-hover:max-h-24 group-hover:opacity-100">
            <div className="border-t pt-3">
              <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Key Features
              </p>
              <div className="flex flex-wrap gap-1">
                {agent.features.slice(0, 3).map((feature) => (
                  <Badge key={feature.id} variant="secondary" size="sm">
                    {feature.name}
                  </Badge>
                ))}
              </div>
            </div>
          </div>
        </CardContent>

        <CardFooter className="flex gap-2 border-t bg-muted/30 p-4">
          <Button
            variant="outline"
            size="sm"
            className="flex-1"
            onClick={handleCompareClick}
            leftIcon={<GitCompare className="h-4 w-4" />}
          >
            {inCompare ? 'Remove' : 'Compare'}
          </Button>
          <Button
            variant="success"
            size="sm"
            className="flex-1"
            onClick={handleDeployClick}
            leftIcon={<Zap className="h-4 w-4" />}
          >
            Deploy
          </Button>
        </CardFooter>
      </Link>
    </Card>
  );
}

// ============================================================================
// Agent Card - List View
// ============================================================================

function AgentCardList({
  agent,
  onDeploy,
  className,
}: Omit<AgentCardProps, 'viewMode'>) {
  const {
    addToCompare,
    removeFromCompare,
    isInCompareList,
    addToFavorites,
    removeFromFavorites,
    isFavorite,
  } = useMarketplaceStore();

  const inCompare = isInCompareList(agent.id);
  const favorited = isFavorite(agent.id);
  const config = categoryConfig[agent.category];

  const handleFavoriteClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (favorited) {
      removeFromFavorites(agent.id);
    } else {
      addToFavorites(agent.id);
    }
  };

  const handleCompareClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (inCompare) {
      removeFromCompare(agent.id);
    } else {
      addToCompare(agent.id);
    }
  };

  const handleDeployClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onDeploy?.(agent);
  };

  return (
    <Card
      variant="interactive"
      padding="none"
      className={cn('group', className)}
    >
      <Link to={`/marketplace/agents/${agent.slug}`} className="block">
        <div className="flex items-center gap-6 p-4">
          {/* Icon */}
          <div
            className={cn(
              'flex h-16 w-16 flex-shrink-0 items-center justify-center rounded-xl text-2xl',
              config.bgColor
            )}
            style={{ color: agent.iconColor }}
          >
            {agent.icon}
          </div>

          {/* Main Content */}
          <div className="min-w-0 flex-1">
            <div className="mb-1 flex items-center gap-2">
              <h3 className="truncate text-lg font-semibold text-foreground group-hover:text-primary">
                {agent.name}
              </h3>
              <Badge
                variant="outline"
                size="sm"
                className={cn(config.color, config.bgColor, config.borderColor)}
              >
                {agent.category.charAt(0).toUpperCase() + agent.category.slice(1)}
              </Badge>
              {agent.isVerified && (
                <Badge variant="success" size="sm" className="gap-1">
                  <CheckCircle2 className="h-3 w-3" />
                  Verified
                </Badge>
              )}
              {agent.isNew && (
                <Badge variant="info" size="sm" className="gap-1">
                  <Sparkles className="h-3 w-3" />
                  New
                </Badge>
              )}
            </div>

            <p className="mb-2 line-clamp-1 text-sm text-muted-foreground">
              {agent.shortDescription}
            </p>

            <div className="flex items-center gap-6 text-sm">
              <StarRating rating={agent.metrics.averageRating} />
              <span className="text-muted-foreground">
                {agent.metrics.totalReviews.toLocaleString()} reviews
              </span>
              <div className="flex items-center gap-1">
                <Target className="h-4 w-4 text-emerald-500" />
                <span className="font-medium">{agent.metrics.accuracy}%</span>
                <span className="text-muted-foreground">accuracy</span>
              </div>
              <div className="flex items-center gap-1">
                <Clock className="h-4 w-4 text-blue-500" />
                <span className="font-medium">{agent.metrics.avgResponseTime}ms</span>
              </div>
            </div>
          </div>

          {/* Features */}
          <div className="hidden flex-shrink-0 xl:block">
            <div className="flex flex-wrap gap-1">
              {agent.features.slice(0, 3).map((feature) => (
                <Badge key={feature.id} variant="secondary" size="sm">
                  {feature.name}
                </Badge>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div className="flex flex-shrink-0 items-center gap-2">
            <button
              onClick={handleFavoriteClick}
              className="rounded-full p-2 hover:bg-muted"
              aria-label={favorited ? 'Remove from favorites' : 'Add to favorites'}
            >
              <Heart
                className={cn(
                  'h-5 w-5 transition-colors',
                  favorited ? 'fill-red-500 text-red-500' : 'text-muted-foreground'
                )}
              />
            </button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleCompareClick}
              leftIcon={<GitCompare className="h-4 w-4" />}
            >
              {inCompare ? 'Remove' : 'Compare'}
            </Button>
            <Button
              variant="success"
              size="sm"
              onClick={handleDeployClick}
              leftIcon={<Zap className="h-4 w-4" />}
            >
              Deploy
            </Button>
            <ChevronRight className="h-5 w-5 text-muted-foreground" />
          </div>
        </div>
      </Link>
    </Card>
  );
}

// ============================================================================
// Agent Card - Skeleton Loading
// ============================================================================

export function AgentCardSkeleton({ viewMode = 'grid' }: { viewMode?: 'grid' | 'list' }) {
  if (viewMode === 'list') {
    return (
      <Card padding="none">
        <div className="flex animate-pulse items-center gap-6 p-4">
          <div className="h-16 w-16 flex-shrink-0 rounded-xl bg-muted" />
          <div className="min-w-0 flex-1 space-y-3">
            <div className="h-5 w-48 rounded bg-muted" />
            <div className="h-4 w-96 rounded bg-muted" />
            <div className="flex gap-4">
              <div className="h-4 w-24 rounded bg-muted" />
              <div className="h-4 w-20 rounded bg-muted" />
              <div className="h-4 w-28 rounded bg-muted" />
            </div>
          </div>
          <div className="flex gap-2">
            <div className="h-9 w-24 rounded bg-muted" />
            <div className="h-9 w-20 rounded bg-muted" />
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card padding="none">
      <CardContent className="animate-pulse p-6">
        <div className="mb-4 flex items-start gap-4">
          <div className="h-14 w-14 rounded-xl bg-muted" />
          <div className="flex-1 space-y-2">
            <div className="h-5 w-32 rounded bg-muted" />
            <div className="h-5 w-20 rounded bg-muted" />
          </div>
        </div>
        <div className="mb-4 space-y-2">
          <div className="h-4 w-full rounded bg-muted" />
          <div className="h-4 w-3/4 rounded bg-muted" />
        </div>
        <div className="mb-4 grid grid-cols-2 gap-3">
          <div className="h-5 rounded bg-muted" />
          <div className="h-5 rounded bg-muted" />
        </div>
        <div className="flex justify-between">
          <div className="h-5 w-28 rounded bg-muted" />
          <div className="h-5 w-20 rounded bg-muted" />
        </div>
      </CardContent>
      <CardFooter className="flex gap-2 border-t bg-muted/30 p-4">
        <div className="h-9 flex-1 rounded bg-muted" />
        <div className="h-9 flex-1 rounded bg-muted" />
      </CardFooter>
    </Card>
  );
}

// ============================================================================
// Main Export
// ============================================================================

export function AgentCard({ viewMode = 'grid', ...props }: AgentCardProps) {
  if (viewMode === 'list') {
    return <AgentCardList {...props} />;
  }
  return <AgentCardGrid {...props} />;
}

export default AgentCard;
