/**
 * AgentDetails Page
 *
 * Detailed view of a marketplace agent including overview, features,
 * use cases, API documentation, pricing, reviews, and related agents.
 */

import * as React from 'react';
import { useParams, useNavigate, useSearchParams, Link } from 'react-router-dom';
import {
  ArrowLeft,
  Star,
  Clock,
  Target,
  Zap,
  Heart,
  GitCompare,
  ExternalLink,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Download,
  Play,
  BookOpen,
  Code,
  Users,
  TrendingUp,
  Shield,
  AlertCircle,
  Copy,
  Check,
  ThumbsUp,
  MessageSquare,
  Calendar,
  Sparkles,
} from 'lucide-react';
import { cn } from '@/utils/cn';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/Tabs';
import { Avatar } from '@/components/ui/Avatar';
import { Skeleton } from '@/components/ui/Skeleton';
import { Dialog } from '@/components/ui/Dialog';
import { AgentCard, AgentCardSkeleton } from '@/components/marketplace/AgentCard';
import { useMarketplaceStore } from '@/stores/marketplaceStore';
import {
  useAgent,
  useAgentVersions,
  useAgentReviews,
  useReviewSummary,
  useRelatedAgents,
  useDeployAgent,
} from '@/api/hooks/marketplace';
import type {
  MarketplaceAgent,
  AgentVersion,
  AgentReview,
  ReviewSummary,
  AgentCategory,
} from '@/api/types/marketplace';

// ============================================================================
// Category Config
// ============================================================================

const categoryColors: Record<AgentCategory, { bg: string; text: string; border: string }> = {
  emissions: { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200' },
  compliance: { bg: 'bg-blue-50', text: 'text-blue-700', border: 'border-blue-200' },
  reporting: { bg: 'bg-purple-50', text: 'text-purple-700', border: 'border-purple-200' },
  industry: { bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200' },
  analytics: { bg: 'bg-cyan-50', text: 'text-cyan-700', border: 'border-cyan-200' },
  integration: { bg: 'bg-rose-50', text: 'text-rose-700', border: 'border-rose-200' },
};

// ============================================================================
// Star Rating Component
// ============================================================================

function StarRating({
  rating,
  showValue = true,
  size = 'md',
}: {
  rating: number;
  showValue?: boolean;
  size?: 'sm' | 'md' | 'lg';
}) {
  const fullStars = Math.floor(rating);
  const hasHalfStar = rating % 1 >= 0.5;
  const sizeClasses = {
    sm: 'h-3 w-3',
    md: 'h-4 w-4',
    lg: 'h-5 w-5',
  };

  return (
    <div className="flex items-center gap-1">
      <div className="flex">
        {[1, 2, 3, 4, 5].map((star) => (
          <Star
            key={star}
            className={cn(
              sizeClasses[size],
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
        <span className="font-medium text-muted-foreground">{rating.toFixed(1)}</span>
      )}
    </div>
  );
}

// ============================================================================
// Hero Section
// ============================================================================

interface HeroSectionProps {
  agent: MarketplaceAgent;
  onDeploy: () => void;
  onTryDemo: () => void;
}

function HeroSection({ agent, onDeploy, onTryDemo }: HeroSectionProps) {
  const {
    addToCompare,
    removeFromCompare,
    isInCompareList,
    addToFavorites,
    removeFromFavorites,
    isFavorite,
    addToRecentlyViewed,
  } = useMarketplaceStore();

  const inCompare = isInCompareList(agent.id);
  const favorited = isFavorite(agent.id);
  const colors = categoryColors[agent.category];

  // Track view
  React.useEffect(() => {
    addToRecentlyViewed(agent.id);
  }, [agent.id, addToRecentlyViewed]);

  return (
    <div className="border-b bg-gradient-to-b from-muted/30 to-background">
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
          {/* Left - Info */}
          <div className="flex-1">
            <div className="mb-4 flex items-start gap-4">
              {/* Icon */}
              <div
                className={cn(
                  'flex h-20 w-20 flex-shrink-0 items-center justify-center rounded-2xl text-4xl',
                  colors.bg
                )}
                style={{ color: agent.iconColor }}
              >
                {agent.icon}
              </div>

              {/* Title & Meta */}
              <div className="min-w-0 flex-1">
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <h1 className="text-2xl font-bold lg:text-3xl">{agent.name}</h1>
                  {agent.isVerified && (
                    <Badge variant="success" className="gap-1">
                      <CheckCircle2 className="h-3 w-3" />
                      Verified
                    </Badge>
                  )}
                  {agent.isNew && (
                    <Badge variant="info" className="gap-1">
                      <Sparkles className="h-3 w-3" />
                      New
                    </Badge>
                  )}
                </div>

                <div className="mb-3 flex flex-wrap items-center gap-3 text-sm">
                  <Badge
                    variant="outline"
                    className={cn(colors.text, colors.bg, colors.border)}
                  >
                    {agent.category.charAt(0).toUpperCase() + agent.category.slice(1)}
                  </Badge>
                  <span className="text-muted-foreground">v{agent.version}</span>
                  <span className="text-muted-foreground">by {agent.author.name}</span>
                </div>

                <p className="mb-4 text-muted-foreground">{agent.shortDescription}</p>

                {/* Stats Row */}
                <div className="flex flex-wrap items-center gap-4 text-sm">
                  <div className="flex items-center gap-1">
                    <StarRating rating={agent.metrics.averageRating} />
                    <span className="text-muted-foreground">
                      ({agent.metrics.totalReviews.toLocaleString()} reviews)
                    </span>
                  </div>
                  <div className="flex items-center gap-1 text-muted-foreground">
                    <Download className="h-4 w-4" />
                    <span>{agent.metrics.totalDeployments.toLocaleString()} deployments</span>
                  </div>
                  <div className="flex items-center gap-1 text-emerald-600">
                    <Target className="h-4 w-4" />
                    <span>{agent.metrics.accuracy}% accuracy</span>
                  </div>
                  <div className="flex items-center gap-1 text-blue-600">
                    <Clock className="h-4 w-4" />
                    <span>{agent.metrics.avgResponseTime}ms avg</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Tags */}
            {agent.tags.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {agent.tags.map((tag) => (
                  <Badge key={tag} variant="secondary" size="sm">
                    {tag}
                  </Badge>
                ))}
              </div>
            )}
          </div>

          {/* Right - Actions */}
          <div className="flex flex-shrink-0 flex-col gap-3 lg:items-end">
            {/* Pricing */}
            <div className="text-right">
              {agent.pricing.tier === 'free' ? (
                <p className="text-2xl font-bold text-emerald-600">Free</p>
              ) : (
                <>
                  <p className="text-2xl font-bold">
                    ${agent.pricing.basePrice}
                    <span className="text-sm font-normal text-muted-foreground">
                      /{agent.pricing.billingPeriod}
                    </span>
                  </p>
                  {agent.pricing.trialDays && (
                    <p className="text-sm text-muted-foreground">
                      {agent.pricing.trialDays}-day free trial
                    </p>
                  )}
                </>
              )}
            </div>

            {/* Action Buttons */}
            <div className="flex gap-2">
              <Button
                variant="success"
                size="lg"
                onClick={onDeploy}
                leftIcon={<Zap className="h-5 w-5" />}
              >
                Deploy Agent
              </Button>
              <Button
                variant="outline"
                size="lg"
                onClick={onTryDemo}
                leftIcon={<Play className="h-5 w-5" />}
              >
                Try Demo
              </Button>
            </div>

            {/* Secondary Actions */}
            <div className="flex gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() =>
                  favorited ? removeFromFavorites(agent.id) : addToFavorites(agent.id)
                }
                leftIcon={
                  <Heart
                    className={cn(
                      'h-4 w-4',
                      favorited && 'fill-red-500 text-red-500'
                    )}
                  />
                }
              >
                {favorited ? 'Saved' : 'Save'}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() =>
                  inCompare ? removeFromCompare(agent.id) : addToCompare(agent.id)
                }
                leftIcon={<GitCompare className="h-4 w-4" />}
              >
                {inCompare ? 'Remove from Compare' : 'Compare'}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Overview Tab
// ============================================================================

function OverviewTab({ agent }: { agent: MarketplaceAgent }) {
  return (
    <div className="space-y-8">
      {/* Description */}
      <section>
        <h2 className="mb-4 text-lg font-semibold">About</h2>
        <div className="prose max-w-none text-muted-foreground">
          <p>{agent.description}</p>
        </div>
      </section>

      {/* Key Metrics */}
      <section>
        <h2 className="mb-4 text-lg font-semibold">Performance Metrics</h2>
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <Card>
            <CardContent className="p-4 text-center">
              <Target className="mx-auto mb-2 h-8 w-8 text-emerald-500" />
              <p className="text-2xl font-bold">{agent.metrics.accuracy}%</p>
              <p className="text-sm text-muted-foreground">Accuracy</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 text-center">
              <Clock className="mx-auto mb-2 h-8 w-8 text-blue-500" />
              <p className="text-2xl font-bold">{agent.metrics.avgResponseTime}ms</p>
              <p className="text-sm text-muted-foreground">Avg Response</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 text-center">
              <TrendingUp className="mx-auto mb-2 h-8 w-8 text-purple-500" />
              <p className="text-2xl font-bold">{agent.metrics.uptime}%</p>
              <p className="text-sm text-muted-foreground">Uptime</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 text-center">
              <Users className="mx-auto mb-2 h-8 w-8 text-amber-500" />
              <p className="text-2xl font-bold">
                {agent.metrics.satisfactionScore}%
              </p>
              <p className="text-sm text-muted-foreground">Satisfaction</p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Regulatory Frameworks */}
      {agent.regulatoryFrameworks.length > 0 && (
        <section>
          <h2 className="mb-4 text-lg font-semibold">Regulatory Compliance</h2>
          <div className="flex flex-wrap gap-2">
            {agent.regulatoryFrameworks.map((framework) => (
              <Badge key={framework} variant="outline" className="gap-1">
                <Shield className="h-3 w-3" />
                {framework}
              </Badge>
            ))}
          </div>
        </section>
      )}

      {/* Integrations */}
      {agent.integrations.length > 0 && (
        <section>
          <h2 className="mb-4 text-lg font-semibold">Integrations</h2>
          <div className="flex flex-wrap gap-2">
            {agent.integrations.map((integration) => (
              <Badge key={integration} variant="secondary">
                {integration}
              </Badge>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

// ============================================================================
// Features Tab
// ============================================================================

function FeaturesTab({ agent }: { agent: MarketplaceAgent }) {
  const groupedFeatures = React.useMemo(() => {
    const groups: Record<string, typeof agent.features> = {};
    agent.features.forEach((feature) => {
      if (!groups[feature.category]) {
        groups[feature.category] = [];
      }
      groups[feature.category].push(feature);
    });
    return groups;
  }, [agent.features]);

  return (
    <div className="space-y-8">
      {Object.entries(groupedFeatures).map(([category, features]) => (
        <section key={category}>
          <h2 className="mb-4 text-lg font-semibold capitalize">{category}</h2>
          <div className="grid gap-4 md:grid-cols-2">
            {features.map((feature) => (
              <Card key={feature.id}>
                <CardContent className="flex gap-4 p-4">
                  <CheckCircle2 className="h-5 w-5 flex-shrink-0 text-emerald-500" />
                  <div>
                    <h3 className="font-medium">{feature.name}</h3>
                    <p className="text-sm text-muted-foreground">
                      {feature.description}
                    </p>
                    {!feature.isIncludedInFree && (
                      <Badge variant="secondary" size="sm" className="mt-2">
                        {feature.requiredTier}+ plan
                      </Badge>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

// ============================================================================
// Use Cases Tab
// ============================================================================

function UseCasesTab({ agent }: { agent: MarketplaceAgent }) {
  return (
    <div className="space-y-6">
      {agent.useCases.map((useCase) => (
        <Card key={useCase.id}>
          <CardContent className="p-6">
            <div className="mb-3 flex items-start justify-between">
              <div>
                <h3 className="text-lg font-semibold">{useCase.title}</h3>
                <Badge variant="outline" size="sm" className="mt-1">
                  {useCase.industry}
                </Badge>
              </div>
              {useCase.estimatedROI && (
                <Badge variant="success" className="gap-1">
                  <TrendingUp className="h-3 w-3" />
                  {useCase.estimatedROI} ROI
                </Badge>
              )}
            </div>
            <p className="mb-4 text-muted-foreground">{useCase.description}</p>
            <div>
              <p className="mb-2 text-sm font-medium">Key Benefits:</p>
              <ul className="grid gap-2 md:grid-cols-2">
                {useCase.benefits.map((benefit, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    <CheckCircle2 className="h-4 w-4 flex-shrink-0 text-emerald-500" />
                    <span>{benefit}</span>
                  </li>
                ))}
              </ul>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// ============================================================================
// API Tab
// ============================================================================

function APITab({ agent }: { agent: MarketplaceAgent }) {
  const [copied, setCopied] = React.useState(false);

  const exampleCode = `import { GreenLangClient } from '@greenlang/sdk';

const client = new GreenLangClient({
  apiKey: process.env.GREENLANG_API_KEY,
});

// Use the ${agent.name} agent
const result = await client.agents.${agent.slug.replace(/-/g, '_')}.analyze({
  data: yourData,
  options: {
    // Agent-specific options
  }
});

console.log(result);`;

  const handleCopy = () => {
    navigator.clipboard.writeText(exampleCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-8">
      {/* Quick Start */}
      <section>
        <h2 className="mb-4 text-lg font-semibold">Quick Start</h2>
        <div className="relative rounded-lg bg-muted p-4">
          <Button
            variant="ghost"
            size="icon-sm"
            className="absolute right-2 top-2"
            onClick={handleCopy}
          >
            {copied ? (
              <Check className="h-4 w-4 text-emerald-500" />
            ) : (
              <Copy className="h-4 w-4" />
            )}
          </Button>
          <pre className="overflow-x-auto text-sm">
            <code>{exampleCode}</code>
          </pre>
        </div>
      </section>

      {/* Documentation Links */}
      <section>
        <h2 className="mb-4 text-lg font-semibold">Documentation</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <a
            href={agent.documentation.quickStartUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 rounded-lg border p-4 hover:bg-muted"
          >
            <BookOpen className="h-5 w-5 text-primary" />
            <div>
              <p className="font-medium">Quick Start Guide</p>
              <p className="text-sm text-muted-foreground">
                Get up and running in minutes
              </p>
            </div>
            <ExternalLink className="ml-auto h-4 w-4 text-muted-foreground" />
          </a>
          <a
            href={agent.documentation.apiReferenceUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 rounded-lg border p-4 hover:bg-muted"
          >
            <Code className="h-5 w-5 text-primary" />
            <div>
              <p className="font-medium">API Reference</p>
              <p className="text-sm text-muted-foreground">
                Complete API documentation
              </p>
            </div>
            <ExternalLink className="ml-auto h-4 w-4 text-muted-foreground" />
          </a>
          <a
            href={agent.documentation.tutorialsUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 rounded-lg border p-4 hover:bg-muted"
          >
            <Play className="h-5 w-5 text-primary" />
            <div>
              <p className="font-medium">Tutorials</p>
              <p className="text-sm text-muted-foreground">
                Step-by-step guides
              </p>
            </div>
            <ExternalLink className="ml-auto h-4 w-4 text-muted-foreground" />
          </a>
          <a
            href={agent.documentation.supportUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 rounded-lg border p-4 hover:bg-muted"
          >
            <MessageSquare className="h-5 w-5 text-primary" />
            <div>
              <p className="font-medium">Support</p>
              <p className="text-sm text-muted-foreground">
                Get help from our team
              </p>
            </div>
            <ExternalLink className="ml-auto h-4 w-4 text-muted-foreground" />
          </a>
        </div>
      </section>
    </div>
  );
}

// ============================================================================
// Pricing Section
// ============================================================================

function PricingSection({ agent }: { agent: MarketplaceAgent }) {
  const navigate = useNavigate();

  return (
    <section className="mt-8">
      <h2 className="mb-4 text-lg font-semibold">Pricing Plans</h2>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {agent.pricing.tiers.map((tier) => (
          <Card
            key={tier.name}
            className={cn(
              'relative',
              tier.recommended && 'border-primary ring-2 ring-primary/20'
            )}
          >
            {tier.recommended && (
              <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                <Badge variant="default">Recommended</Badge>
              </div>
            )}
            <CardHeader>
              <CardTitle className="text-center">
                {tier.name.charAt(0).toUpperCase() + tier.name.slice(1)}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="mb-4 text-center">
                {tier.price === 0 ? (
                  <p className="text-3xl font-bold">Free</p>
                ) : (
                  <p className="text-3xl font-bold">
                    ${tier.price}
                    <span className="text-sm font-normal text-muted-foreground">
                      /mo
                    </span>
                  </p>
                )}
              </div>
              <ul className="mb-4 space-y-2 text-sm">
                {tier.features.map((feature, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 flex-shrink-0 text-emerald-500" />
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>
              <div className="mb-4 space-y-1 text-xs text-muted-foreground">
                <p>{tier.limits.apiCalls.toLocaleString()} API calls/mo</p>
                <p>{tier.limits.dataRetentionDays} days data retention</p>
                <p>{tier.limits.supportLevel} support</p>
              </div>
              <Button
                variant={tier.recommended ? 'primary' : 'outline'}
                className="w-full"
                onClick={() =>
                  navigate(`/marketplace/agents/${agent.slug}?deploy=true&tier=${tier.name}`)
                }
              >
                Get Started
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>
    </section>
  );
}

// ============================================================================
// Version History Accordion
// ============================================================================

function VersionHistory({ agentId }: { agentId: string }) {
  const [expandedVersion, setExpandedVersion] = React.useState<string | null>(null);
  const { data: versionsData, isLoading } = useAgentVersions(agentId);

  if (isLoading) {
    return (
      <section className="mt-8">
        <h2 className="mb-4 text-lg font-semibold">Version History</h2>
        <div className="space-y-2">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-16 w-full rounded-lg" />
          ))}
        </div>
      </section>
    );
  }

  const versions = versionsData?.items || [];

  return (
    <section className="mt-8">
      <h2 className="mb-4 text-lg font-semibold">Version History</h2>
      <div className="space-y-2">
        {versions.map((version) => (
          <div key={version.id} className="rounded-lg border">
            <button
              onClick={() =>
                setExpandedVersion(expandedVersion === version.id ? null : version.id)
              }
              className="flex w-full items-center justify-between p-4 text-left"
            >
              <div className="flex items-center gap-3">
                <span className="font-mono font-medium">v{version.version}</span>
                {version.isLatest && (
                  <Badge variant="success" size="sm">
                    Latest
                  </Badge>
                )}
                {version.isBeta && (
                  <Badge variant="warning" size="sm">
                    Beta
                  </Badge>
                )}
                {version.breakingChanges && (
                  <Badge variant="destructive" size="sm">
                    Breaking
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-3 text-sm text-muted-foreground">
                <span>
                  {new Date(version.releaseDate).toLocaleDateString()}
                </span>
                {expandedVersion === version.id ? (
                  <ChevronDown className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
              </div>
            </button>
            {expandedVersion === version.id && (
              <div className="border-t px-4 py-3">
                <p className="mb-3 text-sm text-muted-foreground">
                  {version.releaseNotes}
                </p>
                <div className="space-y-2">
                  {version.changes.map((change, i) => (
                    <div key={i} className="flex items-start gap-2 text-sm">
                      <Badge
                        variant={
                          change.type === 'feature'
                            ? 'success'
                            : change.type === 'bugfix'
                            ? 'info'
                            : change.type === 'security'
                            ? 'destructive'
                            : 'secondary'
                        }
                        size="sm"
                      >
                        {change.type}
                      </Badge>
                      <span>{change.description}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

// ============================================================================
// Reviews Section
// ============================================================================

function ReviewsSection({ agentId }: { agentId: string }) {
  const { data: summary } = useReviewSummary(agentId);
  const { data: reviewsData, isLoading } = useAgentReviews(agentId, {
    perPage: 5,
    sortBy: 'helpful',
  });

  const reviews = reviewsData?.items || [];

  return (
    <section className="mt-8">
      <h2 className="mb-4 text-lg font-semibold">User Reviews</h2>

      {/* Summary */}
      {summary && (
        <Card className="mb-6">
          <CardContent className="flex flex-col gap-6 p-6 md:flex-row md:items-center">
            {/* Overall Rating */}
            <div className="text-center md:border-r md:pr-6">
              <p className="text-4xl font-bold">{summary.averageRating.toFixed(1)}</p>
              <StarRating rating={summary.averageRating} showValue={false} size="lg" />
              <p className="mt-1 text-sm text-muted-foreground">
                {summary.totalReviews.toLocaleString()} reviews
              </p>
            </div>

            {/* Rating Distribution */}
            <div className="flex-1 space-y-1">
              {[5, 4, 3, 2, 1].map((stars) => {
                const count = summary.ratingDistribution[stars as keyof typeof summary.ratingDistribution];
                const percentage = (count / summary.totalReviews) * 100 || 0;
                return (
                  <div key={stars} className="flex items-center gap-2 text-sm">
                    <span className="w-3">{stars}</span>
                    <Star className="h-3 w-3 fill-amber-400 text-amber-400" />
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full bg-amber-400"
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                    <span className="w-8 text-right text-muted-foreground">
                      {count}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Pros & Cons */}
            <div className="flex-1 space-y-3 md:border-l md:pl-6">
              {summary.topPros.length > 0 && (
                <div>
                  <p className="mb-1 text-xs font-medium text-emerald-600">
                    Top Pros
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {summary.topPros.slice(0, 3).map((pro) => (
                      <Badge key={pro} variant="success" size="sm">
                        {pro}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
              {summary.topCons.length > 0 && (
                <div>
                  <p className="mb-1 text-xs font-medium text-rose-600">
                    Top Cons
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {summary.topCons.slice(0, 3).map((con) => (
                      <Badge key={con} variant="destructive" size="sm">
                        {con}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Reviews List */}
      {isLoading ? (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-32 w-full rounded-lg" />
          ))}
        </div>
      ) : (
        <div className="space-y-4">
          {reviews.map((review) => (
            <Card key={review.id}>
              <CardContent className="p-4">
                <div className="mb-3 flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <Avatar className="h-10 w-10">
                      {review.userAvatar ? (
                        <img src={review.userAvatar} alt={review.userName} />
                      ) : (
                        <span>{review.userName.charAt(0)}</span>
                      )}
                    </Avatar>
                    <div>
                      <p className="font-medium">{review.userName}</p>
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <StarRating rating={review.rating} size="sm" />
                        {review.verifiedPurchase && (
                          <Badge variant="outline" size="sm">
                            Verified
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                  <span className="text-sm text-muted-foreground">
                    {new Date(review.createdAt).toLocaleDateString()}
                  </span>
                </div>

                <h4 className="mb-2 font-medium">{review.title}</h4>
                <p className="mb-3 text-sm text-muted-foreground">
                  {review.content}
                </p>

                {(review.pros.length > 0 || review.cons.length > 0) && (
                  <div className="mb-3 flex flex-wrap gap-4 text-sm">
                    {review.pros.length > 0 && (
                      <div>
                        <span className="font-medium text-emerald-600">Pros: </span>
                        {review.pros.join(', ')}
                      </div>
                    )}
                    {review.cons.length > 0 && (
                      <div>
                        <span className="font-medium text-rose-600">Cons: </span>
                        {review.cons.join(', ')}
                      </div>
                    )}
                  </div>
                )}

                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                  <button className="flex items-center gap-1 hover:text-foreground">
                    <ThumbsUp className="h-4 w-4" />
                    Helpful ({review.helpfulCount})
                  </button>
                </div>
              </CardContent>
            </Card>
          ))}

          {reviews.length > 0 && (
            <Button variant="outline" className="w-full">
              View All Reviews
            </Button>
          )}
        </div>
      )}
    </section>
  );
}

// ============================================================================
// Related Agents Carousel
// ============================================================================

function RelatedAgents({ agentId }: { agentId: string }) {
  const { data: related, isLoading } = useRelatedAgents(agentId, 4);

  if (isLoading) {
    return (
      <section className="mt-8">
        <h2 className="mb-4 text-lg font-semibold">Related Agents</h2>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
          {[1, 2, 3, 4].map((i) => (
            <AgentCardSkeleton key={i} viewMode="grid" />
          ))}
        </div>
      </section>
    );
  }

  if (!related || related.length === 0) return null;

  return (
    <section className="mt-8">
      <h2 className="mb-4 text-lg font-semibold">Related Agents</h2>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        {related.map((agent) => (
          <AgentCard key={agent.id} agent={agent} viewMode="grid" />
        ))}
      </div>
    </section>
  );
}

// ============================================================================
// Main AgentDetails Page
// ============================================================================

export function AgentDetails() {
  const { slug } = useParams<{ slug: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { activeTab, setActiveTab } = useMarketplaceStore();

  const { data: agent, isLoading, isError, error } = useAgent(slug || '');
  const deployMutation = useDeployAgent();

  const showDeployModal = searchParams.get('deploy') === 'true';

  const handleDeploy = () => {
    navigate(`/marketplace/agents/${slug}?deploy=true`);
  };

  const handleTryDemo = () => {
    // Open demo modal or navigate to demo page
    window.open(`/demo/${slug}`, '_blank');
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="min-h-screen">
        <div className="border-b bg-gradient-to-b from-muted/30 to-background">
          <div className="container mx-auto px-4 py-8">
            <div className="flex gap-6">
              <Skeleton className="h-20 w-20 rounded-2xl" />
              <div className="flex-1 space-y-3">
                <Skeleton className="h-8 w-64" />
                <Skeleton className="h-4 w-48" />
                <Skeleton className="h-4 w-96" />
              </div>
            </div>
          </div>
        </div>
        <div className="container mx-auto px-4 py-8">
          <Skeleton className="mb-6 h-10 w-96" />
          <div className="space-y-4">
            <Skeleton className="h-48 w-full rounded-lg" />
            <Skeleton className="h-48 w-full rounded-lg" />
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (isError || !agent) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-center">
          <AlertCircle className="mx-auto mb-4 h-12 w-12 text-destructive" />
          <h1 className="mb-2 text-xl font-bold">Agent Not Found</h1>
          <p className="mb-4 text-muted-foreground">
            {error instanceof Error
              ? error.message
              : 'The agent you are looking for does not exist.'}
          </p>
          <Button onClick={() => navigate('/marketplace')}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Marketplace
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      {/* Back Link */}
      <div className="container mx-auto px-4 py-4">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigate('/marketplace')}
          leftIcon={<ArrowLeft className="h-4 w-4" />}
        >
          Back to Marketplace
        </Button>
      </div>

      {/* Hero Section */}
      <HeroSection agent={agent} onDeploy={handleDeploy} onTryDemo={handleTryDemo} />

      {/* Content */}
      <div className="container mx-auto px-4 py-8">
        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-6">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="features">Features</TabsTrigger>
            <TabsTrigger value="usecases">Use Cases</TabsTrigger>
            <TabsTrigger value="api">API</TabsTrigger>
          </TabsList>

          <TabsContent value="overview">
            <OverviewTab agent={agent} />
          </TabsContent>

          <TabsContent value="features">
            <FeaturesTab agent={agent} />
          </TabsContent>

          <TabsContent value="usecases">
            <UseCasesTab agent={agent} />
          </TabsContent>

          <TabsContent value="api">
            <APITab agent={agent} />
          </TabsContent>
        </Tabs>

        {/* Pricing */}
        <PricingSection agent={agent} />

        {/* Version History */}
        <VersionHistory agentId={agent.id} />

        {/* Reviews */}
        <ReviewsSection agentId={agent.id} />

        {/* Related Agents */}
        <RelatedAgents agentId={agent.id} />
      </div>
    </div>
  );
}

export default AgentDetails;
