# GreenLang Hub Specification
## Agent Registry & Community Platform

### 1. Technical Architecture

#### Core Stack
```yaml
Frontend:
  Framework: Next.js 14 (App Router)
  Language: TypeScript 5.3+
  Styling: Tailwind CSS + shadcn/ui
  State: TanStack Query + Zustand
  Search: Meilisearch (self-hosted)
  Analytics: Mixpanel + Custom Analytics

Backend:
  API: GraphQL (Apollo Server)
  Database: PostgreSQL + TimescaleDB
  Cache: Redis + CDN
  Storage: S3-compatible (agent packages)
  Queue: BullMQ (background jobs)

Infrastructure:
  Container: Docker + Kubernetes
  Registry: Private NPM/PyPI registry
  CDN: CloudFront
  Monitoring: Grafana + Prometheus
```

### 2. Hub Frontend Architecture

```typescript
// Main Hub Interface
export const GreenLangHub: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState<FilterState>({
    category: [],
    tags: [],
    certification: [],
    priceRange: null,
    rating: null,
    sortBy: 'downloads',
  });

  const { data: agents, isLoading } = useInfiniteQuery({
    queryKey: ['agents', searchQuery, filters],
    queryFn: ({ pageParam = 0 }) =>
      searchAgents({
        query: searchQuery,
        filters,
        page: pageParam,
        limit: 20,
      }),
    getNextPageParam: (lastPage) => lastPage.nextCursor,
  });

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-green-600 to-emerald-600 text-white">
        <div className="container mx-auto px-4 py-16">
          <h1 className="text-5xl font-bold mb-4">GreenLang Hub</h1>
          <p className="text-xl mb-8">
            Discover, share, and deploy climate intelligence agents
          </p>

          {/* Search Bar */}
          <div className="max-w-3xl mx-auto">
            <SearchBar
              value={searchQuery}
              onChange={setSearchQuery}
              placeholder="Search 5,000+ climate agents..."
              suggestions={[
                'CSRD reporting',
                'carbon calculator',
                'scope 3 emissions',
                'CBAM compliance',
              ]}
            />
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-4 gap-8 mt-12 max-w-4xl mx-auto">
            <StatCard icon="package" label="Total Agents" value="5,247" />
            <StatCard icon="users" label="Contributors" value="1,832" />
            <StatCard icon="download" label="Downloads/Month" value="2.4M" />
            <StatCard icon="star" label="GitHub Stars" value="45.2K" />
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        <div className="flex gap-8">
          {/* Filters Sidebar */}
          <aside className="w-64 flex-shrink-0">
            <FiltersPanel
              filters={filters}
              onChange={setFilters}
              facets={agents?.pages[0]?.facets}
            />
          </aside>

          {/* Main Content */}
          <main className="flex-1">
            {/* Category Pills */}
            <div className="flex gap-2 mb-6 flex-wrap">
              {['All', 'CSRD', 'CBAM', 'Carbon', 'Energy', 'Supply Chain'].map(
                (category) => (
                  <button
                    key={category}
                    className="px-4 py-2 rounded-full border hover:bg-green-50"
                  >
                    {category}
                  </button>
                )
              )}
            </div>

            {/* Agent Grid */}
            {isLoading ? (
              <AgentGridSkeleton />
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {agents?.pages.flatMap((page) => page.items).map((agent) => (
                  <AgentCard key={agent.id} agent={agent} />
                ))}
              </div>
            )}

            {/* Load More */}
            <div className="mt-12 text-center">
              <button
                onClick={() => fetchNextPage()}
                disabled={!hasNextPage || isFetchingNextPage}
                className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700"
              >
                {isFetchingNextPage ? 'Loading...' : 'Load More Agents'}
              </button>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
};
```

### 3. Agent Card Component

```typescript
interface Agent {
  id: string;
  name: string;
  description: string;
  category: string;
  tags: string[];
  author: Author;
  version: string;
  downloads: number;
  rating: number;
  reviews: number;
  certification: 'verified' | 'enterprise' | 'community' | null;
  pricing: {
    type: 'free' | 'paid' | 'freemium';
    price?: number;
    currency?: string;
  };
  lastUpdated: Date;
}

export const AgentCard: React.FC<{ agent: Agent }> = ({ agent }) => {
  const [isHovered, setIsHovered] = useState(false);
  const [showQuickView, setShowQuickView] = useState(false);

  return (
    <>
      <motion.div
        className="bg-white rounded-lg border hover:shadow-lg transition-all cursor-pointer"
        onHoverStart={() => setIsHovered(true)}
        onHoverEnd={() => setIsHovered(false)}
        whileHover={{ y: -4 }}
      >
        <Link href={`/agents/${agent.id}`}>
          <div className="p-6">
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-gradient-to-br from-green-400 to-emerald-500 rounded-lg flex items-center justify-center">
                  <Icon name={agent.icon || 'package'} className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="font-semibold text-lg flex items-center gap-2">
                    {agent.name}
                    {agent.certification && (
                      <Badge variant={agent.certification}>
                        {agent.certification}
                      </Badge>
                    )}
                  </h3>
                  <p className="text-sm text-gray-500">by {agent.author.name}</p>
                </div>
              </div>

              {/* Quick Actions */}
              <AnimatePresence>
                {isHovered && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    className="flex gap-1"
                  >
                    <button
                      onClick={(e) => {
                        e.preventDefault();
                        setShowQuickView(true);
                      }}
                      className="p-2 hover:bg-gray-100 rounded"
                    >
                      <Eye className="w-4 h-4" />
                    </button>
                    <button
                      onClick={(e) => {
                        e.preventDefault();
                        installAgent(agent.id);
                      }}
                      className="p-2 hover:bg-gray-100 rounded"
                    >
                      <Download className="w-4 h-4" />
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Description */}
            <p className="text-gray-600 mb-4 line-clamp-2">
              {agent.description}
            </p>

            {/* Tags */}
            <div className="flex flex-wrap gap-2 mb-4">
              {agent.tags.slice(0, 3).map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-1 bg-gray-100 text-gray-700 rounded text-xs"
                >
                  {tag}
                </span>
              ))}
              {agent.tags.length > 3 && (
                <span className="px-2 py-1 text-gray-500 text-xs">
                  +{agent.tags.length - 3} more
                </span>
              )}
            </div>

            {/* Stats */}
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1">
                  <Download className="w-4 h-4 text-gray-400" />
                  {formatNumber(agent.downloads)}
                </span>
                <span className="flex items-center gap-1">
                  <Star className="w-4 h-4 text-yellow-500" />
                  {agent.rating.toFixed(1)}
                  <span className="text-gray-400">({agent.reviews})</span>
                </span>
              </div>

              {/* Pricing */}
              <div className="font-medium">
                {agent.pricing.type === 'free' ? (
                  <span className="text-green-600">Free</span>
                ) : agent.pricing.type === 'paid' ? (
                  <span>
                    {agent.pricing.currency}
                    {agent.pricing.price}/mo
                  </span>
                ) : (
                  <span className="text-blue-600">Freemium</span>
                )}
              </div>
            </div>

            {/* Version & Update */}
            <div className="mt-4 pt-4 border-t flex items-center justify-between text-xs text-gray-500">
              <span>v{agent.version}</span>
              <span>Updated {formatRelativeTime(agent.lastUpdated)}</span>
            </div>
          </div>
        </Link>
      </motion.div>

      {/* Quick View Modal */}
      {showQuickView && (
        <AgentQuickView
          agent={agent}
          onClose={() => setShowQuickView(false)}
        />
      )}
    </>
  );
};
```

### 4. Agent Detail Page

```typescript
export const AgentDetailPage: React.FC<{ agentId: string }> = ({ agentId }) => {
  const { data: agent } = useQuery({
    queryKey: ['agent', agentId],
    queryFn: () => getAgentDetails(agentId),
  });

  const [activeTab, setActiveTab] = useState<
    'overview' | 'documentation' | 'reviews' | 'versions' | 'support'
  >('overview');

  if (!agent) return <AgentDetailSkeleton />;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b">
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-start gap-8">
            {/* Agent Icon */}
            <div className="w-20 h-20 bg-gradient-to-br from-green-400 to-emerald-500 rounded-xl flex items-center justify-center">
              <Icon name={agent.icon} className="w-10 h-10 text-white" />
            </div>

            {/* Agent Info */}
            <div className="flex-1">
              <div className="flex items-center gap-4 mb-2">
                <h1 className="text-3xl font-bold">{agent.name}</h1>
                {agent.certification && (
                  <Badge variant={agent.certification} size="lg">
                    {agent.certification}
                  </Badge>
                )}
                <Badge variant="outline">v{agent.version}</Badge>
              </div>

              <p className="text-lg text-gray-600 mb-4">
                {agent.description}
              </p>

              <div className="flex items-center gap-6 text-sm">
                <Link
                  href={`/users/${agent.author.id}`}
                  className="flex items-center gap-2 hover:underline"
                >
                  <Avatar src={agent.author.avatar} size="sm" />
                  {agent.author.name}
                </Link>

                <div className="flex items-center gap-1">
                  <Star className="w-4 h-4 text-yellow-500" />
                  <span className="font-medium">{agent.rating.toFixed(1)}</span>
                  <span className="text-gray-500">({agent.reviews} reviews)</span>
                </div>

                <span className="flex items-center gap-1">
                  <Download className="w-4 h-4" />
                  {formatNumber(agent.downloads)} downloads
                </span>

                <span className="flex items-center gap-1">
                  <GitBranch className="w-4 h-4" />
                  {agent.forks} forks
                </span>
              </div>
            </div>

            {/* Actions */}
            <div className="flex flex-col gap-3">
              <InstallButton agent={agent} size="lg" />

              <button className="px-6 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
                <GitBranch className="w-4 h-4 inline mr-2" />
                Fork
              </button>

              <button className="px-6 py-2 border border-gray-300 rounded-lg hover:bg-gray-50">
                <Star className="w-4 h-4 inline mr-2" />
                Star
              </button>
            </div>
          </div>

          {/* Tags */}
          <div className="flex flex-wrap gap-2 mt-6">
            {agent.tags.map((tag) => (
              <Link
                key={tag}
                href={`/search?tag=${tag}`}
                className="px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-full text-sm"
              >
                {tag}
              </Link>
            ))}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white border-b sticky top-0 z-10">
        <div className="container mx-auto px-4">
          <nav className="flex gap-8">
            {[
              { id: 'overview', label: 'Overview', icon: 'home' },
              { id: 'documentation', label: 'Documentation', icon: 'book' },
              { id: 'reviews', label: 'Reviews', icon: 'message-circle' },
              { id: 'versions', label: 'Versions', icon: 'git-commit' },
              { id: 'support', label: 'Support', icon: 'help-circle' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`
                  py-4 px-2 border-b-2 font-medium text-sm transition-colors
                  ${
                    activeTab === tab.id
                      ? 'border-green-600 text-green-600'
                      : 'border-transparent text-gray-600 hover:text-gray-900'
                  }
                `}
              >
                <Icon name={tab.icon} className="w-4 h-4 inline mr-2" />
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Tab Content */}
      <div className="container mx-auto px-4 py-8">
        {activeTab === 'overview' && <AgentOverview agent={agent} />}
        {activeTab === 'documentation' && <AgentDocumentation agent={agent} />}
        {activeTab === 'reviews' && <AgentReviews agent={agent} />}
        {activeTab === 'versions' && <AgentVersions agent={agent} />}
        {activeTab === 'support' && <AgentSupport agent={agent} />}
      </div>
    </div>
  );
};
```

### 5. Installation System

```typescript
// One-click Installation Component
export const InstallButton: React.FC<{
  agent: Agent;
  size?: 'sm' | 'md' | 'lg';
}> = ({ agent, size = 'md' }) => {
  const [isInstalling, setIsInstalling] = useState(false);
  const [installMethod, setInstallMethod] = useState<'cli' | 'api' | 'download'>('cli');
  const [showInstructions, setShowInstructions] = useState(false);

  const handleInstall = async () => {
    setIsInstalling(true);

    try {
      // Track installation
      await trackInstallation(agent.id);

      // Generate installation command
      const command = generateInstallCommand(agent, installMethod);

      // Copy to clipboard
      await navigator.clipboard.writeText(command);

      // Show success notification
      toast.success('Installation command copied to clipboard!');
      setShowInstructions(true);
    } catch (error) {
      toast.error('Installation failed');
    } finally {
      setIsInstalling(false);
    }
  };

  const sizeClasses = {
    sm: 'px-3 py-1 text-sm',
    md: 'px-4 py-2',
    lg: 'px-6 py-3 text-lg',
  };

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            className={`
              ${sizeClasses[size]}
              bg-green-600 text-white rounded-lg hover:bg-green-700
              flex items-center gap-2
              ${isInstalling ? 'opacity-50 cursor-not-allowed' : ''}
            `}
            disabled={isInstalling}
          >
            <Download className="w-4 h-4" />
            {isInstalling ? 'Installing...' : 'Install'}
            <ChevronDown className="w-4 h-4" />
          </button>
        </DropdownMenuTrigger>

        <DropdownMenuContent>
          <DropdownMenuItem onClick={() => handleInstallWithMethod('cli')}>
            <Terminal className="w-4 h-4 mr-2" />
            Install via CLI
          </DropdownMenuItem>

          <DropdownMenuItem onClick={() => handleInstallWithMethod('python')}>
            <Code className="w-4 h-4 mr-2" />
            Install via pip
          </DropdownMenuItem>

          <DropdownMenuItem onClick={() => handleInstallWithMethod('npm')}>
            <Package className="w-4 h-4 mr-2" />
            Install via npm
          </DropdownMenuItem>

          <DropdownMenuSeparator />

          <DropdownMenuItem onClick={() => handleInstallWithMethod('docker')}>
            <Box className="w-4 h-4 mr-2" />
            Docker Image
          </DropdownMenuItem>

          <DropdownMenuItem onClick={() => handleInstallWithMethod('download')}>
            <Download className="w-4 h-4 mr-2" />
            Download Package
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      {/* Installation Instructions Modal */}
      {showInstructions && (
        <InstallationInstructions
          agent={agent}
          method={installMethod}
          onClose={() => setShowInstructions(false)}
        />
      )}
    </>
  );
};

// Installation Instructions Modal
export const InstallationInstructions: React.FC<{
  agent: Agent;
  method: string;
  onClose: () => void;
}> = ({ agent, method, onClose }) => {
  const instructions = getInstallationInstructions(agent, method);

  return (
    <Dialog open onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Installing {agent.name}</DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {/* Step 1: Install */}
          <div>
            <h3 className="font-medium mb-2">1. Install the agent</h3>
            <CodeBlock
              language="bash"
              code={instructions.installCommand}
              copyButton
            />
          </div>

          {/* Step 2: Configure */}
          <div>
            <h3 className="font-medium mb-2">2. Configure your environment</h3>
            <CodeBlock
              language="bash"
              code={instructions.configCommand}
              copyButton
            />
          </div>

          {/* Step 3: Usage Example */}
          <div>
            <h3 className="font-medium mb-2">3. Usage example</h3>
            <CodeBlock
              language="python"
              code={instructions.usageExample}
              copyButton
            />
          </div>

          {/* Documentation Link */}
          <Alert>
            <InfoIcon className="w-4 h-4" />
            <AlertTitle>Need help?</AlertTitle>
            <AlertDescription>
              Check out the{' '}
              <Link
                href={`/agents/${agent.id}/documentation`}
                className="text-green-600 hover:underline"
              >
                full documentation
              </Link>{' '}
              for detailed setup instructions and examples.
            </AlertDescription>
          </Alert>
        </div>
      </DialogContent>
    </Dialog>
  );
};
```

### 6. Rating and Review System

```typescript
// Review Component
export const ReviewSystem: React.FC<{ agentId: string }> = ({ agentId }) => {
  const [rating, setRating] = useState(0);
  const [review, setReview] = useState('');
  const [showReviewForm, setShowReviewForm] = useState(false);

  const { data: reviews } = useQuery({
    queryKey: ['reviews', agentId],
    queryFn: () => getAgentReviews(agentId),
  });

  const submitReview = useMutation({
    mutationFn: (data: ReviewData) => postReview(agentId, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['reviews', agentId]);
      setShowReviewForm(false);
      toast.success('Review submitted successfully!');
    },
  });

  return (
    <div className="space-y-6">
      {/* Review Summary */}
      <div className="bg-white rounded-lg p-6">
        <div className="flex items-start gap-8">
          <div className="text-center">
            <div className="text-5xl font-bold">{reviews?.summary.average.toFixed(1)}</div>
            <RatingStars rating={reviews?.summary.average || 0} size="lg" />
            <div className="text-sm text-gray-500 mt-2">
              {reviews?.summary.total} reviews
            </div>
          </div>

          <div className="flex-1">
            <h3 className="font-semibold mb-4">Rating Distribution</h3>
            {[5, 4, 3, 2, 1].map((star) => (
              <div key={star} className="flex items-center gap-3 mb-2">
                <span className="text-sm w-4">{star}</span>
                <div className="flex-1 bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-green-600 h-full rounded-full"
                    style={{
                      width: `${
                        ((reviews?.summary.distribution[star] || 0) /
                          reviews?.summary.total) *
                        100
                      }%`,
                    }}
                  />
                </div>
                <span className="text-sm text-gray-600 w-12">
                  {reviews?.summary.distribution[star] || 0}
                </span>
              </div>
            ))}
          </div>

          <div>
            <button
              onClick={() => setShowReviewForm(true)}
              className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700"
            >
              Write a Review
            </button>
          </div>
        </div>
      </div>

      {/* Review Form */}
      {showReviewForm && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg p-6"
        >
          <h3 className="font-semibold mb-4">Write Your Review</h3>

          {/* Star Rating */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Rating</label>
            <div className="flex gap-2">
              {[1, 2, 3, 4, 5].map((star) => (
                <button
                  key={star}
                  onClick={() => setRating(star)}
                  className="text-3xl"
                >
                  <Star
                    className={`w-8 h-8 ${
                      star <= rating
                        ? 'text-yellow-500 fill-current'
                        : 'text-gray-300'
                    }`}
                  />
                </button>
              ))}
            </div>
          </div>

          {/* Review Text */}
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">
              Your Review
            </label>
            <textarea
              value={review}
              onChange={(e) => setReview(e.target.value)}
              rows={4}
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-green-500"
              placeholder="Share your experience with this agent..."
            />
          </div>

          {/* Submit */}
          <div className="flex gap-3">
            <button
              onClick={() => submitReview.mutate({ rating, review })}
              disabled={!rating || !review}
              className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
            >
              Submit Review
            </button>
            <button
              onClick={() => setShowReviewForm(false)}
              className="px-6 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
          </div>
        </motion.div>
      )}

      {/* Review List */}
      <div className="space-y-4">
        {reviews?.items.map((review) => (
          <ReviewCard key={review.id} review={review} />
        ))}
      </div>
    </div>
  );
};
```

### 7. Search and Filtering System

```typescript
// Advanced Search Component
export const AdvancedSearch: React.FC = () => {
  const [searchState, setSearchState] = useState<SearchState>({
    query: '',
    filters: {
      categories: [],
      tags: [],
      certifications: [],
      priceRange: { min: 0, max: 1000 },
      ratingMin: 0,
      languages: [],
      compatibility: [],
    },
    sort: 'relevance',
  });

  const { data: results, isLoading } = useQuery({
    queryKey: ['search', searchState],
    queryFn: () => performSearch(searchState),
    debounceWait: 300,
  });

  return (
    <div className="max-w-7xl mx-auto">
      {/* Search Header */}
      <div className="mb-8">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            value={searchState.query}
            onChange={(e) =>
              setSearchState((s) => ({ ...s, query: e.target.value }))
            }
            className="w-full pl-12 pr-4 py-4 text-lg border rounded-xl focus:ring-2 focus:ring-green-500"
            placeholder="Search agents, categories, or tags..."
          />

          {/* Search Suggestions */}
          {searchState.query && (
            <SearchSuggestions
              query={searchState.query}
              onSelect={(suggestion) =>
                setSearchState((s) => ({ ...s, query: suggestion }))
              }
            />
          )}
        </div>
      </div>

      {/* Filter Tags */}
      <div className="mb-6">
        <ActiveFilters
          filters={searchState.filters}
          onRemove={(filterType, value) => {
            // Remove filter logic
          }}
        />
      </div>

      {/* Results Header */}
      <div className="flex items-center justify-between mb-6">
        <p className="text-gray-600">
          {results?.total || 0} agents found
          {searchState.query && ` for "${searchState.query}"`}
        </p>

        <Select
          value={searchState.sort}
          onValueChange={(value) =>
            setSearchState((s) => ({ ...s, sort: value }))
          }
        >
          <SelectTrigger className="w-48">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="relevance">Most Relevant</SelectItem>
            <SelectItem value="downloads">Most Downloaded</SelectItem>
            <SelectItem value="rating">Highest Rated</SelectItem>
            <SelectItem value="recent">Recently Updated</SelectItem>
            <SelectItem value="trending">Trending</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Search Results */}
      {isLoading ? (
        <SearchResultsSkeleton />
      ) : (
        <SearchResults results={results} />
      )}
    </div>
  );
};
```

### 8. Analytics Dashboard

```typescript
// Agent Analytics for Publishers
export const PublisherDashboard: React.FC = () => {
  const { data: analytics } = useQuery({
    queryKey: ['publisher-analytics'],
    queryFn: getPublisherAnalytics,
  });

  return (
    <div className="space-y-6">
      {/* Overview Cards */}
      <div className="grid grid-cols-4 gap-6">
        <MetricCard
          title="Total Downloads"
          value={formatNumber(analytics?.totalDownloads || 0)}
          change={analytics?.downloadsChange}
          icon="download"
        />
        <MetricCard
          title="Active Users"
          value={formatNumber(analytics?.activeUsers || 0)}
          change={analytics?.usersChange}
          icon="users"
        />
        <MetricCard
          title="Revenue"
          value={`$${formatNumber(analytics?.revenue || 0)}`}
          change={analytics?.revenueChange}
          icon="dollar-sign"
        />
        <MetricCard
          title="Average Rating"
          value={analytics?.averageRating?.toFixed(1) || '0.0'}
          subtitle={`${analytics?.totalReviews || 0} reviews`}
          icon="star"
        />
      </div>

      {/* Downloads Chart */}
      <div className="bg-white rounded-lg p-6">
        <h3 className="font-semibold mb-4">Downloads Over Time</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={analytics?.downloadsTimeline}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="downloads"
              stroke="#10b981"
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Agent Performance Table */}
      <div className="bg-white rounded-lg p-6">
        <h3 className="font-semibold mb-4">Agent Performance</h3>
        <DataTable
          columns={[
            { key: 'name', label: 'Agent Name' },
            { key: 'downloads', label: 'Downloads', sortable: true },
            { key: 'rating', label: 'Rating', sortable: true },
            { key: 'revenue', label: 'Revenue', sortable: true },
            { key: 'status', label: 'Status' },
          ]}
          data={analytics?.agents || []}
        />
      </div>
    </div>
  );
};
```

### 9. Category and Tag Management

```typescript
// Category Browser
export const CategoryBrowser: React.FC = () => {
  const categories = [
    {
      id: 'compliance',
      name: 'Compliance & Reporting',
      icon: 'shield',
      color: 'blue',
      subcategories: ['CSRD', 'CBAM', 'SEC', 'TCFD'],
      agentCount: 234,
    },
    {
      id: 'carbon',
      name: 'Carbon Accounting',
      icon: 'leaf',
      color: 'green',
      subcategories: ['Scope 1', 'Scope 2', 'Scope 3', 'Product Carbon'],
      agentCount: 456,
    },
    {
      id: 'energy',
      name: 'Energy Management',
      icon: 'zap',
      color: 'yellow',
      subcategories: ['Consumption', 'Renewable', 'Efficiency', 'Grid'],
      agentCount: 189,
    },
    {
      id: 'supply-chain',
      name: 'Supply Chain',
      icon: 'truck',
      color: 'purple',
      subcategories: ['Logistics', 'Suppliers', 'Transportation', 'Warehousing'],
      agentCount: 312,
    },
    {
      id: 'data',
      name: 'Data & Integration',
      icon: 'database',
      color: 'gray',
      subcategories: ['ERP', 'IoT', 'APIs', 'ETL'],
      agentCount: 567,
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6">
      {categories.map((category) => (
        <Link
          key={category.id}
          href={`/category/${category.id}`}
          className="group"
        >
          <div className="bg-white rounded-lg p-6 hover:shadow-lg transition-all">
            <div
              className={`
                w-12 h-12 rounded-lg flex items-center justify-center mb-4
                bg-${category.color}-100 group-hover:bg-${category.color}-200
              `}
            >
              <Icon
                name={category.icon}
                className={`w-6 h-6 text-${category.color}-600`}
              />
            </div>

            <h3 className="font-semibold mb-2">{category.name}</h3>
            <p className="text-sm text-gray-600 mb-4">
              {category.agentCount} agents
            </p>

            <div className="space-y-1">
              {category.subcategories.slice(0, 3).map((sub) => (
                <div
                  key={sub}
                  className="text-xs text-gray-500 hover:text-gray-700"
                >
                  {sub}
                </div>
              ))}
              {category.subcategories.length > 3 && (
                <div className="text-xs text-gray-400">
                  +{category.subcategories.length - 3} more
                </div>
              )}
            </div>
          </div>
        </Link>
      ))}
    </div>
  );
};
```

### 10. Performance & Infrastructure

```yaml
Performance Targets:
  - Search response: <50ms (with Meilisearch)
  - Page load: <2s (with SSG/ISR)
  - Agent install: <5s (CDN distributed)
  - API response: <100ms (with caching)

Caching Strategy:
  - Agent metadata: 1 hour
  - Search results: 5 minutes
  - User data: Real-time
  - Downloads count: 1 minute
  - Static assets: 1 year

Scalability:
  - Support 10,000+ agents
  - Handle 1M+ monthly downloads
  - 100K+ concurrent users
  - 10TB+ package storage

Infrastructure:
  - Multi-region deployment
  - Auto-scaling groups
  - Load balancing
  - DDoS protection
  - 99.9% uptime SLA
```

### 11. Timeline & Milestones

```yaml
Q3 2026: Beta Launch
  Month 1:
    - Core hub interface
    - Basic search functionality
    - Agent cards and listings
    - Installation system

  Month 2:
    - Agent detail pages
    - Review system
    - User authentication
    - Publisher dashboard

  Month 3:
    - Advanced search
    - Category browsing
    - Analytics integration
    - Beta testing with 100 agents

Q4 2026: Public Launch
  - 500+ agents available
  - Full feature set
  - Mobile responsive
  - API access
  - Documentation complete

Q1 2027: Growth Phase
  - 2,000+ agents
  - Marketplace integration
  - Enterprise features
  - AI-powered recommendations
  - Global CDN deployment
```