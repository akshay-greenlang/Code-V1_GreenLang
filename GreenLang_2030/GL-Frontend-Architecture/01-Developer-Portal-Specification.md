# GreenLang Developer Portal Specification
## docs.greenlang.io - Next.js/React Architecture

### 1. Technical Architecture

#### Core Stack
```yaml
Framework: Next.js 14 with App Router
Language: TypeScript 5.3+
Styling: Tailwind CSS 3.4 + CSS Modules
State: Zustand 4.5 + React Query 5.0
Search: Algolia DocSearch
Analytics: Vercel Analytics + PostHog
CDN: Vercel Edge Network
Database: PostgreSQL (Neon) + Redis (Upstash)
```

#### Project Structure
```
docs.greenlang.io/
├── app/
│   ├── (docs)/
│   │   ├── getting-started/
│   │   ├── api-reference/
│   │   ├── guides/
│   │   ├── tutorials/
│   │   └── examples/
│   ├── (playground)/
│   │   ├── editor/
│   │   └── share/
│   ├── api/
│   │   ├── execute/
│   │   ├── search/
│   │   └── feedback/
│   └── layout.tsx
├── components/
│   ├── documentation/
│   │   ├── CodeBlock/
│   │   ├── ApiReference/
│   │   ├── Navigation/
│   │   └── TableOfContents/
│   ├── playground/
│   │   ├── Editor/
│   │   ├── Console/
│   │   └── ShareModal/
│   └── shared/
│       ├── Search/
│       ├── ThemeToggle/
│       └── Feedback/
├── lib/
│   ├── mdx/
│   ├── search/
│   ├── analytics/
│   └── api/
└── content/
    ├── docs/
    ├── api/
    └── examples/
```

### 2. Core Components Library

#### Documentation Components (25+ components)

```typescript
// CodeBlock Component with live execution
interface CodeBlockProps {
  language: 'python' | 'typescript' | 'gcel' | 'yaml';
  code: string;
  executable?: boolean;
  showLineNumbers?: boolean;
  highlightLines?: number[];
  title?: string;
  copyButton?: boolean;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({
  language,
  code,
  executable = false,
  showLineNumbers = true,
  highlightLines = [],
  title,
  copyButton = true
}) => {
  const [output, setOutput] = useState<string>('');
  const [isRunning, setIsRunning] = useState(false);

  const executeCode = async () => {
    setIsRunning(true);
    try {
      const response = await fetch('/api/execute', {
        method: 'POST',
        body: JSON.stringify({ code, language }),
      });
      const result = await response.json();
      setOutput(result.output);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-800">
      {title && (
        <div className="px-4 py-2 border-b border-gray-200 dark:border-gray-800">
          <span className="text-sm font-medium">{title}</span>
        </div>
      )}
      <div className="relative">
        <Highlight
          code={code}
          language={language}
          theme={themes.nightOwlLight}
        >
          {({ className, style, tokens, getLineProps, getTokenProps }) => (
            <pre className={className} style={style}>
              {tokens.map((line, i) => (
                <div
                  key={i}
                  {...getLineProps({ line, key: i })}
                  className={highlightLines.includes(i + 1) ? 'bg-yellow-100' : ''}
                >
                  {showLineNumbers && (
                    <span className="select-none text-gray-500 mr-4">
                      {i + 1}
                    </span>
                  )}
                  {line.map((token, key) => (
                    <span key={key} {...getTokenProps({ token, key })} />
                  ))}
                </div>
              ))}
            </pre>
          )}
        </Highlight>
        {copyButton && <CopyButton text={code} />}
        {executable && (
          <button
            onClick={executeCode}
            disabled={isRunning}
            className="absolute top-2 right-12 px-3 py-1 text-sm bg-green-600 text-white rounded"
          >
            {isRunning ? 'Running...' : 'Run'}
          </button>
        )}
      </div>
      {output && (
        <div className="border-t border-gray-200 dark:border-gray-800 p-4">
          <pre className="text-sm">{output}</pre>
        </div>
      )}
    </div>
  );
};
```

#### API Reference Component
```typescript
interface ApiEndpoint {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  path: string;
  description: string;
  parameters: Parameter[];
  requestBody?: RequestBody;
  responses: Response[];
  examples: Example[];
}

export const ApiReference: React.FC<{ endpoint: ApiEndpoint }> = ({ endpoint }) => {
  const [activeTab, setActiveTab] = useState<'parameters' | 'request' | 'response'>('parameters');
  const [testResponse, setTestResponse] = useState(null);

  return (
    <div className="border rounded-lg p-6 my-8">
      <div className="flex items-center gap-4 mb-6">
        <span className={`px-3 py-1 rounded text-white font-medium ${
          endpoint.method === 'GET' ? 'bg-blue-600' :
          endpoint.method === 'POST' ? 'bg-green-600' :
          endpoint.method === 'PUT' ? 'bg-yellow-600' :
          'bg-red-600'
        }`}>
          {endpoint.method}
        </span>
        <code className="text-lg font-mono">{endpoint.path}</code>
      </div>

      <p className="text-gray-600 dark:text-gray-400 mb-6">
        {endpoint.description}
      </p>

      <Tabs value={activeTab} onChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="parameters">Parameters</TabsTrigger>
          <TabsTrigger value="request">Request</TabsTrigger>
          <TabsTrigger value="response">Response</TabsTrigger>
          <TabsTrigger value="try">Try it</TabsTrigger>
        </TabsList>

        <TabsContent value="parameters">
          <ParametersTable parameters={endpoint.parameters} />
        </TabsContent>

        <TabsContent value="request">
          <RequestBodyEditor body={endpoint.requestBody} />
        </TabsContent>

        <TabsContent value="response">
          <ResponseViewer responses={endpoint.responses} />
        </TabsContent>

        <TabsContent value="try">
          <ApiTester endpoint={endpoint} onResponse={setTestResponse} />
        </TabsContent>
      </Tabs>
    </div>
  );
};
```

### 3. Interactive Playground

```typescript
// Live GCEL Playground Component
export const GCELPlayground: React.FC = () => {
  const [code, setCode] = useState(`
# Build your climate intelligence chain
from greenlang import GCEL

chain = (
  intake >>
  validate >>
  calculate >>
  report
)

result = chain.run(data)
  `);

  const [output, setOutput] = useState('');
  const [visualGraph, setVisualGraph] = useState(null);
  const [shareUrl, setShareUrl] = useState('');

  const runCode = async () => {
    const response = await fetch('/api/playground/execute', {
      method: 'POST',
      body: JSON.stringify({ code }),
    });
    const result = await response.json();
    setOutput(result.output);
    setVisualGraph(result.graph);
  };

  const shareCode = async () => {
    const response = await fetch('/api/playground/share', {
      method: 'POST',
      body: JSON.stringify({ code }),
    });
    const { url } = await response.json();
    setShareUrl(url);
  };

  return (
    <div className="grid grid-cols-2 gap-4 h-screen">
      <div className="flex flex-col">
        <div className="flex justify-between items-center p-4 border-b">
          <h3 className="font-semibold">GCEL Editor</h3>
          <div className="flex gap-2">
            <button onClick={runCode} className="px-4 py-2 bg-green-600 text-white rounded">
              Run
            </button>
            <button onClick={shareCode} className="px-4 py-2 bg-blue-600 text-white rounded">
              Share
            </button>
          </div>
        </div>
        <MonacoEditor
          height="100%"
          language="python"
          value={code}
          onChange={setCode}
          theme="vs-dark"
          options={{
            minimap: { enabled: false },
            fontSize: 14,
          }}
        />
      </div>

      <div className="flex flex-col">
        <Tabs defaultValue="output">
          <TabsList>
            <TabsTrigger value="output">Output</TabsTrigger>
            <TabsTrigger value="graph">Visual Graph</TabsTrigger>
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
          </TabsList>

          <TabsContent value="output">
            <pre className="p-4 bg-gray-100 dark:bg-gray-900 rounded">
              {output || 'Run your code to see output'}
            </pre>
          </TabsContent>

          <TabsContent value="graph">
            {visualGraph && <ChainVisualizer graph={visualGraph} />}
          </TabsContent>

          <TabsContent value="metrics">
            <PerformanceMetrics code={code} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};
```

### 4. Search Implementation (Algolia)

```typescript
// Search Configuration
export const searchConfig = {
  appId: process.env.NEXT_PUBLIC_ALGOLIA_APP_ID,
  apiKey: process.env.NEXT_PUBLIC_ALGOLIA_SEARCH_KEY,
  indexName: 'greenlang_docs',

  // Search parameters
  facetFilters: [
    'type:guide',
    'type:api',
    'type:tutorial',
    'type:example',
    'version:latest'
  ],

  // UI Configuration
  placeholder: 'Search documentation...',
  searchParameters: {
    hitsPerPage: 10,
    attributesToRetrieve: ['title', 'content', 'url', 'type', 'category'],
    attributesToHighlight: ['title', 'content'],
    attributesToSnippet: ['content:50'],
  }
};

// Search Component with AI-powered suggestions
export const SearchBar: React.FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [aiSuggestions, setAiSuggestions] = useState([]);

  const search = useDebouncedCallback(async (searchQuery: string) => {
    // Algolia search
    const algoliaResults = await searchClient.search(searchQuery);
    setResults(algoliaResults.hits);

    // AI-powered suggestions
    const suggestions = await fetch('/api/search/suggestions', {
      method: 'POST',
      body: JSON.stringify({ query: searchQuery }),
    }).then(r => r.json());
    setAiSuggestions(suggestions);
  }, 300);

  return (
    <Command className="rounded-lg border shadow-md">
      <CommandInput
        placeholder={searchConfig.placeholder}
        value={query}
        onValueChange={(value) => {
          setQuery(value);
          search(value);
        }}
      />
      <CommandList>
        {aiSuggestions.length > 0 && (
          <CommandGroup heading="Suggested">
            {aiSuggestions.map((suggestion) => (
              <CommandItem key={suggestion.id}>
                <Link href={suggestion.url}>{suggestion.title}</Link>
              </CommandItem>
            ))}
          </CommandGroup>
        )}

        <CommandGroup heading="Search Results">
          {results.map((result) => (
            <CommandItem key={result.objectID}>
              <Link href={result.url}>
                <div>
                  <div className="font-medium">{result.title}</div>
                  <div className="text-sm text-gray-500">
                    {result._snippetResult.content.value}
                  </div>
                </div>
              </Link>
            </CommandItem>
          ))}
        </CommandGroup>
      </CommandList>
    </Command>
  );
};
```

### 5. Dark Mode Implementation

```typescript
// Theme Provider with system preference detection
export const ThemeProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [theme, setTheme] = useState<'light' | 'dark' | 'system'>('system');
  const [resolvedTheme, setResolvedTheme] = useState<'light' | 'dark'>('light');

  useEffect(() => {
    const stored = localStorage.getItem('theme') as 'light' | 'dark' | 'system';
    if (stored) setTheme(stored);

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = () => {
      if (theme === 'system') {
        setResolvedTheme(mediaQuery.matches ? 'dark' : 'light');
      }
    };

    handleChange();
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [theme]);

  useEffect(() => {
    if (theme === 'system') {
      const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      setResolvedTheme(isDark ? 'dark' : 'light');
    } else {
      setResolvedTheme(theme);
    }

    document.documentElement.classList.toggle('dark', resolvedTheme === 'dark');
  }, [theme, resolvedTheme]);

  return (
    <ThemeContext.Provider value={{ theme, setTheme, resolvedTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};
```

### 6. Mobile Responsive Design

```typescript
// Responsive Navigation Component
export const Navigation: React.FC = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [activeSection, setActiveSection] = useState('');

  return (
    <>
      {/* Desktop Navigation */}
      <nav className="hidden lg:flex fixed left-0 top-16 bottom-0 w-64 border-r overflow-y-auto">
        <div className="p-4 space-y-2">
          <NavSection title="Getting Started" items={gettingStartedItems} />
          <NavSection title="Core Concepts" items={coreConceptItems} />
          <NavSection title="API Reference" items={apiReferenceItems} />
          <NavSection title="Guides" items={guideItems} />
        </div>
      </nav>

      {/* Mobile Navigation */}
      <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
        <SheetTrigger asChild>
          <button className="lg:hidden fixed bottom-4 right-4 z-50 p-4 bg-green-600 text-white rounded-full shadow-lg">
            <Menu />
          </button>
        </SheetTrigger>
        <SheetContent side="left" className="w-[300px]">
          <SheetHeader>
            <SheetTitle>Documentation</SheetTitle>
          </SheetHeader>
          <div className="mt-8 space-y-2">
            <NavSection title="Getting Started" items={gettingStartedItems} mobile />
            <NavSection title="Core Concepts" items={coreConceptItems} mobile />
            <NavSection title="API Reference" items={apiReferenceItems} mobile />
          </div>
        </SheetContent>
      </Sheet>

      {/* Table of Contents (Desktop) */}
      <aside className="hidden xl:block fixed right-0 top-16 bottom-0 w-64 p-4">
        <div className="sticky top-4">
          <h4 className="font-semibold mb-4">On this page</h4>
          <TableOfContents activeSection={activeSection} />
        </div>
      </aside>
    </>
  );
};
```

### 7. Performance Optimization

```typescript
// Performance optimizations configuration
export const performanceConfig = {
  // Next.js optimizations
  images: {
    formats: ['image/avif', 'image/webp'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
  },

  // Bundle optimization
  experimental: {
    optimizeCss: true,
    optimizePackageImports: ['@greenlang/ui', 'lucide-react'],
  },

  // Caching strategy
  cacheStrategy: {
    static: 31536000, // 1 year
    dynamic: 3600, // 1 hour
    api: 60, // 1 minute
  },

  // Prefetch configuration
  prefetch: {
    viewport: true,
    hover: true,
    priority: ['getting-started', 'api-reference'],
  }
};

// Lazy loading components
const CodePlayground = dynamic(() => import('./CodePlayground'), {
  loading: () => <PlaygroundSkeleton />,
  ssr: false,
});

// Virtual scrolling for long documentation
export const VirtualDocList: React.FC<{ items: DocItem[] }> = ({ items }) => {
  const parentRef = useRef(null);

  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 100,
    overscan: 5,
  });

  return (
    <div ref={parentRef} className="h-screen overflow-auto">
      <div style={{ height: `${virtualizer.getTotalSize()}px` }}>
        {virtualizer.getVirtualItems().map((virtualItem) => (
          <div
            key={virtualItem.key}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: `${virtualItem.size}px`,
              transform: `translateY(${virtualItem.start}px)`,
            }}
          >
            <DocCard item={items[virtualItem.index]} />
          </div>
        ))}
      </div>
    </div>
  );
};
```

### 8. Component Library Catalog (50+ Components)

```yaml
Documentation Components:
  - CodeBlock
  - ApiReference
  - MethodSignature
  - ParameterTable
  - ResponseViewer
  - ExampleTabs
  - VersionSelector
  - LanguageToggle
  - CopyButton
  - RunButton

Navigation Components:
  - Sidebar
  - Breadcrumbs
  - TableOfContents
  - PageNavigation
  - SearchBar
  - MobileMenu
  - FooterNav
  - QuickLinks
  - CategoryFilter
  - TagCloud

Content Components:
  - Alert
  - Callout
  - Card
  - Tabs
  - Accordion
  - Timeline
  - StepGuide
  - VideoEmbed
  - ImageGallery
  - Diagram

Interactive Components:
  - Playground
  - Terminal
  - Console
  - Editor
  - GraphVisualizer
  - MetricsDisplay
  - FeedbackForm
  - RatingWidget
  - ShareModal
  - ExportDialog

Data Display Components:
  - DataTable
  - Chart
  - StatCard
  - ProgressBar
  - Badge
  - Status
  - Tooltip
  - Popover
  - Modal
  - Notification

Form Components:
  - Input
  - Select
  - Checkbox
  - Radio
  - Switch
  - Slider
  - DatePicker
  - FileUpload
  - ColorPicker
  - CodeInput
```

### 9. Content Management Strategy

```typescript
// MDX-based content with frontmatter
interface DocContent {
  title: string;
  description: string;
  category: 'getting-started' | 'core-concepts' | 'api' | 'guides' | 'examples';
  tags: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  readTime: number; // minutes
  lastUpdated: Date;
  authors: Author[];
  relatedDocs: string[];
}

// Content organization
const contentStructure = {
  'getting-started': {
    'installation': 'Installation guide with system requirements',
    'quickstart': '5-minute quickstart tutorial',
    'first-chain': 'Building your first GCEL chain',
    'basic-concepts': 'Understanding core concepts',
  },
  'core-concepts': {
    'agents': 'Deep dive into GreenLang agents',
    'gcel': 'GreenLang Climate Expression Language',
    'composability': 'Building composable workflows',
    'data-flow': 'Understanding data flow',
  },
  'api-reference': {
    'agents': 'Complete agent API reference',
    'gcel': 'GCEL operators and methods',
    'client': 'Client SDK reference',
    'webhooks': 'Webhook endpoints',
  },
  'guides': {
    'csrd-reporting': 'Complete CSRD reporting guide',
    'cbam-compliance': 'CBAM compliance implementation',
    'scope3-calculation': 'Scope 3 emissions calculation',
    'data-quality': 'Improving data quality scores',
  },
  'examples': {
    'basic': 'Simple calculation examples',
    'advanced': 'Complex workflow examples',
    'industry': 'Industry-specific implementations',
    'integrations': 'Integration examples',
  }
};
```

### 10. Analytics and Metrics

```typescript
// Analytics tracking configuration
export const analytics = {
  // Page views
  trackPageView: (path: string) => {
    vercelAnalytics.track('page_view', { path });
    posthog.capture('$pageview', { path });
  },

  // Documentation engagement
  trackDocEngagement: (event: {
    action: 'copy_code' | 'run_example' | 'download' | 'feedback';
    page: string;
    section?: string;
    value?: any;
  }) => {
    vercelAnalytics.track('doc_engagement', event);
    posthog.capture(`doc_${event.action}`, event);
  },

  // Search metrics
  trackSearch: (query: string, results: number) => {
    posthog.capture('search', { query, results });
  },

  // Playground usage
  trackPlayground: (action: 'run' | 'share' | 'export', code: string) => {
    posthog.capture(`playground_${action}`, {
      codeLength: code.length,
      timestamp: new Date().toISOString(),
    });
  }
};

// Performance monitoring
export const performanceMonitoring = {
  // Core Web Vitals
  reportWebVitals: (metric: Metric) => {
    const body = JSON.stringify({
      name: metric.name,
      value: metric.value,
      rating: metric.rating,
      delta: metric.delta,
      id: metric.id,
    });

    fetch('/api/metrics', {
      method: 'POST',
      body,
      keepalive: true,
    });
  },

  // Custom metrics
  measureApiLatency: async (endpoint: string) => {
    const start = performance.now();
    await fetch(endpoint);
    const duration = performance.now() - start;

    analytics.trackDocEngagement({
      action: 'api_latency',
      page: endpoint,
      value: duration,
    });
  }
};
```

### 11. Deployment & Infrastructure

```yaml
Infrastructure:
  Hosting: Vercel Pro
  Database: Neon (PostgreSQL)
  Cache: Upstash (Redis)
  Search: Algolia
  CDN: Vercel Edge Network
  Analytics: Vercel Analytics + PostHog

CI/CD Pipeline:
  - GitHub Actions for testing
  - Vercel Preview Deployments
  - Automatic production deploys
  - Lighthouse CI for performance
  - Bundle size monitoring

Performance Targets:
  - Lighthouse Score: 95+
  - First Contentful Paint: <1.2s
  - Time to Interactive: <3.5s
  - Bundle Size: <200KB (initial)
  - Search Response: <100ms
```

### 12. Timeline & Milestones

```yaml
Q1 2026:
  Month 1:
    - Setup Next.js project structure
    - Implement core components (20)
    - Basic documentation pages (50)
    - Search integration

  Month 2:
    - Complete component library (50)
    - Interactive playground v1
    - Dark mode support
    - Mobile responsive design

  Month 3:
    - 500+ documentation pages
    - API reference complete
    - Performance optimization
    - Launch beta version

Q2 2026:
  - 1000+ pages of documentation
  - Advanced playground features
  - Community contributions
  - Analytics dashboard
  - International support (i18n)

Q3 2026:
  - AI-powered search
  - Video tutorials integration
  - Advanced code examples
  - Performance monitoring
  - Enterprise documentation portal
```