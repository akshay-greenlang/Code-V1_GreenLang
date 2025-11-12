# GreenLang Studio Specification
## Observability & Testing Platform (Like LangSmith for GreenLang)

### 1. Technical Architecture

#### Core Stack
```yaml
Frontend:
  Framework: Next.js 14 + React 18
  Language: TypeScript 5.3+
  UI: Tremor + Tailwind CSS
  State: Valtio + React Query
  Visualization: Cytoscape.js + D3.js
  Code Editor: Monaco Editor

Backend Integration:
  Tracing: OpenTelemetry
  Storage: ClickHouse (traces)
  Search: Elasticsearch
  Streaming: Apache Kafka
  Analytics: Apache Druid

Observability Stack:
  Metrics: Prometheus + Grafana
  Logs: Loki
  Traces: Jaeger/Tempo
  Profiling: Pyroscope
  APM: Custom implementation
```

### 2. Trace Visualization System

```typescript
// Main Studio Interface
export const GreenLangStudio: React.FC = () => {
  const [selectedProject, setSelectedProject] = useState<string>();
  const [timeRange, setTimeRange] = useState<TimeRange>(last24Hours);
  const [activeView, setActiveView] = useState<'traces' | 'playground' | 'datasets' | 'analytics'>('traces');

  return (
    <StudioLayout>
      {/* Header */}
      <StudioHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Logo />
            <ProjectSelector
              value={selectedProject}
              onChange={setSelectedProject}
            />
            <EnvironmentBadge />
          </div>

          <nav className="flex gap-6">
            <NavItem
              active={activeView === 'traces'}
              onClick={() => setActiveView('traces')}
              icon={<Activity />}
            >
              Traces
            </NavItem>
            <NavItem
              active={activeView === 'playground'}
              onClick={() => setActiveView('playground')}
              icon={<PlayCircle />}
            >
              Playground
            </NavItem>
            <NavItem
              active={activeView === 'datasets'}
              onClick={() => setActiveView('datasets')}
              icon={<Database />}
            >
              Datasets
            </NavItem>
            <NavItem
              active={activeView === 'analytics'}
              onClick={() => setActiveView('analytics')}
              icon={<BarChart />}
            >
              Analytics
            </NavItem>
          </nav>

          <div className="flex items-center gap-4">
            <TimeRangeSelector value={timeRange} onChange={setTimeRange} />
            <NotificationCenter />
            <UserMenu />
          </div>
        </div>
      </StudioHeader>

      {/* Content */}
      <StudioContent>
        {activeView === 'traces' && <TracesExplorer project={selectedProject} />}
        {activeView === 'playground' && <ChainPlayground project={selectedProject} />}
        {activeView === 'datasets' && <DatasetManager project={selectedProject} />}
        {activeView === 'analytics' && <AnalyticsHub project={selectedProject} />}
      </StudioContent>
    </StudioLayout>
  );
};

// Trace Explorer Component
export const TracesExplorer: React.FC<{ project: string }> = ({ project }) => {
  const [selectedTrace, setSelectedTrace] = useState<string>();
  const [filters, setFilters] = useState<TraceFilters>({});
  const [view, setView] = useState<'list' | 'timeline' | 'graph'>('list');

  const { data: traces, isLoading } = useTraces({
    project,
    filters,
    limit: 100,
  });

  return (
    <div className="flex h-full">
      {/* Traces List */}
      <div className="w-96 border-r bg-gray-50 overflow-y-auto">
        {/* Filters */}
        <div className="p-4 border-b bg-white">
          <TraceFilters
            value={filters}
            onChange={setFilters}
            stats={traces?.stats}
          />
        </div>

        {/* Trace List */}
        <div className="divide-y">
          {traces?.items.map((trace) => (
            <TraceListItem
              key={trace.id}
              trace={trace}
              selected={selectedTrace === trace.id}
              onClick={() => setSelectedTrace(trace.id)}
            />
          ))}
        </div>
      </div>

      {/* Trace Detail */}
      <div className="flex-1 overflow-hidden">
        {selectedTrace ? (
          <TraceDetail
            traceId={selectedTrace}
            view={view}
            onViewChange={setView}
          />
        ) : (
          <EmptyState
            icon={<Activity />}
            title="Select a trace"
            description="Choose a trace from the list to view details"
          />
        )}
      </div>
    </div>
  );
};

// Trace Detail Visualization
export const TraceDetail: React.FC<{
  traceId: string;
  view: 'list' | 'timeline' | 'graph';
  onViewChange: (view: string) => void;
}> = ({ traceId, view, onViewChange }) => {
  const { data: trace } = useTraceDetail(traceId);

  if (!trace) return <LoadingSpinner />;

  return (
    <div className="h-full flex flex-col">
      {/* Trace Header */}
      <div className="p-4 border-b bg-white">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-xl font-semibold mb-2">{trace.name}</h2>
            <div className="flex items-center gap-4 text-sm text-gray-600">
              <span>ID: {trace.id.substring(0, 8)}</span>
              <span>Duration: {trace.duration}ms</span>
              <span>Tokens: {trace.tokens}</span>
              <StatusBadge status={trace.status} />
            </div>
          </div>

          <div className="flex items-center gap-2">
            <ViewToggle
              value={view}
              onChange={onViewChange}
              options={[
                { value: 'list', icon: <List /> },
                { value: 'timeline', icon: <Clock /> },
                { value: 'graph', icon: <GitBranch /> },
              ]}
            />
            <Button icon={<Share2 />}>Share</Button>
            <Button icon={<Download />}>Export</Button>
          </div>
        </div>

        {/* Metadata Tags */}
        <div className="flex flex-wrap gap-2 mt-4">
          {Object.entries(trace.metadata).map(([key, value]) => (
            <Tag key={key}>
              {key}: {value}
            </Tag>
          ))}
        </div>
      </div>

      {/* Trace Visualization */}
      <div className="flex-1 overflow-auto p-4">
        {view === 'list' && <TraceListView trace={trace} />}
        {view === 'timeline' && <TraceTimeline trace={trace} />}
        {view === 'graph' && <TraceGraph trace={trace} />}
      </div>

      {/* Trace Footer with Metrics */}
      <div className="p-4 border-t bg-gray-50">
        <div className="grid grid-cols-6 gap-4">
          <MetricCard label="Total Duration" value={`${trace.duration}ms`} />
          <MetricCard label="LLM Calls" value={trace.llmCalls} />
          <MetricCard label="Total Tokens" value={trace.tokens} />
          <MetricCard label="Total Cost" value={`$${trace.cost.toFixed(4)}`} />
          <MetricCard label="Carbon Impact" value={`${trace.carbon}g CO₂`} />
          <MetricCard label="Data Quality" value={`${trace.quality}%`} />
        </div>
      </div>
    </div>
  );
};

// Trace Graph Visualization
export const TraceGraph: React.FC<{ trace: Trace }> = ({ trace }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedNode, setSelectedNode] = useState<string>();

  useEffect(() => {
    if (!containerRef.current) return;

    const cy = cytoscape({
      container: containerRef.current,
      elements: buildGraphElements(trace),
      style: graphStyles,
      layout: {
        name: 'dagre',
        rankDir: 'TB',
        animate: true,
        animationDuration: 500,
      },
    });

    cy.on('tap', 'node', (evt) => {
      setSelectedNode(evt.target.id());
    });

    return () => cy.destroy();
  }, [trace]);

  return (
    <div className="relative h-full">
      <div ref={containerRef} className="absolute inset-0" />

      {/* Node Inspector */}
      {selectedNode && (
        <div className="absolute top-4 right-4 w-96 bg-white rounded-lg shadow-lg p-4">
          <NodeInspector
            nodeId={selectedNode}
            trace={trace}
            onClose={() => setSelectedNode(undefined)}
          />
        </div>
      )}

      {/* Graph Legend */}
      <div className="absolute bottom-4 left-4 bg-white rounded-lg shadow p-3">
        <GraphLegend />
      </div>
    </div>
  );
};

// Build graph elements from trace
const buildGraphElements = (trace: Trace) => {
  const elements = [];

  // Add nodes
  trace.spans.forEach(span => {
    elements.push({
      data: {
        id: span.id,
        label: span.name,
        type: span.type,
        duration: span.duration,
        status: span.status,
        tokens: span.tokens,
      },
    });
  });

  // Add edges
  trace.spans.forEach(span => {
    if (span.parentId) {
      elements.push({
        data: {
          id: `${span.parentId}-${span.id}`,
          source: span.parentId,
          target: span.id,
        },
      });
    }
  });

  return elements;
};
```

### 3. Performance Metrics Dashboard

```typescript
// Performance Analytics Component
export const PerformanceAnalytics: React.FC = () => {
  const [selectedMetric, setSelectedMetric] = useState<'latency' | 'throughput' | 'errors' | 'cost'>('latency');
  const [aggregation, setAggregation] = useState<'p50' | 'p90' | 'p99' | 'avg'>('p50');
  const [groupBy, setGroupBy] = useState<string>('chain');

  const { data: metrics } = usePerformanceMetrics({
    metric: selectedMetric,
    aggregation,
    groupBy,
  });

  return (
    <div className="p-6 space-y-6">
      {/* Metric Selector */}
      <div className="flex items-center justify-between">
        <SegmentedControl
          value={selectedMetric}
          onChange={setSelectedMetric}
          options={[
            { value: 'latency', label: 'Latency' },
            { value: 'throughput', label: 'Throughput' },
            { value: 'errors', label: 'Error Rate' },
            { value: 'cost', label: 'Cost' },
          ]}
        />

        <div className="flex items-center gap-4">
          <Select
            value={aggregation}
            onChange={setAggregation}
            options={[
              { value: 'p50', label: 'Median (P50)' },
              { value: 'p90', label: 'P90' },
              { value: 'p99', label: 'P99' },
              { value: 'avg', label: 'Average' },
            ]}
          />

          <Select
            value={groupBy}
            onChange={setGroupBy}
            options={[
              { value: 'chain', label: 'By Chain' },
              { value: 'agent', label: 'By Agent' },
              { value: 'user', label: 'By User' },
              { value: 'environment', label: 'By Environment' },
            ]}
          />
        </div>
      </div>

      {/* Performance Chart */}
      <Card>
        <PerformanceChart
          data={metrics?.timeseries}
          metric={selectedMetric}
          aggregation={aggregation}
        />
      </Card>

      {/* Performance Breakdown */}
      <div className="grid grid-cols-2 gap-6">
        <Card title="Top Performers">
          <PerformanceTable
            data={metrics?.topPerformers}
            metric={selectedMetric}
            positive
          />
        </Card>

        <Card title="Needs Attention">
          <PerformanceTable
            data={metrics?.bottomPerformers}
            metric={selectedMetric}
            negative
          />
        </Card>
      </div>

      {/* Detailed Metrics Grid */}
      <Card title="Detailed Metrics">
        <div className="grid grid-cols-4 gap-4">
          <MetricBox
            title="Avg Latency"
            value={`${metrics?.summary.avgLatency}ms`}
            trend={metrics?.summary.latencyTrend}
          />
          <MetricBox
            title="Success Rate"
            value={`${metrics?.summary.successRate}%`}
            trend={metrics?.summary.successTrend}
          />
          <MetricBox
            title="Throughput"
            value={`${metrics?.summary.throughput} req/s`}
            trend={metrics?.summary.throughputTrend}
          />
          <MetricBox
            title="Total Cost"
            value={`$${metrics?.summary.totalCost}`}
            trend={metrics?.summary.costTrend}
          />
        </div>
      </Card>

      {/* Latency Distribution */}
      {selectedMetric === 'latency' && (
        <Card title="Latency Distribution">
          <LatencyHistogram data={metrics?.latencyDistribution} />
        </Card>
      )}
    </div>
  );
};
```

### 4. Debug Tools & Chain Testing

```typescript
// Chain Testing Playground
export const ChainPlayground: React.FC = () => {
  const [chain, setChain] = useState<ChainConfig>();
  const [testData, setTestData] = useState<any>({});
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<TestResult>();
  const [activeTab, setActiveTab] = useState<'input' | 'output' | 'trace' | 'compare'>('input');

  const runTest = async () => {
    setIsRunning(true);
    try {
      const res = await testChain(chain, testData);
      setResult(res);
      setActiveTab('output');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="flex h-full">
      {/* Chain Configuration */}
      <div className="w-1/3 border-r p-4 bg-gray-50">
        <h3 className="font-semibold mb-4">Chain Configuration</h3>

        {/* Chain Builder */}
        <ChainBuilder
          value={chain}
          onChange={setChain}
          templates={chainTemplates}
        />

        {/* Test Configuration */}
        <div className="mt-6">
          <h4 className="font-medium mb-2">Test Configuration</h4>
          <TestConfigEditor
            config={testConfig}
            onChange={setTestConfig}
          />
        </div>

        {/* Run Controls */}
        <div className="mt-6 flex gap-3">
          <Button
            type="primary"
            icon={<PlayCircle />}
            onClick={runTest}
            loading={isRunning}
            block
          >
            Run Test
          </Button>
          <Button icon={<Save />}>Save Test</Button>
        </div>
      </div>

      {/* Test Interface */}
      <div className="flex-1 flex flex-col">
        {/* Tabs */}
        <div className="border-b">
          <Tabs value={activeTab} onChange={setActiveTab}>
            <Tab value="input" label="Input" />
            <Tab value="output" label="Output" />
            <Tab value="trace" label="Trace" />
            <Tab value="compare" label="Compare" />
          </Tabs>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-auto p-4">
          {activeTab === 'input' && (
            <TestInputEditor
              value={testData}
              onChange={setTestData}
              schema={chain?.inputSchema}
            />
          )}

          {activeTab === 'output' && result && (
            <TestOutput
              result={result}
              showMetrics
              showLogs
            />
          )}

          {activeTab === 'trace' && result?.trace && (
            <TestTrace
              trace={result.trace}
              interactive
            />
          )}

          {activeTab === 'compare' && (
            <TestComparison
              runs={previousRuns}
              current={result}
            />
          )}
        </div>
      </div>
    </div>
  );
};

// Chain Debugger Component
export const ChainDebugger: React.FC<{ chainId: string }> = ({ chainId }) => {
  const [breakpoints, setBreakpoints] = useState<Set<string>>(new Set());
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [variables, setVariables] = useState<Record<string, any>>({});
  const [isDebugging, setIsDebugging] = useState(false);

  const { data: chain } = useChain(chainId);
  const debugSession = useDebugSession();

  const handleStep = async () => {
    const result = await debugSession.step();
    setCurrentStep(result.step);
    setVariables(result.variables);
  };

  const handleContinue = async () => {
    const result = await debugSession.continue();
    setCurrentStep(result.step);
    setVariables(result.variables);
  };

  return (
    <div className="grid grid-cols-12 gap-4 h-full">
      {/* Chain Flow */}
      <div className="col-span-8">
        <Card title="Chain Execution Flow">
          <ChainFlowDiagram
            chain={chain}
            currentStep={currentStep}
            breakpoints={breakpoints}
            onToggleBreakpoint={(step) => {
              const next = new Set(breakpoints);
              if (next.has(step)) {
                next.delete(step);
              } else {
                next.add(step);
              }
              setBreakpoints(next);
            }}
          />
        </Card>
      </div>

      {/* Debug Controls */}
      <div className="col-span-4 space-y-4">
        {/* Controls */}
        <Card title="Debug Controls">
          <div className="flex gap-2">
            <Button
              icon={<Play />}
              onClick={() => debugSession.start()}
              disabled={isDebugging}
            >
              Start
            </Button>
            <Button
              icon={<StepForward />}
              onClick={handleStep}
              disabled={!isDebugging}
            >
              Step
            </Button>
            <Button
              icon={<FastForward />}
              onClick={handleContinue}
              disabled={!isDebugging}
            >
              Continue
            </Button>
            <Button
              icon={<Square />}
              onClick={() => debugSession.stop()}
              disabled={!isDebugging}
              danger
            >
              Stop
            </Button>
          </div>
        </Card>

        {/* Variables Inspector */}
        <Card title="Variables">
          <VariableInspector
            variables={variables}
            onEdit={(key, value) => {
              debugSession.setVariable(key, value);
            }}
          />
        </Card>

        {/* Call Stack */}
        <Card title="Call Stack">
          <CallStack
            stack={debugSession.callStack}
            currentFrame={debugSession.currentFrame}
            onSelectFrame={(frame) => debugSession.selectFrame(frame)}
          />
        </Card>

        {/* Console Output */}
        <Card title="Console">
          <Console
            logs={debugSession.logs}
            onExecute={(command) => debugSession.evaluate(command)}
          />
        </Card>
      </div>
    </div>
  );
};
```

### 5. Cost Analysis Tools

```typescript
// Cost Analysis Dashboard
export const CostAnalysis: React.FC = () => {
  const [timeRange, setTimeRange] = useState<TimeRange>(last7Days);
  const [groupBy, setGroupBy] = useState<'chain' | 'agent' | 'model'>('chain');

  const { data: costs } = useCostAnalysis({
    timeRange,
    groupBy,
  });

  return (
    <div className="p-6 space-y-6">
      {/* Cost Overview */}
      <div className="grid grid-cols-4 gap-4">
        <CostCard
          title="Total Cost"
          value={`$${costs?.total.toFixed(2)}`}
          period={timeRange.label}
          change={costs?.change}
        />
        <CostCard
          title="LLM Costs"
          value={`$${costs?.llm.toFixed(2)}`}
          percentage={costs?.llmPercentage}
          icon={<Brain />}
        />
        <CostCard
          title="Infrastructure"
          value={`$${costs?.infrastructure.toFixed(2)}`}
          percentage={costs?.infraPercentage}
          icon={<Server />}
        />
        <CostCard
          title="Data Processing"
          value={`$${costs?.dataProcessing.toFixed(2)}`}
          percentage={costs?.dataPercentage}
          icon={<Database />}
        />
      </div>

      {/* Cost Trend Chart */}
      <Card title="Cost Trends">
        <CostTrendChart
          data={costs?.trends}
          groupBy={groupBy}
          stacked
        />
      </Card>

      {/* Cost Breakdown */}
      <div className="grid grid-cols-2 gap-6">
        {/* By Model */}
        <Card title="Cost by Model">
          <ModelCostBreakdown
            data={costs?.byModel}
            showTokens
            showRequests
          />
        </Card>

        {/* By Chain */}
        <Card title="Cost by Chain">
          <ChainCostBreakdown
            data={costs?.byChain}
            showExecutions
            showAvgCost
          />
        </Card>
      </div>

      {/* Cost Optimization Suggestions */}
      <Card
        title="Optimization Opportunities"
        extra={
          <Tag color="green">
            Potential savings: ${costs?.potentialSavings.toFixed(2)}/month
          </Tag>
        }
      >
        <OptimizationSuggestions
          suggestions={costs?.optimizations}
          onApply={(suggestion) => applySuggestion(suggestion)}
        />
      </Card>

      {/* Detailed Cost Table */}
      <Card title="Detailed Cost Breakdown">
        <CostTable
          data={costs?.detailed}
          columns={[
            { key: 'resource', label: 'Resource' },
            { key: 'usage', label: 'Usage' },
            { key: 'rate', label: 'Rate' },
            { key: 'cost', label: 'Cost' },
            { key: 'trend', label: 'Trend' },
          ]}
          expandable
          exportable
        />
      </Card>
    </div>
  );
};

// Token Usage Analyzer
export const TokenAnalyzer: React.FC = () => {
  const { data: usage } = useTokenUsage();

  return (
    <div className="space-y-4">
      {/* Token Metrics */}
      <div className="grid grid-cols-3 gap-4">
        <MetricCard
          title="Total Tokens"
          value={formatNumber(usage?.total || 0)}
          subtitle="This billing period"
        />
        <MetricCard
          title="Avg per Request"
          value={formatNumber(usage?.avgPerRequest || 0)}
          trend={usage?.avgTrend}
        />
        <MetricCard
          title="Token Efficiency"
          value={`${usage?.efficiency || 0}%`}
          subtitle="vs baseline"
        />
      </div>

      {/* Token Distribution */}
      <Card title="Token Distribution">
        <TokenDistributionChart
          data={usage?.distribution}
          categories={['input', 'output', 'system']}
        />
      </Card>

      {/* Token Optimization */}
      <Card title="Token Optimization Opportunities">
        <TokenOptimizationList
          opportunities={usage?.optimizations}
          potentialSavings={usage?.potentialTokenSavings}
        />
      </Card>
    </div>
  );
};
```

### 6. A/B Testing Interface

```typescript
// A/B Testing Dashboard
export const ABTestingDashboard: React.FC = () => {
  const [activeExperiment, setActiveExperiment] = useState<string>();
  const { data: experiments } = useExperiments();

  return (
    <div className="p-6">
      {/* Experiments List */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold">A/B Experiments</h2>
          <Button
            type="primary"
            icon={<Plus />}
            onClick={() => openNewExperimentModal()}
          >
            New Experiment
          </Button>
        </div>

        <ExperimentsList
          experiments={experiments}
          onSelect={setActiveExperiment}
          selected={activeExperiment}
        />
      </div>

      {/* Experiment Detail */}
      {activeExperiment && (
        <ExperimentDetail experimentId={activeExperiment} />
      )}
    </div>
  );
};

// Experiment Detail Component
export const ExperimentDetail: React.FC<{ experimentId: string }> = ({
  experimentId,
}) => {
  const { data: experiment } = useExperiment(experimentId);
  const [selectedMetric, setSelectedMetric] = useState<string>('primary');

  if (!experiment) return <LoadingSkeleton />;

  return (
    <div className="space-y-6">
      {/* Experiment Header */}
      <Card>
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-xl font-semibold mb-2">{experiment.name}</h3>
            <p className="text-gray-600 mb-4">{experiment.description}</p>
            <div className="flex items-center gap-4">
              <StatusBadge status={experiment.status} />
              <span className="text-sm text-gray-500">
                Started {formatDate(experiment.startDate)}
              </span>
              <span className="text-sm text-gray-500">
                {experiment.traffic}% traffic allocation
              </span>
            </div>
          </div>

          <ExperimentControls
            experiment={experiment}
            onStart={() => startExperiment(experimentId)}
            onStop={() => stopExperiment(experimentId)}
            onFinalize={() => finalizeExperiment(experimentId)}
          />
        </div>
      </Card>

      {/* Results Summary */}
      <Card title="Results Summary">
        <div className="grid grid-cols-3 gap-6">
          <div>
            <h4 className="font-medium mb-2">Variant A (Control)</h4>
            <VariantMetrics
              variant={experiment.variants.control}
              metrics={experiment.metrics}
            />
          </div>

          <div>
            <h4 className="font-medium mb-2">Variant B (Treatment)</h4>
            <VariantMetrics
              variant={experiment.variants.treatment}
              metrics={experiment.metrics}
            />
          </div>

          <div>
            <h4 className="font-medium mb-2">Statistical Significance</h4>
            <SignificanceAnalysis
              control={experiment.variants.control}
              treatment={experiment.variants.treatment}
              confidence={experiment.confidence}
            />
          </div>
        </div>
      </Card>

      {/* Metrics Charts */}
      <Card title="Metric Performance">
        <div className="mb-4">
          <Select
            value={selectedMetric}
            onChange={setSelectedMetric}
            options={experiment.metrics.map(m => ({
              value: m.id,
              label: m.name,
            }))}
          />
        </div>

        <MetricComparisonChart
          metric={selectedMetric}
          control={experiment.variants.control}
          treatment={experiment.variants.treatment}
          timeRange={experiment.duration}
        />
      </Card>

      {/* Sample Size Calculator */}
      <Card title="Sample Size Analysis">
        <SampleSizeCalculator
          currentSamples={experiment.samples}
          requiredSamples={experiment.requiredSamples}
          confidence={experiment.confidence}
          mde={experiment.minimumDetectableEffect}
        />
      </Card>
    </div>
  );
};

// Experiment Creator Modal
export const ExperimentCreator: React.FC = () => {
  const [config, setConfig] = useState<ExperimentConfig>({
    name: '',
    description: '',
    hypothesis: '',
    variants: {
      control: { name: 'Control', config: {} },
      treatment: { name: 'Treatment', config: {} },
    },
    metrics: [],
    trafficAllocation: 50,
    duration: 14,
  });

  return (
    <Modal
      title="Create New Experiment"
      width={800}
      footer={[
        <Button key="cancel">Cancel</Button>,
        <Button key="create" type="primary" onClick={() => createExperiment(config)}>
          Create Experiment
        </Button>,
      ]}
    >
      <Form layout="vertical">
        {/* Basic Info */}
        <Form.Item label="Experiment Name" required>
          <Input
            value={config.name}
            onChange={(e) => setConfig({ ...config, name: e.target.value })}
            placeholder="e.g., Improved Carbon Calculation Algorithm"
          />
        </Form.Item>

        <Form.Item label="Hypothesis" required>
          <TextArea
            value={config.hypothesis}
            onChange={(e) => setConfig({ ...config, hypothesis: e.target.value })}
            placeholder="We believe that..."
            rows={3}
          />
        </Form.Item>

        {/* Variants Configuration */}
        <Form.Item label="Variants">
          <VariantConfigurator
            variants={config.variants}
            onChange={(variants) => setConfig({ ...config, variants })}
          />
        </Form.Item>

        {/* Metrics Selection */}
        <Form.Item label="Success Metrics">
          <MetricsSelector
            selected={config.metrics}
            onChange={(metrics) => setConfig({ ...config, metrics })}
          />
        </Form.Item>

        {/* Traffic Allocation */}
        <Form.Item label="Traffic Allocation">
          <Slider
            value={config.trafficAllocation}
            onChange={(value) => setConfig({ ...config, trafficAllocation: value })}
            marks={{
              0: '0%',
              25: '25%',
              50: '50%',
              75: '75%',
              100: '100%',
            }}
          />
        </Form.Item>

        {/* Duration */}
        <Form.Item label="Experiment Duration">
          <Select
            value={config.duration}
            onChange={(value) => setConfig({ ...config, duration: value })}
            options={[
              { value: 7, label: '1 week' },
              { value: 14, label: '2 weeks' },
              { value: 30, label: '1 month' },
              { value: 0, label: 'Until manually stopped' },
            ]}
          />
        </Form.Item>
      </Form>
    </Modal>
  );
};
```

### 7. Dataset Management

```typescript
// Dataset Manager Component
export const DatasetManager: React.FC = () => {
  const [selectedDataset, setSelectedDataset] = useState<string>();
  const { data: datasets } = useDatasets();

  return (
    <div className="flex h-full">
      {/* Dataset List */}
      <div className="w-80 border-r bg-gray-50 p-4">
        <div className="mb-4">
          <Button
            type="primary"
            icon={<Upload />}
            block
            onClick={() => openUploadModal()}
          >
            Upload Dataset
          </Button>
        </div>

        <div className="space-y-2">
          {datasets?.map((dataset) => (
            <DatasetCard
              key={dataset.id}
              dataset={dataset}
              selected={selectedDataset === dataset.id}
              onClick={() => setSelectedDataset(dataset.id)}
            />
          ))}
        </div>
      </div>

      {/* Dataset Detail */}
      <div className="flex-1">
        {selectedDataset ? (
          <DatasetDetail datasetId={selectedDataset} />
        ) : (
          <EmptyState
            title="Select a dataset"
            description="Choose a dataset to view and manage"
          />
        )}
      </div>
    </div>
  );
};

// Dataset Detail View
export const DatasetDetail: React.FC<{ datasetId: string }> = ({ datasetId }) => {
  const { data: dataset } = useDataset(datasetId);
  const [activeTab, setActiveTab] = useState<'overview' | 'data' | 'tests' | 'versions'>('overview');

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-xl font-semibold mb-2">{dataset.name}</h2>
            <p className="text-gray-600">{dataset.description}</p>
            <div className="flex items-center gap-4 mt-2 text-sm">
              <span>{dataset.records} records</span>
              <span>{formatBytes(dataset.size)}</span>
              <span>Updated {formatRelativeTime(dataset.updatedAt)}</span>
            </div>
          </div>

          <DatasetActions
            dataset={dataset}
            onEdit={() => editDataset(dataset)}
            onDelete={() => deleteDataset(dataset.id)}
            onExport={() => exportDataset(dataset.id)}
          />
        </div>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onChange={setActiveTab}>
        <Tab value="overview" label="Overview" />
        <Tab value="data" label="Data" />
        <Tab value="tests" label="Test Results" />
        <Tab value="versions" label="Versions" />
      </Tabs>

      {/* Tab Content */}
      <div className="flex-1 overflow-auto p-4">
        {activeTab === 'overview' && <DatasetOverview dataset={dataset} />}
        {activeTab === 'data' && <DatasetExplorer dataset={dataset} />}
        {activeTab === 'tests' && <DatasetTestResults dataset={dataset} />}
        {activeTab === 'versions' && <DatasetVersions dataset={dataset} />}
      </div>
    </div>
  );
};
```

### 8. Performance & Architecture

```yaml
Performance Requirements:
  - Trace ingestion: 100K traces/second
  - Query latency: <200ms for complex queries
  - UI responsiveness: <100ms for interactions
  - Data retention: 90 days hot, 1 year cold
  - Concurrent users: 1,000+

Architecture Decisions:
  - Write-optimized storage (ClickHouse)
  - Read-optimized caching (Redis)
  - Async processing (Kafka)
  - Distributed tracing (OpenTelemetry)
  - Horizontal scaling

Data Pipeline:
  - Ingestion: OpenTelemetry collectors
  - Processing: Stream processing with Kafka
  - Storage: ClickHouse for traces, PostgreSQL for metadata
  - Query: GraphQL with DataLoader
  - Real-time: WebSocket subscriptions
```

### 9. Timeline & Milestones

```yaml
Q2 2026: Foundation
  Month 1:
    - Trace collection infrastructure
    - Basic trace visualization
    - Simple playground
    - Authentication

  Month 2:
    - Performance metrics
    - Cost analysis
    - Debug tools
    - Dataset management

  Month 3:
    - A/B testing framework
    - Advanced analytics
    - Export capabilities
    - Beta launch

Q3 2026: Enhancement
  - AI-powered insights
  - Anomaly detection
  - Predictive analytics
  - Collaboration features
  - Public launch

Q4 2026: Scale
  - Enterprise features
  - Custom dashboards
  - API access
  - White-label options
  - 10,000+ users target
```