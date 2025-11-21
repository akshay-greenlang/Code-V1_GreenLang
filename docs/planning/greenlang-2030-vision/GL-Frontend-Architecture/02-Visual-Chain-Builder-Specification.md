# GreenLang Visual Chain Builder Specification
## No-Code Platform for Climate Intelligence Workflows

### 1. Technical Architecture

#### Core Stack
```yaml
Framework: React 18 with TypeScript
Canvas: React Flow (node-based editor)
State: Zustand + Immer (immutable updates)
Real-time: Socket.io (collaboration)
Testing: Playwright (E2E) + React Testing Library
Build: Vite 5.0
Styling: Tailwind CSS + Radix UI
Backend: Node.js + GraphQL + PostgreSQL
```

#### Component Architecture
```typescript
// Core Visual Builder Architecture
interface VisualBuilderArchitecture {
  canvas: {
    engine: 'React Flow';
    features: ['drag-drop', 'zoom', 'pan', 'minimap', 'controls'];
    performance: 'WebGL rendering for 1000+ nodes';
  };

  nodes: {
    types: ['agent', 'operator', 'data', 'control', 'output'];
    customization: 'Full theming and styling';
    validation: 'Real-time type checking';
  };

  edges: {
    types: ['data-flow', 'control-flow', 'conditional'];
    animation: 'Flow visualization';
    validation: 'Connection compatibility checking';
  };

  collaboration: {
    cursors: 'Live multi-user cursors';
    changes: 'Real-time sync via CRDT';
    comments: 'Inline commenting system';
  };
}
```

### 2. Node System Architecture

```typescript
// Node Type Definitions
export interface NodeType {
  id: string;
  type: 'agent' | 'operator' | 'data' | 'control' | 'output';
  category: string;
  data: {
    label: string;
    description: string;
    icon: string;
    config: Record<string, any>;
    inputs: PortDefinition[];
    outputs: PortDefinition[];
    validation: ValidationRule[];
  };
  position: { x: number; y: number };
  selected: boolean;
}

// Agent Node Component
export const AgentNode: React.FC<NodeComponentProps> = ({ data, selected }) => {
  const [config, setConfig] = useState(data.config);
  const [isConfiguring, setIsConfiguring] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  return (
    <div className={`
      agent-node rounded-lg border-2 bg-white shadow-lg
      ${selected ? 'border-green-500' : 'border-gray-300'}
      ${validationErrors.length > 0 ? 'border-red-500' : ''}
    `}>
      {/* Node Header */}
      <div className="flex items-center gap-2 p-3 border-b bg-gradient-to-r from-green-50 to-emerald-50">
        <Icon name={data.icon} className="w-5 h-5 text-green-600" />
        <span className="font-semibold text-sm">{data.label}</span>
        <button
          onClick={() => setIsConfiguring(true)}
          className="ml-auto p-1 hover:bg-gray-100 rounded"
        >
          <Settings className="w-4 h-4" />
        </button>
      </div>

      {/* Input Ports */}
      <div className="p-2">
        {data.inputs.map((input, index) => (
          <Handle
            key={input.id}
            type="target"
            position={Position.Left}
            id={input.id}
            style={{ top: `${30 + index * 25}px` }}
            className="w-3 h-3 bg-blue-500"
          >
            <Tooltip content={`${input.label}: ${input.type}`}>
              <div className="port-label text-xs">{input.label}</div>
            </Tooltip>
          </Handle>
        ))}
      </div>

      {/* Configuration Preview */}
      <div className="px-3 pb-3">
        <div className="text-xs text-gray-600">
          {Object.entries(config).slice(0, 3).map(([key, value]) => (
            <div key={key} className="truncate">
              {key}: {String(value)}
            </div>
          ))}
        </div>
      </div>

      {/* Output Ports */}
      <div className="p-2">
        {data.outputs.map((output, index) => (
          <Handle
            key={output.id}
            type="source"
            position={Position.Right}
            id={output.id}
            style={{ top: `${30 + index * 25}px` }}
            className="w-3 h-3 bg-green-500"
          >
            <Tooltip content={`${output.label}: ${output.type}`}>
              <div className="port-label text-xs">{output.label}</div>
            </Tooltip>
          </Handle>
        ))}
      </div>

      {/* Validation Errors */}
      {validationErrors.length > 0 && (
        <div className="px-3 pb-2">
          {validationErrors.map((error, i) => (
            <div key={i} className="text-xs text-red-600">
              {error}
            </div>
          ))}
        </div>
      )}

      {/* Configuration Modal */}
      {isConfiguring && (
        <NodeConfigModal
          node={data}
          config={config}
          onSave={(newConfig) => {
            setConfig(newConfig);
            setIsConfiguring(false);
          }}
          onClose={() => setIsConfiguring(false)}
        />
      )}
    </div>
  );
};
```

### 3. Visual Canvas Implementation

```typescript
// Main Visual Builder Canvas
export const VisualChainBuilder: React.FC = () => {
  const [nodes, setNodes] = useNodesState([]);
  const [edges, setEdges] = useEdgesState([]);
  const [selectedNodes, setSelectedNodes] = useState<string[]>([]);
  const [isTestMode, setIsTestMode] = useState(false);
  const [executionTrace, setExecutionTrace] = useState<ExecutionTrace | null>(null);

  // Drag and drop handlers
  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();

    const type = event.dataTransfer.getData('nodeType');
    const position = reactFlowInstance.project({
      x: event.clientX,
      y: event.clientY,
    });

    const newNode = createNode(type, position);
    setNodes((nds) => nds.concat(newNode));
  }, [reactFlowInstance]);

  // Connection validation
  const isValidConnection = useCallback((connection: Connection) => {
    const sourceNode = nodes.find(n => n.id === connection.source);
    const targetNode = nodes.find(n => n.id === connection.target);

    if (!sourceNode || !targetNode) return false;

    const sourcePort = sourceNode.data.outputs.find(p => p.id === connection.sourceHandle);
    const targetPort = targetNode.data.inputs.find(p => p.id === connection.targetHandle);

    return isCompatibleTypes(sourcePort?.type, targetPort?.type);
  }, [nodes]);

  // Export to GCEL code
  const exportToCode = useCallback(() => {
    const gcelCode = generateGCELFromGraph(nodes, edges);
    return gcelCode;
  }, [nodes, edges]);

  // Test execution
  const executeTest = useCallback(async () => {
    setIsTestMode(true);
    const trace = await testChainExecution(nodes, edges);
    setExecutionTrace(trace);
  }, [nodes, edges]);

  return (
    <div className="h-screen flex">
      {/* Node Palette */}
      <NodePalette className="w-64 border-r bg-gray-50" />

      {/* Main Canvas */}
      <div className="flex-1 relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onDragOver={onDragOver}
          onDrop={onDrop}
          isValidConnection={isValidConnection}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
        >
          <Background variant="dots" gap={20} size={1} />
          <Controls />
          <MiniMap
            nodeColor={(node) => {
              if (node.type === 'agent') return '#10b981';
              if (node.type === 'operator') return '#3b82f6';
              if (node.type === 'data') return '#f59e0b';
              return '#6b7280';
            }}
          />

          {/* Custom Controls */}
          <Panel position="top-right" className="flex gap-2">
            <button
              onClick={executeTest}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            >
              {isTestMode ? 'Testing...' : 'Test Chain'}
            </button>
            <button
              onClick={() => {
                const code = exportToCode();
                navigator.clipboard.writeText(code);
              }}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Export to Code
            </button>
            <button
              onClick={() => saveWorkflow(nodes, edges)}
              className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
            >
              Save Workflow
            </button>
          </Panel>

          {/* Execution Trace Overlay */}
          {executionTrace && (
            <ExecutionTraceOverlay
              trace={executionTrace}
              onClose={() => setExecutionTrace(null)}
            />
          )}
        </ReactFlow>
      </div>

      {/* Properties Panel */}
      <PropertiesPanel
        selectedNodes={selectedNodes}
        nodes={nodes}
        onUpdateNode={(nodeId, data) => {
          setNodes((nds) =>
            nds.map((node) =>
              node.id === nodeId ? { ...node, data } : node
            )
          );
        }}
      />
    </div>
  );
};
```

### 4. Node Palette Component

```typescript
// Draggable Node Palette
export const NodePalette: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const nodeCategories = {
    'Data Intake': [
      { type: 'csv_reader', label: 'CSV Reader', icon: 'file-csv' },
      { type: 'api_connector', label: 'API Connector', icon: 'api' },
      { type: 'database', label: 'Database', icon: 'database' },
      { type: 'excel_reader', label: 'Excel Reader', icon: 'file-excel' },
    ],
    'Processing': [
      { type: 'validator', label: 'Validator', icon: 'check-circle' },
      { type: 'transformer', label: 'Transformer', icon: 'transform' },
      { type: 'aggregator', label: 'Aggregator', icon: 'sum' },
      { type: 'filter', label: 'Filter', icon: 'filter' },
    ],
    'Climate Agents': [
      { type: 'carbon_calculator', label: 'Carbon Calculator', icon: 'calculator' },
      { type: 'scope3_analyzer', label: 'Scope 3 Analyzer', icon: 'analytics' },
      { type: 'csrd_reporter', label: 'CSRD Reporter', icon: 'report' },
      { type: 'cbam_processor', label: 'CBAM Processor', icon: 'compliance' },
    ],
    'Control Flow': [
      { type: 'conditional', label: 'Conditional', icon: 'branch' },
      { type: 'loop', label: 'Loop', icon: 'repeat' },
      { type: 'parallel', label: 'Parallel', icon: 'parallel' },
      { type: 'sequence', label: 'Sequence', icon: 'arrow-right' },
    ],
    'Output': [
      { type: 'pdf_generator', label: 'PDF Generator', icon: 'file-pdf' },
      { type: 'excel_export', label: 'Excel Export', icon: 'download' },
      { type: 'api_webhook', label: 'Webhook', icon: 'webhook' },
      { type: 'email_sender', label: 'Email Sender', icon: 'mail' },
    ],
  };

  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData('nodeType', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="p-4">
      <h3 className="font-semibold mb-4">Node Library</h3>

      {/* Search */}
      <div className="mb-4">
        <input
          type="text"
          placeholder="Search nodes..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full px-3 py-2 border rounded-lg"
        />
      </div>

      {/* Categories */}
      <div className="space-y-2">
        {Object.entries(nodeCategories).map(([category, nodes]) => (
          <Collapsible
            key={category}
            open={selectedCategory === category || searchQuery.length > 0}
            onOpenChange={(open) => setSelectedCategory(open ? category : null)}
          >
            <CollapsibleTrigger className="flex items-center justify-between w-full p-2 hover:bg-gray-100 rounded">
              <span className="font-medium">{category}</span>
              <ChevronDown className="w-4 h-4" />
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="grid grid-cols-2 gap-2 mt-2">
                {nodes
                  .filter(node =>
                    node.label.toLowerCase().includes(searchQuery.toLowerCase())
                  )
                  .map((node) => (
                    <div
                      key={node.type}
                      draggable
                      onDragStart={(e) => onDragStart(e, node.type)}
                      className="p-3 bg-white border rounded-lg cursor-move hover:shadow-md transition-shadow"
                    >
                      <Icon name={node.icon} className="w-5 h-5 mb-1" />
                      <div className="text-xs font-medium">{node.label}</div>
                    </div>
                  ))}
              </div>
            </CollapsibleContent>
          </Collapsible>
        ))}
      </div>
    </div>
  );
};
```

### 5. Real-time Collaboration Features

```typescript
// Collaboration System
export const CollaborationProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [collaborators, setCollaborators] = useState<Collaborator[]>([]);
  const [cursors, setCursors] = useState<Record<string, CursorPosition>>({});
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    // Initialize WebSocket connection
    socketRef.current = io(process.env.NEXT_PUBLIC_COLLAB_SERVER, {
      auth: {
        token: getAuthToken(),
        workflowId: getCurrentWorkflowId(),
      },
    });

    // Handle collaborator events
    socketRef.current.on('collaborator:joined', (collaborator: Collaborator) => {
      setCollaborators((prev) => [...prev, collaborator]);
      showNotification(`${collaborator.name} joined the session`);
    });

    socketRef.current.on('collaborator:left', (collaboratorId: string) => {
      setCollaborators((prev) => prev.filter(c => c.id !== collaboratorId));
      setCursors((prev) => {
        const { [collaboratorId]: _, ...rest } = prev;
        return rest;
      });
    });

    // Handle cursor movements
    socketRef.current.on('cursor:move', ({ userId, position }: CursorEvent) => {
      setCursors((prev) => ({
        ...prev,
        [userId]: position,
      }));
    });

    // Handle workflow changes
    socketRef.current.on('workflow:change', (change: WorkflowChange) => {
      applyWorkflowChange(change);
    });

    return () => {
      socketRef.current?.disconnect();
    };
  }, []);

  const broadcastChange = useCallback((change: WorkflowChange) => {
    socketRef.current?.emit('workflow:change', change);
  }, []);

  const broadcastCursor = useCallback((position: CursorPosition) => {
    socketRef.current?.emit('cursor:move', position);
  }, []);

  return (
    <CollaborationContext.Provider
      value={{
        collaborators,
        cursors,
        broadcastChange,
        broadcastCursor,
      }}
    >
      {children}
      <CollaboratorAvatars collaborators={collaborators} />
      <CollaboratorCursors cursors={cursors} />
    </CollaborationContext.Provider>
  );
};

// Collaborator Cursors Component
export const CollaboratorCursors: React.FC<{ cursors: Record<string, CursorPosition> }> = ({
  cursors,
}) => {
  return (
    <>
      {Object.entries(cursors).map(([userId, position]) => (
        <div
          key={userId}
          className="absolute pointer-events-none z-50 transition-all duration-100"
          style={{
            left: `${position.x}px`,
            top: `${position.y}px`,
          }}
        >
          <svg width="24" height="24" viewBox="0 0 24 24">
            <path
              d="M5.65 5.53l9.22 12.97H9.78L7.1 24l-1.45-5.5L0 17.07z"
              fill={getCollaboratorColor(userId)}
              stroke="#fff"
              strokeWidth="1"
            />
          </svg>
          <div className="ml-2 -mt-1 px-2 py-1 bg-black text-white text-xs rounded">
            {getCollaboratorName(userId)}
          </div>
        </div>
      ))}
    </>
  );
};
```

### 6. Testing and Debugging Interface

```typescript
// Test Execution Panel
export const TestExecutionPanel: React.FC<{
  nodes: Node[];
  edges: Edge[];
}> = ({ nodes, edges }) => {
  const [testData, setTestData] = useState<any>({});
  const [executionResult, setExecutionResult] = useState<ExecutionResult | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionStep, setExecutionStep] = useState<number>(0);
  const [breakpoints, setBreakpoints] = useState<Set<string>>(new Set());

  const executeStep = async () => {
    const currentNode = getExecutionOrder(nodes, edges)[executionStep];
    const result = await executeNode(currentNode, testData);

    setExecutionResult((prev) => ({
      ...prev,
      steps: [...(prev?.steps || []), {
        nodeId: currentNode.id,
        input: testData,
        output: result.output,
        duration: result.duration,
        status: result.status,
      }],
    }));

    if (breakpoints.has(currentNode.id)) {
      setIsExecuting(false);
      showDebugInfo(currentNode, result);
    } else if (executionStep < nodes.length - 1) {
      setExecutionStep((prev) => prev + 1);
    } else {
      setIsExecuting(false);
      showExecutionComplete(executionResult);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Test Data Input */}
      <div className="p-4 border-b">
        <h3 className="font-semibold mb-2">Test Data</h3>
        <JsonEditor
          value={testData}
          onChange={setTestData}
          className="h-40"
        />
      </div>

      {/* Execution Controls */}
      <div className="p-4 border-b flex gap-2">
        <button
          onClick={() => {
            setIsExecuting(true);
            setExecutionStep(0);
            executeStep();
          }}
          disabled={isExecuting}
          className="px-4 py-2 bg-green-600 text-white rounded"
        >
          <Play className="w-4 h-4 inline mr-2" />
          Run Test
        </button>

        <button
          onClick={() => executeStep()}
          disabled={!isExecuting}
          className="px-4 py-2 bg-blue-600 text-white rounded"
        >
          <StepForward className="w-4 h-4 inline mr-2" />
          Step
        </button>

        <button
          onClick={() => setIsExecuting(false)}
          disabled={!isExecuting}
          className="px-4 py-2 bg-red-600 text-white rounded"
        >
          <Square className="w-4 h-4 inline mr-2" />
          Stop
        </button>
      </div>

      {/* Execution Trace */}
      <div className="flex-1 overflow-auto p-4">
        <h3 className="font-semibold mb-2">Execution Trace</h3>
        {executionResult?.steps.map((step, index) => (
          <ExecutionStepCard
            key={index}
            step={step}
            isActive={index === executionStep}
            onToggleBreakpoint={(nodeId) => {
              setBreakpoints((prev) => {
                const next = new Set(prev);
                if (next.has(nodeId)) {
                  next.delete(nodeId);
                } else {
                  next.add(nodeId);
                }
                return next;
              });
            }}
          />
        ))}
      </div>

      {/* Debug Console */}
      <div className="p-4 border-t h-48 overflow-auto bg-gray-900 text-green-400 font-mono text-sm">
        {executionResult?.logs.map((log, i) => (
          <div key={i}>
            <span className="text-gray-500">[{log.timestamp}]</span> {log.message}
          </div>
        ))}
      </div>
    </div>
  );
};
```

### 7. Export to Code Functionality

```typescript
// Code Generation Engine
export class GCELCodeGenerator {
  private nodes: Node[];
  private edges: Edge[];
  private imports: Set<string> = new Set();

  constructor(nodes: Node[], edges: Edge[]) {
    this.nodes = nodes;
    this.edges = edges;
  }

  generateCode(): string {
    const sortedNodes = this.topologicalSort();
    const chainDefinition = this.buildChainDefinition(sortedNodes);
    const imports = this.generateImports();
    const configuration = this.generateConfiguration(sortedNodes);

    return `
${imports}

# GreenLang Climate Intelligence Chain
# Generated from Visual Builder

${configuration}

# Chain Definition
chain = ${chainDefinition}

# Execute Chain
async def run_chain(input_data):
    """Execute the climate intelligence chain"""
    try:
        result = await chain.run(input_data)
        return result
    except Exception as e:
        print(f"Chain execution failed: {e}")
        raise

# Example Usage
if __name__ == "__main__":
    sample_data = {
        # Add your test data here
    }

    result = asyncio.run(run_chain(sample_data))
    print(json.dumps(result, indent=2))
`;
  }

  private buildChainDefinition(nodes: Node[]): string {
    const nodeMap = new Map(nodes.map(n => [n.id, n]));

    // Build GCEL expression
    const expressions: string[] = [];

    for (const node of nodes) {
      const nodeVar = this.getNodeVariable(node);
      const nodeConfig = this.getNodeConfig(node);

      switch (node.type) {
        case 'agent':
          expressions.push(`${nodeVar} = ${node.data.agentType}(${nodeConfig})`);
          this.imports.add(`from greenlang.agents import ${node.data.agentType}`);
          break;

        case 'operator':
          const operator = this.getOperatorExpression(node);
          expressions.push(`${nodeVar} = ${operator}`);
          break;

        case 'control':
          const control = this.getControlExpression(node);
          expressions.push(`${nodeVar} = ${control}`);
          break;
      }
    }

    // Build chain expression using GCEL operators
    const chainExpression = this.buildGCELChain(nodes);

    return `(
    ${expressions.join('\n    ')}

    # Chain composition
    ${chainExpression}
)`;
  }

  private buildGCELChain(nodes: Node[]): string {
    // Analyze graph structure and build GCEL expression
    const startNodes = this.findStartNodes();
    const paths = this.findAllPaths(startNodes);

    if (paths.length === 1) {
      // Sequential chain
      return paths[0].map(n => this.getNodeVariable(n)).join(' >> ');
    } else {
      // Parallel chains
      const parallelChains = paths.map(path =>
        `(${path.map(n => this.getNodeVariable(n)).join(' >> ')})`
      );
      return `GCEL.parallel(\n        ${parallelChains.join(',\n        ')}\n    )`;
    }
  }

  private generateImports(): string {
    return [
      'import asyncio',
      'import json',
      'from greenlang import GCEL',
      ...Array.from(this.imports),
    ].join('\n');
  }

  private topologicalSort(): Node[] {
    // Kahn's algorithm for topological sorting
    const sorted: Node[] = [];
    const inDegree = new Map<string, number>();
    const adjList = new Map<string, string[]>();

    // Initialize
    this.nodes.forEach(node => {
      inDegree.set(node.id, 0);
      adjList.set(node.id, []);
    });

    // Build adjacency list and in-degree map
    this.edges.forEach(edge => {
      adjList.get(edge.source)?.push(edge.target);
      inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
    });

    // Find nodes with no incoming edges
    const queue = this.nodes.filter(node => inDegree.get(node.id) === 0);

    while (queue.length > 0) {
      const node = queue.shift()!;
      sorted.push(node);

      adjList.get(node.id)?.forEach(neighbor => {
        const degree = (inDegree.get(neighbor) || 0) - 1;
        inDegree.set(neighbor, degree);

        if (degree === 0) {
          const neighborNode = this.nodes.find(n => n.id === neighbor);
          if (neighborNode) queue.push(neighborNode);
        }
      });
    }

    return sorted;
  }
}

// Export Dialog Component
export const ExportCodeDialog: React.FC<{
  nodes: Node[];
  edges: Edge[];
  onClose: () => void;
}> = ({ nodes, edges, onClose }) => {
  const [code, setCode] = useState('');
  const [language, setLanguage] = useState<'python' | 'typescript'>('python');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const generator = new GCELCodeGenerator(nodes, edges);
    setCode(generator.generateCode());
  }, [nodes, edges, language]);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `greenlang_chain.${language === 'python' ? 'py' : 'ts'}`;
    a.click();
  };

  return (
    <Dialog open onOpenChange={onClose}>
      <DialogContent className="max-w-4xl h-[80vh]">
        <DialogHeader>
          <DialogTitle>Export Chain to Code</DialogTitle>
          <div className="flex gap-2 mt-4">
            <Select value={language} onValueChange={setLanguage}>
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="python">Python</SelectItem>
                <SelectItem value="typescript">TypeScript</SelectItem>
              </SelectContent>
            </Select>

            <button
              onClick={handleCopy}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              {copied ? 'Copied!' : 'Copy Code'}
            </button>

            <button
              onClick={handleDownload}
              className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            >
              Download
            </button>
          </div>
        </DialogHeader>

        <div className="flex-1 overflow-auto mt-4">
          <SyntaxHighlighter
            language={language}
            style={atomDark}
            showLineNumbers
            customStyle={{ height: '100%' }}
          >
            {code}
          </SyntaxHighlighter>
        </div>
      </DialogContent>
    </Dialog>
  );
};
```

### 8. Performance Optimization

```typescript
// Performance optimizations for large workflows
export const OptimizedCanvas = {
  // Virtual rendering for 1000+ nodes
  virtualRendering: {
    enabled: true,
    viewportBuffer: 2, // Render 2x viewport size
    throttleMs: 16, // 60 FPS max
  },

  // WebGL acceleration
  webglConfig: {
    enabled: true,
    antialias: true,
    preserveDrawingBuffer: false,
  },

  // Optimization strategies
  optimizations: {
    // Batch updates
    batchNodeUpdates: true,
    batchEdgeUpdates: true,

    // Debounce expensive operations
    debounceValidation: 300,
    debounceAutoSave: 1000,

    // Lazy loading
    lazyLoadNodeDetails: true,
    lazyLoadProperties: true,

    // Memory management
    maxUndoStack: 50,
    cleanupInterval: 60000, // 1 minute
  },

  // Performance monitoring
  monitoring: {
    trackFPS: true,
    trackMemory: true,
    trackRenderTime: true,
    alertThreshold: {
      fps: 30,
      memory: 500, // MB
      renderTime: 100, // ms
    },
  },
};
```

### 9. Accessibility Features

```typescript
// Accessibility implementation
export const AccessibleVisualBuilder = {
  // Keyboard navigation
  keyboardShortcuts: {
    'Tab': 'Navigate between nodes',
    'Enter': 'Edit selected node',
    'Space': 'Toggle node selection',
    'Delete': 'Delete selected nodes',
    'Ctrl+C': 'Copy nodes',
    'Ctrl+V': 'Paste nodes',
    'Ctrl+Z': 'Undo',
    'Ctrl+Y': 'Redo',
    'Ctrl+A': 'Select all',
    'Escape': 'Deselect all',
  },

  // Screen reader support
  ariaLabels: {
    canvas: 'Visual workflow builder canvas',
    node: 'Workflow node: {label}',
    edge: 'Connection from {source} to {target}',
    palette: 'Node library palette',
    properties: 'Node properties panel',
  },

  // Focus management
  focusManagement: {
    trapFocus: true,
    restoreFocus: true,
    focusIndicator: 'ring-2 ring-blue-500',
  },

  // High contrast mode
  highContrastMode: {
    enabled: false,
    nodeOutline: '3px solid',
    edgeWidth: 3,
    fontSize: 16,
  },
};
```

### 10. Deployment & Timeline

```yaml
Development Timeline:
  Q2 2026:
    Month 1:
      - React Flow integration
      - Basic node system
      - Drag-and-drop implementation
      - Canvas controls

    Month 2:
      - Node configuration panels
      - Connection validation
      - GCEL code generation
      - Test execution framework

    Month 3:
      - Collaboration features
      - Export functionality
      - Performance optimization
      - Beta release

  Q3 2026:
    - Advanced debugging tools
    - Template library
    - AI-assisted building
    - GA release

Technical Specifications:
  Performance:
    - Support 1000+ nodes
    - 60 FPS rendering
    - <100ms interaction response
    - <3s load time

  Browser Support:
    - Chrome 90+
    - Firefox 88+
    - Safari 14+
    - Edge 90+

  Accessibility:
    - WCAG 2.1 AA compliant
    - Full keyboard navigation
    - Screen reader support
    - High contrast mode
```