# GreenLang Visual Workflow Builder - Frontend

This is the frontend implementation for the GreenLang Visual Workflow Builder, providing a comprehensive drag-and-drop interface for building, validating, and executing agent-based workflows.

## Features

### 1. Workflow Canvas (WorkflowCanvas.tsx)
- **React Flow Integration**: Custom canvas with drag-and-drop workflow creation
- **Node Management**: Add, remove, update workflow nodes with visual feedback
- **Edge Management**: Connect nodes with intelligent type checking
- **Execution Status**: Real-time visual indicators for node execution states
- **Auto-layout**: Automatic workflow layout using Dagre algorithm
- **Undo/Redo**: Full history management with keyboard shortcuts
- **Export/Import**: Save and load workflows in JSON format
- **Multi-select**: Select multiple nodes for batch operations
- **Alignment Tools**: Align, distribute, and center selected nodes
- **Minimap**: Visual overview of large workflows
- **Pan & Zoom**: Navigate large workflows easily

### 2. Agent Palette (AgentPalette.tsx)
- **Searchable Library**: 12+ pre-built agents across 4 categories
  - Data Processing: CSV Processor, JSON Parser, Data Validator
  - AI/ML: OpenAI Agent, HuggingFace Agent, Custom ML Agent
  - Integration: API Connector, Database Agent, FileSystem Agent
  - Utilities: Logger, Scheduler, Error Handler
- **Drag-to-Canvas**: Intuitive drag-and-drop agent placement
- **Agent Preview**: View inputs, outputs, and documentation
- **Favorites**: Star frequently used agents
- **Category Filtering**: Organize agents by category
- **Search**: Quick search with Cmd+K keyboard shortcut
- **Usage Statistics**: Track agent usage frequency
- **Collapsible Categories**: Organize large agent libraries

### 3. DAG Editor (DAGEditor.tsx)
- **Real-time Validation**: Instant feedback on workflow validity
- **Cycle Detection**: Prevent circular dependencies in workflows
- **Type Checking**: Ensure compatible data types between connections
- **Required Input Validation**: Check all required inputs are connected
- **Execution Path Preview**: Visualize workflow execution order
- **Critical Path Highlighting**: Identify bottlenecks in workflow
- **Parallel Execution Groups**: Detect nodes that can run in parallel
- **Node Configuration Panel**: Configure node settings inline
- **Visual Error Indicators**: Clear error messages with suggested fixes
- **Connection Info**: View workflow statistics and metrics

### 4. Validation Engine (useWorkflowValidation.ts)
- **Cycle Detection**: DFS-based algorithm to detect cycles
- **Type Compatibility**: Validate data type compatibility with auto-conversion warnings
- **Required Field Checking**: Ensure all required inputs are satisfied
- **Orphaned Node Detection**: Find disconnected nodes
- **Port Validation**: Verify all connections use valid ports
- **Connection Preview**: Check if proposed connection is valid before creating

### 5. Layout Engine (layoutEngine.ts)
- **Auto-layout**: Dagre-based hierarchical layout
- **Edge Optimization**: Minimize edge crossings
- **Grid Snapping**: Align nodes to grid
- **Alignment Tools**: Left, right, top, bottom, center alignment
- **Distribution**: Evenly distribute nodes horizontally or vertically
- **Multiple Algorithms**: Network-simplex, tight-tree, longest-path rankers

## Technology Stack

- **React 18.2**: Modern React with hooks and functional components
- **TypeScript 5.3**: Full type safety and IntelliSense
- **React Flow 11.10**: Powerful workflow visualization library
- **Zustand 4.4**: Lightweight state management
- **Dagre 0.8**: Graph layout algorithms
- **Tailwind CSS 3.4**: Utility-first CSS framework
- **Lucide React**: Beautiful icon library
- **Vitest 1.0**: Fast unit testing framework
- **Vite 5.0**: Lightning-fast build tool

## Project Structure

```
greenlang/frontend/
├── src/
│   ├── components/
│   │   └── WorkflowBuilder/
│   │       ├── WorkflowCanvas.tsx      (850 lines) - Main canvas component
│   │       ├── AgentPalette.tsx        (650 lines) - Agent library
│   │       ├── DAGEditor.tsx           (720 lines) - Validation & config
│   │       ├── types.ts                (320 lines) - TypeScript types
│   │       ├── hooks/
│   │       │   └── useWorkflowValidation.ts (450 lines) - Validation logic
│   │       ├── utils/
│   │       │   └── layoutEngine.ts     (350 lines) - Layout algorithms
│   │       └── __tests__/
│   │           ├── WorkflowCanvas.test.tsx (450 lines) - Canvas tests
│   │           └── validation.test.ts  (350 lines) - Validation tests
│   └── test/
│       └── setup.ts                    - Test configuration
├── package.json                        - Dependencies
├── tsconfig.json                       - TypeScript config
├── vite.config.ts                      - Vite config
├── vitest.config.ts                    - Test config
├── tailwind.config.js                  - Tailwind config
├── postcss.config.js                   - PostCSS config
└── README.md                           - This file
```

## Getting Started

### Installation

```bash
cd greenlang/frontend
npm install
```

### Development

```bash
# Start development server
npm run dev

# Run tests
npm run test

# Run tests with coverage
npm run test:coverage

# Run tests in UI mode
npm run test:ui

# Type checking
npm run type-check

# Linting
npm run lint

# Build for production
npm run build

# Preview production build
npm run preview
```

## Usage

### Basic Workflow Creation

1. **Add Agents**: Drag agents from the palette onto the canvas
2. **Connect Nodes**: Click and drag from output port to input port
3. **Configure Nodes**: Click on a node to open the configuration panel
4. **Validate**: Check the DAG Editor panel for validation errors
5. **Execute**: Click the play button to run the workflow
6. **Export**: Save your workflow as JSON

### Keyboard Shortcuts

- **Cmd+K**: Quick search agents
- **Cmd+Z**: Undo
- **Cmd+Y / Cmd+Shift+Z**: Redo
- **Cmd+S**: Export workflow
- **Cmd+O**: Import workflow
- **Delete / Backspace**: Delete selected nodes

### State Management

The application uses Zustand for state management with the following features:

- **Centralized Store**: Single source of truth for workflow state
- **Immer Integration**: Immutable state updates
- **History Management**: Undo/redo with past/future stacks
- **Validation State**: Real-time validation errors
- **Execution State**: Track workflow execution progress

## Testing

The project includes comprehensive test coverage (>90%):

### Test Files

1. **WorkflowCanvas.test.tsx**
   - Canvas rendering
   - Node and edge management
   - Undo/redo functionality
   - Export/import workflows
   - State management
   - Selection and alignment

2. **validation.test.ts**
   - Cycle detection algorithms
   - Type compatibility checking
   - Required field validation
   - Orphaned node detection
   - Complex workflow scenarios
   - Connection validation

### Running Tests

```bash
# Run all tests
npm run test

# Watch mode
npm run test -- --watch

# Coverage report
npm run test:coverage

# UI mode
npm run test:ui
```

## Type Definitions

The project includes comprehensive TypeScript types:

- **WorkflowNode**: Represents a workflow node with agent data
- **WorkflowEdge**: Represents a connection between nodes
- **AgentMetadata**: Describes agent capabilities and configuration
- **ValidationError**: Structured validation error with severity
- **ExecutionStatus**: Node execution state (idle, running, success, error, etc.)
- **CanvasState**: Complete canvas state with actions

## Validation Rules

The workflow builder enforces the following rules:

1. **No Cycles**: Workflows must be directed acyclic graphs (DAGs)
2. **Type Compatibility**: Connections must have compatible data types
3. **Required Inputs**: All required inputs must be connected or configured
4. **Valid Ports**: Connections must use valid input/output ports
5. **No Duplicates**: Prevent duplicate connections between same ports

## Layout Algorithms

The layout engine supports multiple algorithms:

- **Network Simplex**: Default, balanced layouts
- **Tight Tree**: Compact layouts for tree structures
- **Longest Path**: Emphasizes critical path

## Architecture Decisions

### Why React Flow?
- Production-ready workflow visualization
- Extensive customization options
- Great performance with large graphs
- Active community and maintenance

### Why Zustand?
- Minimal boilerplate compared to Redux
- Excellent TypeScript support
- No need for providers/context
- Great DevTools integration

### Why Dagre?
- Industry-standard graph layout algorithm
- Handles complex DAGs efficiently
- Multiple layout strategies
- Well-tested and reliable

### Why Vitest?
- Faster than Jest
- Native ESM support
- Compatible with Vite
- Better error messages

## Performance Considerations

- **React.memo**: Custom nodes are memoized
- **Lazy Validation**: Validation runs only when needed
- **Debounced Updates**: State updates are batched
- **Virtual Scrolling**: Large agent lists are virtualized
- **Code Splitting**: Components are lazy-loaded

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

When contributing to the workflow builder:

1. Follow TypeScript strict mode
2. Add tests for new features
3. Update documentation
4. Use semantic commit messages
5. Ensure >90% test coverage

## License

Part of the GreenLang project. See main repository for license information.

## Support

For issues and questions:
- GitHub Issues: [GreenLang Repository]
- Documentation: See main project docs
- Examples: Check `/examples` directory
