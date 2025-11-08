/**
 * AgentPalette - Searchable agent library component
 *
 * Features:
 * - Searchable agent library with categories
 * - Drag-to-canvas functionality
 * - Agent preview with description and I/O
 * - Recent agents list
 * - Favorites/starred agents
 * - Filter by category, tags, capabilities
 * - Agent usage statistics
 * - Keyboard shortcuts (Cmd+K for quick search)
 */

import React, { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import { nanoid } from 'nanoid';
import {
  Search,
  Star,
  Clock,
  Filter,
  X,
  ChevronDown,
  ChevronRight,
  Database,
  Brain,
  Plug,
  Wrench,
  Package,
  Info,
  TrendingUp,
} from 'lucide-react';

import {
  AgentMetadata,
  AgentCategory,
  AgentLibraryItem,
  AgentSearchOptions,
  DataType,
  WorkflowNode,
  ExecutionStatus,
} from './types';
import { useCanvasStore } from './WorkflowCanvas';

/**
 * Mock agent library data
 */
const AGENT_LIBRARY: AgentLibraryItem[] = [
  // Data Processing
  {
    id: 'csv-processor',
    name: 'CSV Processor',
    description: 'Process and transform CSV files with filtering and mapping',
    category: AgentCategory.DATA_PROCESSING,
    version: '1.0.0',
    icon: '📊',
    color: '#3b82f6',
    tags: ['csv', 'data', 'transform'],
    inputs: [
      {
        id: 'file',
        name: 'CSV File',
        type: DataType.FILE,
        required: true,
        description: 'Input CSV file',
      },
      {
        id: 'delimiter',
        name: 'Delimiter',
        type: DataType.STRING,
        required: false,
        defaultValue: ',',
        description: 'CSV delimiter character',
      },
    ],
    outputs: [
      {
        id: 'data',
        name: 'Processed Data',
        type: DataType.ARRAY,
        required: true,
        description: 'Processed CSV data as array',
      },
    ],
    isFavorite: false,
    usageCount: 45,
  },
  {
    id: 'json-parser',
    name: 'JSON Parser',
    description: 'Parse and validate JSON data with schema validation',
    category: AgentCategory.DATA_PROCESSING,
    version: '1.0.0',
    icon: '🔍',
    color: '#10b981',
    tags: ['json', 'parse', 'validate'],
    inputs: [
      {
        id: 'json',
        name: 'JSON String',
        type: DataType.STRING,
        required: true,
        description: 'JSON string to parse',
      },
    ],
    outputs: [
      {
        id: 'data',
        name: 'Parsed Data',
        type: DataType.OBJECT,
        required: true,
        description: 'Parsed JSON object',
      },
    ],
    isFavorite: true,
    usageCount: 89,
  },
  {
    id: 'data-validator',
    name: 'Data Validator',
    description: 'Validate data against custom rules and schemas',
    category: AgentCategory.DATA_PROCESSING,
    version: '1.0.0',
    icon: '✅',
    color: '#8b5cf6',
    tags: ['validation', 'schema', 'rules'],
    inputs: [
      {
        id: 'data',
        name: 'Input Data',
        type: DataType.ANY,
        required: true,
        description: 'Data to validate',
      },
      {
        id: 'schema',
        name: 'Validation Schema',
        type: DataType.OBJECT,
        required: true,
        description: 'JSON schema for validation',
      },
    ],
    outputs: [
      {
        id: 'valid',
        name: 'Is Valid',
        type: DataType.BOOLEAN,
        required: true,
        description: 'Validation result',
      },
      {
        id: 'errors',
        name: 'Errors',
        type: DataType.ARRAY,
        required: false,
        description: 'Validation errors if any',
      },
    ],
    isFavorite: false,
    usageCount: 34,
  },

  // AI/ML
  {
    id: 'openai-agent',
    name: 'OpenAI Agent',
    description: 'Integration with OpenAI GPT models for text generation',
    category: AgentCategory.AI_ML,
    version: '1.0.0',
    icon: '🤖',
    color: '#ec4899',
    tags: ['openai', 'gpt', 'llm', 'ai'],
    inputs: [
      {
        id: 'prompt',
        name: 'Prompt',
        type: DataType.STRING,
        required: true,
        description: 'Text prompt for the model',
      },
      {
        id: 'model',
        name: 'Model',
        type: DataType.STRING,
        required: false,
        defaultValue: 'gpt-4',
        description: 'OpenAI model to use',
      },
    ],
    outputs: [
      {
        id: 'response',
        name: 'Response',
        type: DataType.STRING,
        required: true,
        description: 'Generated text response',
      },
    ],
    isFavorite: true,
    usageCount: 156,
  },
  {
    id: 'huggingface-agent',
    name: 'HuggingFace Agent',
    description: 'Run HuggingFace models for various ML tasks',
    category: AgentCategory.AI_ML,
    version: '1.0.0',
    icon: '🤗',
    color: '#f59e0b',
    tags: ['huggingface', 'ml', 'transformers'],
    inputs: [
      {
        id: 'input',
        name: 'Input',
        type: DataType.ANY,
        required: true,
        description: 'Model input',
      },
      {
        id: 'model_id',
        name: 'Model ID',
        type: DataType.STRING,
        required: true,
        description: 'HuggingFace model identifier',
      },
    ],
    outputs: [
      {
        id: 'output',
        name: 'Output',
        type: DataType.ANY,
        required: true,
        description: 'Model output',
      },
    ],
    isFavorite: false,
    usageCount: 67,
  },
  {
    id: 'custom-ml-agent',
    name: 'Custom ML Agent',
    description: 'Run custom machine learning models',
    category: AgentCategory.AI_ML,
    version: '1.0.0',
    icon: '🧠',
    color: '#06b6d4',
    tags: ['ml', 'custom', 'model'],
    inputs: [
      {
        id: 'features',
        name: 'Features',
        type: DataType.ARRAY,
        required: true,
        description: 'Input features for prediction',
      },
    ],
    outputs: [
      {
        id: 'prediction',
        name: 'Prediction',
        type: DataType.ANY,
        required: true,
        description: 'Model prediction',
      },
    ],
    isFavorite: false,
    usageCount: 23,
  },

  // Integration
  {
    id: 'api-connector',
    name: 'API Connector',
    description: 'Make HTTP requests to external APIs',
    category: AgentCategory.INTEGRATION,
    version: '1.0.0',
    icon: '🔌',
    color: '#14b8a6',
    tags: ['api', 'http', 'rest'],
    inputs: [
      {
        id: 'url',
        name: 'URL',
        type: DataType.STRING,
        required: true,
        description: 'API endpoint URL',
      },
      {
        id: 'method',
        name: 'Method',
        type: DataType.STRING,
        required: false,
        defaultValue: 'GET',
        description: 'HTTP method',
      },
      {
        id: 'body',
        name: 'Request Body',
        type: DataType.OBJECT,
        required: false,
        description: 'Request body for POST/PUT',
      },
    ],
    outputs: [
      {
        id: 'response',
        name: 'Response',
        type: DataType.OBJECT,
        required: true,
        description: 'API response',
      },
    ],
    isFavorite: true,
    usageCount: 123,
  },
  {
    id: 'database-agent',
    name: 'Database Agent',
    description: 'Query and manipulate database records',
    category: AgentCategory.INTEGRATION,
    version: '1.0.0',
    icon: '💾',
    color: '#6366f1',
    tags: ['database', 'sql', 'query'],
    inputs: [
      {
        id: 'query',
        name: 'SQL Query',
        type: DataType.STRING,
        required: true,
        description: 'SQL query to execute',
      },
      {
        id: 'params',
        name: 'Parameters',
        type: DataType.ARRAY,
        required: false,
        description: 'Query parameters',
      },
    ],
    outputs: [
      {
        id: 'results',
        name: 'Results',
        type: DataType.ARRAY,
        required: true,
        description: 'Query results',
      },
    ],
    isFavorite: false,
    usageCount: 78,
  },
  {
    id: 'filesystem-agent',
    name: 'FileSystem Agent',
    description: 'Read and write files from the filesystem',
    category: AgentCategory.INTEGRATION,
    version: '1.0.0',
    icon: '📁',
    color: '#84cc16',
    tags: ['file', 'filesystem', 'io'],
    inputs: [
      {
        id: 'path',
        name: 'File Path',
        type: DataType.STRING,
        required: true,
        description: 'Path to file',
      },
      {
        id: 'operation',
        name: 'Operation',
        type: DataType.STRING,
        required: true,
        description: 'read or write',
      },
    ],
    outputs: [
      {
        id: 'content',
        name: 'Content',
        type: DataType.STRING,
        required: true,
        description: 'File content',
      },
    ],
    isFavorite: false,
    usageCount: 56,
  },

  // Utilities
  {
    id: 'logger',
    name: 'Logger',
    description: 'Log messages and data for debugging',
    category: AgentCategory.UTILITIES,
    version: '1.0.0',
    icon: '📝',
    color: '#64748b',
    tags: ['log', 'debug', 'monitor'],
    inputs: [
      {
        id: 'message',
        name: 'Message',
        type: DataType.ANY,
        required: true,
        description: 'Message to log',
      },
      {
        id: 'level',
        name: 'Log Level',
        type: DataType.STRING,
        required: false,
        defaultValue: 'info',
        description: 'Log level (info, warn, error)',
      },
    ],
    outputs: [
      {
        id: 'logged',
        name: 'Logged',
        type: DataType.BOOLEAN,
        required: true,
        description: 'Log status',
      },
    ],
    isFavorite: false,
    usageCount: 234,
  },
  {
    id: 'scheduler',
    name: 'Scheduler',
    description: 'Schedule workflow execution at specific times',
    category: AgentCategory.UTILITIES,
    version: '1.0.0',
    icon: '⏰',
    color: '#f97316',
    tags: ['schedule', 'cron', 'timer'],
    inputs: [
      {
        id: 'schedule',
        name: 'Schedule',
        type: DataType.STRING,
        required: true,
        description: 'Cron expression',
      },
    ],
    outputs: [
      {
        id: 'triggered',
        name: 'Triggered',
        type: DataType.BOOLEAN,
        required: true,
        description: 'Trigger status',
      },
    ],
    isFavorite: false,
    usageCount: 45,
  },
  {
    id: 'error-handler',
    name: 'Error Handler',
    description: 'Handle and recover from workflow errors',
    category: AgentCategory.UTILITIES,
    version: '1.0.0',
    icon: '🛡️',
    color: '#ef4444',
    tags: ['error', 'handler', 'recovery'],
    inputs: [
      {
        id: 'error',
        name: 'Error',
        type: DataType.ANY,
        required: true,
        description: 'Error to handle',
      },
      {
        id: 'strategy',
        name: 'Strategy',
        type: DataType.STRING,
        required: false,
        defaultValue: 'retry',
        description: 'Error handling strategy',
      },
    ],
    outputs: [
      {
        id: 'handled',
        name: 'Handled',
        type: DataType.BOOLEAN,
        required: true,
        description: 'Handle status',
      },
    ],
    isFavorite: false,
    usageCount: 67,
  },
];

/**
 * Category icon mapping
 */
const categoryIcons = {
  [AgentCategory.DATA_PROCESSING]: Database,
  [AgentCategory.AI_ML]: Brain,
  [AgentCategory.INTEGRATION]: Plug,
  [AgentCategory.UTILITIES]: Wrench,
  [AgentCategory.CUSTOM]: Package,
};

/**
 * Agent card component
 */
const AgentCard: React.FC<{
  agent: AgentLibraryItem;
  onToggleFavorite: (agentId: string) => void;
  onDragStart: (agent: AgentMetadata) => void;
}> = ({ agent, onToggleFavorite, onDragStart }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className="bg-white border border-gray-200 rounded-lg p-3 hover:shadow-md transition-shadow cursor-move"
      draggable
      onDragStart={(e) => {
        onDragStart(agent);
        e.dataTransfer.effectAllowed = 'move';
      }}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2 flex-1">
          <span className="text-2xl">{agent.icon}</span>
          <div className="flex-1 min-w-0">
            <h3 className="font-semibold text-sm truncate">{agent.name}</h3>
            <p className="text-xs text-gray-500">{agent.category}</p>
          </div>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={() => onToggleFavorite(agent.id)}
            className="p-1 hover:bg-gray-100 rounded"
          >
            <Star
              size={16}
              className={agent.isFavorite ? 'fill-yellow-400 text-yellow-400' : ''}
            />
          </button>
          <button
            onClick={() => setExpanded(!expanded)}
            className="p-1 hover:bg-gray-100 rounded"
          >
            {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </button>
        </div>
      </div>

      <p className="text-xs text-gray-600 mb-2 line-clamp-2">{agent.description}</p>

      {expanded && (
        <div className="mt-3 pt-3 border-t border-gray-200 space-y-2">
          <div>
            <div className="text-xs font-semibold text-gray-700 mb-1">Inputs:</div>
            <div className="space-y-1">
              {agent.inputs.map((input) => (
                <div
                  key={input.id}
                  className="text-xs text-gray-600 flex items-center gap-1"
                >
                  <span className="font-mono text-blue-600">{input.type}</span>
                  <span>{input.name}</span>
                  {input.required && (
                    <span className="text-red-500">*</span>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div>
            <div className="text-xs font-semibold text-gray-700 mb-1">Outputs:</div>
            <div className="space-y-1">
              {agent.outputs.map((output) => (
                <div
                  key={output.id}
                  className="text-xs text-gray-600 flex items-center gap-1"
                >
                  <span className="font-mono text-green-600">{output.type}</span>
                  <span>{output.name}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-3 text-xs text-gray-500">
            <span className="flex items-center gap-1">
              <TrendingUp size={12} />
              {agent.usageCount} uses
            </span>
            <span>v{agent.version}</span>
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * Main AgentPalette component
 */
export const AgentPalette: React.FC = () => {
  const [agents, setAgents] = useState<AgentLibraryItem[]>(AGENT_LIBRARY);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<AgentCategory | null>(null);
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);
  const [showRecent, setShowRecent] = useState(false);
  const [sortBy, setSortBy] = useState<'name' | 'usage' | 'recent'>('name');
  const [collapsedCategories, setCollapsedCategories] = useState<Set<AgentCategory>>(
    new Set()
  );
  const searchInputRef = useRef<HTMLInputElement>(null);

  const { addNode } = useCanvasStore();

  // Filter and sort agents
  const filteredAgents = useMemo(() => {
    let filtered = agents;

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (agent) =>
          agent.name.toLowerCase().includes(query) ||
          agent.description.toLowerCase().includes(query) ||
          agent.tags.some((tag) => tag.toLowerCase().includes(query))
      );
    }

    // Category filter
    if (selectedCategory) {
      filtered = filtered.filter((agent) => agent.category === selectedCategory);
    }

    // Favorites filter
    if (showFavoritesOnly) {
      filtered = filtered.filter((agent) => agent.isFavorite);
    }

    // Sort
    filtered = [...filtered].sort((a, b) => {
      switch (sortBy) {
        case 'usage':
          return b.usageCount - a.usageCount;
        case 'recent':
          return (b.lastUsed?.getTime() || 0) - (a.lastUsed?.getTime() || 0);
        case 'name':
        default:
          return a.name.localeCompare(b.name);
      }
    });

    return filtered;
  }, [agents, searchQuery, selectedCategory, showFavoritesOnly, sortBy]);

  // Group agents by category
  const agentsByCategory = useMemo(() => {
    const grouped = new Map<AgentCategory, AgentLibraryItem[]>();

    filteredAgents.forEach((agent) => {
      const category = agent.category;
      const existing = grouped.get(category) || [];
      existing.push(agent);
      grouped.set(category, existing);
    });

    return grouped;
  }, [filteredAgents]);

  // Toggle favorite
  const handleToggleFavorite = useCallback((agentId: string) => {
    setAgents((prev) =>
      prev.map((agent) =>
        agent.id === agentId
          ? { ...agent, isFavorite: !agent.isFavorite }
          : agent
      )
    );
  }, []);

  // Handle drag start
  const handleDragStart = useCallback((agent: AgentMetadata) => {
    // Store agent data in dataTransfer for drop handling
    if (typeof window !== 'undefined') {
      (window as any).__draggedAgent = agent;
    }
  }, []);

  // Add agent to canvas
  const handleAddAgent = useCallback(
    (agent: AgentMetadata, position = { x: 100, y: 100 }) => {
      const newNode: WorkflowNode = {
        id: `node-${nanoid()}`,
        type: 'custom',
        position,
        data: {
          agent,
          label: agent.name,
          status: ExecutionStatus.IDLE,
          config: {},
          inputs: {},
          outputs: {},
        },
      };

      addNode(newNode);

      // Update usage stats
      setAgents((prev) =>
        prev.map((a) =>
          a.id === agent.id
            ? { ...a, usageCount: a.usageCount + 1, lastUsed: new Date() }
            : a
        )
      );
    },
    [addNode]
  );

  // Toggle category collapse
  const toggleCategory = useCallback((category: AgentCategory) => {
    setCollapsedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
        event.preventDefault();
        searchInputRef.current?.focus();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className="w-80 h-screen bg-gray-50 border-r border-gray-200 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 bg-white">
        <h2 className="text-lg font-bold mb-3">Agent Library</h2>

        {/* Search */}
        <div className="relative mb-3">
          <Search className="absolute left-3 top-2.5 text-gray-400" size={18} />
          <input
            ref={searchInputRef}
            type="text"
            placeholder="Search agents... (Cmd+K)"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-8 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-3 top-2.5 text-gray-400 hover:text-gray-600"
            >
              <X size={18} />
            </button>
          )}
        </div>

        {/* Filters */}
        <div className="flex gap-2 mb-3">
          <button
            onClick={() => setShowFavoritesOnly(!showFavoritesOnly)}
            className={`flex items-center gap-1 px-3 py-1.5 text-xs rounded-lg border ${
              showFavoritesOnly
                ? 'bg-yellow-100 border-yellow-400 text-yellow-700'
                : 'bg-white border-gray-300 text-gray-700'
            }`}
          >
            <Star size={14} />
            Favorites
          </button>

          <button
            onClick={() => setShowRecent(!showRecent)}
            className={`flex items-center gap-1 px-3 py-1.5 text-xs rounded-lg border ${
              showRecent
                ? 'bg-blue-100 border-blue-400 text-blue-700'
                : 'bg-white border-gray-300 text-gray-700'
            }`}
          >
            <Clock size={14} />
            Recent
          </button>
        </div>

        {/* Sort */}
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as any)}
          className="w-full px-3 py-2 text-xs border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="name">Sort by Name</option>
          <option value="usage">Sort by Usage</option>
          <option value="recent">Sort by Recent</option>
        </select>
      </div>

      {/* Agent List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {agentsByCategory.size === 0 ? (
          <div className="text-center text-gray-500 text-sm py-8">
            No agents found
          </div>
        ) : (
          Array.from(agentsByCategory.entries()).map(([category, categoryAgents]) => {
            const Icon = categoryIcons[category];
            const isCollapsed = collapsedCategories.has(category);

            return (
              <div key={category}>
                <button
                  onClick={() => toggleCategory(category)}
                  className="flex items-center justify-between w-full mb-2 px-2 py-1 hover:bg-gray-100 rounded"
                >
                  <div className="flex items-center gap-2">
                    <Icon size={16} className="text-gray-600" />
                    <span className="font-semibold text-sm">{category}</span>
                    <span className="text-xs text-gray-500">
                      ({categoryAgents.length})
                    </span>
                  </div>
                  {isCollapsed ? (
                    <ChevronRight size={16} />
                  ) : (
                    <ChevronDown size={16} />
                  )}
                </button>

                {!isCollapsed && (
                  <div className="space-y-2">
                    {categoryAgents.map((agent) => (
                      <AgentCard
                        key={agent.id}
                        agent={agent}
                        onToggleFavorite={handleToggleFavorite}
                        onDragStart={handleDragStart}
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200 bg-white">
        <div className="text-xs text-gray-500 text-center">
          {filteredAgents.length} agent{filteredAgents.length !== 1 ? 's' : ''}{' '}
          available
        </div>
      </div>
    </div>
  );
};

export default AgentPalette;
