"""
GreenLang Framework - Advanced Capabilities Module

Enhanced AI Agent capabilities based on global best practices from:
- US Frameworks: LangChain, LangGraph, AutoGPT, CrewAI, OpenAI Agents SDK
- Chinese Frameworks: Qwen-Agent, AgentScope, ModelScope
- Enterprise Standards: ISO 42001, NIST AI RMF, EU AI Act

Modules:
- mcp_protocol: Model Context Protocol for universal tool interface
- state_machine: LangGraph-style graph-based workflow orchestration
- memory_rag: RAG-enabled semantic memory management
- guardrails: Enterprise AI guardrails and safety controls
- a2a_protocol: Agent-to-Agent collaboration protocol
"""

from .mcp_protocol import (
    MCPVersion,
    ToolCategory,
    SecurityLevel,
    ExecutionMode,
    ToolParameter,
    ToolDefinition,
    ToolCallRequest,
    ToolCallResponse,
    MCPTool,
    MCPToolRegistry,
    mcp_tool,
    GREENLANG_MCP_REGISTRY,
)

from .state_machine import (
    NodeStatus,
    EdgeType,
    NodeResult,
    WorkflowState,
    GraphNode,
    GraphEdge,
    ConditionalRouter,
    WorkflowGraph,
    FunctionNode,
    BranchNode,
    ToolNode,
    WorkflowBuilder,
)

from .memory_rag import (
    MemoryType,
    MemoryStatus,
    MemoryEntry,
    RetrievalResult,
    EmbeddingProvider,
    SimpleEmbedding,
    VectorIndex,
    SimpleVectorIndex,
    MemoryStore,
    GREENLANG_MEMORY,
    remember,
    with_context,
)

from .guardrails import (
    GuardrailType,
    ViolationSeverity,
    ActionType,
    GuardrailViolation,
    GuardrailResult,
    Guardrail,
    PromptInjectionGuardrail,
    DataLeakageGuardrail,
    ActionGate,
    PhysicalConstraint,
    SafetyEnvelopeGuardrail,
    Policy,
    PolicyEnforcementGuardrail,
    GuardrailOrchestrator,
    create_default_orchestrator,
    GREENLANG_GUARDRAILS,
)

from .a2a_protocol import (
    MessageType,
    AgentRole,
    TaskStatus,
    AgentCard,
    A2AMessage,
    DelegatedTask,
    A2AAgent,
    A2ARouter,
    CrewTask,
    AgentCrew,
    GREENLANG_A2A_ROUTER,
)


__all__ = [
    # MCP Protocol
    "MCPVersion",
    "ToolCategory",
    "SecurityLevel",
    "ExecutionMode",
    "ToolParameter",
    "ToolDefinition",
    "ToolCallRequest",
    "ToolCallResponse",
    "MCPTool",
    "MCPToolRegistry",
    "mcp_tool",
    "GREENLANG_MCP_REGISTRY",

    # State Machine
    "NodeStatus",
    "EdgeType",
    "NodeResult",
    "WorkflowState",
    "GraphNode",
    "GraphEdge",
    "ConditionalRouter",
    "WorkflowGraph",
    "FunctionNode",
    "BranchNode",
    "ToolNode",
    "WorkflowBuilder",

    # Memory RAG
    "MemoryType",
    "MemoryStatus",
    "MemoryEntry",
    "RetrievalResult",
    "EmbeddingProvider",
    "SimpleEmbedding",
    "VectorIndex",
    "SimpleVectorIndex",
    "MemoryStore",
    "GREENLANG_MEMORY",
    "remember",
    "with_context",

    # Guardrails
    "GuardrailType",
    "ViolationSeverity",
    "ActionType",
    "GuardrailViolation",
    "GuardrailResult",
    "Guardrail",
    "PromptInjectionGuardrail",
    "DataLeakageGuardrail",
    "ActionGate",
    "PhysicalConstraint",
    "SafetyEnvelopeGuardrail",
    "Policy",
    "PolicyEnforcementGuardrail",
    "GuardrailOrchestrator",
    "create_default_orchestrator",
    "GREENLANG_GUARDRAILS",

    # A2A Protocol
    "MessageType",
    "AgentRole",
    "TaskStatus",
    "AgentCard",
    "A2AMessage",
    "DelegatedTask",
    "A2AAgent",
    "A2ARouter",
    "CrewTask",
    "AgentCrew",
    "GREENLANG_A2A_ROUTER",
]

__version__ = "2.0.0"
