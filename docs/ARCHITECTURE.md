# Architecture Overview

## System Design

The Multi-Agent Financial Research Assistant uses a **supervisor pattern** where a central orchestrator (built with LangGraph) coordinates specialized worker agents. This architecture was chosen over peer-to-peer agent communication for three reasons:

1. **Deterministic routing** — The orchestrator controls which agents run and in what order, making the system predictable and debuggable.
2. **Parallel execution** — Independent agents (SEC, News, Quant) run concurrently after planning, reducing total latency.
3. **Graceful degradation** — If one agent fails, the others still produce results, and the synthesizer works with whatever data is available.

## Data Flow

```
User Query
    │
    ▼
┌─────────────────────────┐
│     Planner Node         │  ← LLM call: parse intent → ResearchPlan
│  (Intent Classification) │
└─────────┬───────────────┘
          │
          │  ResearchPlan.tasks determines routing
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
  [SEC] [News] [Quant]      ← Run in parallel
    │     │     │
    │     │     │            Each writes to its own state field:
    │     │     │              sec_data, news_data, quant_data
    ▼     ▼     ▼
┌─────────────────────────┐
│    Synthesizer Node      │  ← LLM call: merge data → research memo
│  (Report Generation)     │
└─────────────────────────┘
          │
          ▼
    Final Report (Markdown)
```

## State Management

The system uses LangGraph's `AgentState` TypedDict as the single source of truth. Key design choices:

- **Immutable agent outputs**: Each agent writes to its own dedicated field (`sec_data`, `news_data`, `quant_data`), preventing write conflicts during parallel execution.
- **Append-only lists**: `messages`, `errors`, and `agent_trace` use LangGraph's `operator.add` reducer, so concurrent agents can safely append without overwriting.
- **Nullable fields**: All agent output fields are `Optional`, allowing the synthesizer to handle partial data gracefully.

## Tool Architecture (MCP)

Tools are exposed via a Model Context Protocol (MCP) server that provides:

- **Tool discovery**: `list_tools()` returns JSON Schema definitions for all available tools.
- **Standardized invocation**: `call_tool(name, args)` provides a uniform interface regardless of the underlying data source.
- **Schema validation**: Input parameters are validated against JSON Schema before execution.

This abstraction means adding a new data source (e.g., Bloomberg API) requires only:
1. Writing a tool wrapper class
2. Registering it in `MCPToolServer._register_tools()`

No changes to agent code or the graph topology.

## Observability

Every node in the graph emits structured trace data:

```json
{
    "agent": "sec_filing",
    "status": "success",
    "duration_sec": 2.34,
    "filing_type": "10-K"
}
```

When LangSmith is enabled (`LANGCHAIN_TRACING_V2=true`), the full execution trace — including LLM inputs/outputs, token counts, and latencies — is sent to LangSmith for analysis. This provides:

- End-to-end latency breakdown by agent
- Token cost attribution per agent
- Error rate tracking and alerting
- Prompt regression testing across versions

## Error Handling Strategy

The system uses a **fail-soft** approach:

1. **Agent-level try/catch**: Each agent catches its own exceptions and writes error context to `state["errors"]`.
2. **Fallback data**: Tools provide mock data when API keys are missing, enabling development without external dependencies.
3. **Synthesizer adaptation**: The synthesizer explicitly notes which data sources are unavailable rather than hallucinating missing information.
4. **Timeout protection**: All external API calls use `httpx` with configurable timeouts (default 15s).
