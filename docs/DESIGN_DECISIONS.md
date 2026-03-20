# Design Decisions

This document captures key design decisions, the alternatives considered, and the rationale for each choice.

---

## 1. LangGraph over CrewAI / AutoGen

**Decision**: Use LangGraph as the agent orchestration framework.

**Alternatives considered**:
- **CrewAI**: Higher-level abstraction with role-based agents. Easier to prototype but less control over execution flow, state management, and error handling.
- **AutoGen**: Microsoft's multi-agent framework. Strong for conversational agent patterns but opinionated about agent-to-agent communication.

**Rationale**:
- LangGraph provides **explicit state machine semantics** — every node, edge, and conditional branch is visible in code, making the system auditable (important for financial applications).
- **Fine-grained control** over parallel vs. sequential execution, which matters when optimizing latency.
- **Native LangSmith integration** for production-grade observability.
- **No magic**: The graph topology is defined in code, not inferred from agent descriptions.

**Tradeoff**: More boilerplate than CrewAI for simple use cases, but the explicitness pays off in debugging and production reliability.

---

## 2. Parallel Agent Dispatch

**Decision**: Run SEC, News, and Quant agents in parallel after planning.

**Alternatives considered**:
- **Sequential execution**: Simpler but 3x slower since agents are independent.
- **Dynamic ordering**: Let agents depend on each other's outputs (e.g., Quant agent uses SEC data). More sophisticated but adds complexity and latency.

**Rationale**:
- The three research agents are **independent** — none requires output from another.
- Parallel execution reduces total latency from ~15s (sequential) to ~6s (parallel, limited by the slowest agent).
- LangGraph's conditional fan-out natively supports this pattern.

**Tradeoff**: If we later add agents that depend on earlier results (e.g., a "Valuation Agent" that needs both SEC and Quant data), we'll need to add a second dispatch phase.

---

## 3. MCP for Tool Standardization

**Decision**: Wrap all external tools behind a Model Context Protocol (MCP) server interface.

**Alternatives considered**:
- **Direct function calls**: Simpler but couples agents to specific tool implementations.
- **LangChain Tools**: Built-in tool abstraction, but less portable outside the LangChain ecosystem.

**Rationale**:
- MCP provides a **standard protocol** for tool discovery and invocation that works across LLM providers and agent frameworks.
- Adding a new data source requires only implementing the tool handler and registering it — **no changes to agent code**.
- The JSON Schema-based parameter validation catches malformed tool calls before they hit external APIs.

**Tradeoff**: Slight overhead from the abstraction layer, but negligible compared to actual API call latency.

---

## 4. FAISS for Caching (Not a Full RAG Pipeline)

**Decision**: Use FAISS as a lightweight cache for repeated queries, not as a full document retrieval system.

**Alternatives considered**:
- **ChromaDB**: Managed vector store with metadata filtering. Better for production RAG but heavier dependency.
- **Pinecone**: Fully managed cloud vector DB. Great for scale but adds external dependency and cost.
- **No caching**: Simplest but wasteful — identical queries hit all APIs again.

**Rationale**:
- For a portfolio project, FAISS demonstrates **vector store proficiency** without requiring cloud infrastructure.
- Embedding previous research results enables **semantic cache hits** — "Analyze AAPL" and "How is Apple doing?" can return cached results.
- Easy to upgrade to ChromaDB or Pinecone later by swapping the store implementation.

**Tradeoff**: No built-in metadata filtering or persistence guarantees. Acceptable for a demonstration project.

---

## 5. Mock Data Fallbacks

**Decision**: Every external tool provides mock data when API keys are missing.

**Rationale**:
- Allows **zero-config local development** — `git clone && pip install && streamlit run` works immediately.
- Reviewers (recruiters, hiring managers) can explore the project without obtaining API keys.
- Mock data is clearly labeled and uses realistic but synthetic values.
- Tests run reliably in CI without API key secrets.

**Tradeoff**: Mock data doesn't exercise real API error handling paths. Integration tests with real APIs are separate and require manual execution.

---

## 6. Synthesizer as Final LLM Call (Not Templating)

**Decision**: Use an LLM call to synthesize the final report rather than a template-based approach.

**Alternatives considered**:
- **Jinja2 templates**: Deterministic, fast, but produces rigid reports that can't adapt to varying data availability.
- **Hybrid**: Template for structure, LLM for narrative sections.

**Rationale**:
- The synthesizer must **handle partial data gracefully** — if one agent fails, the report should still be coherent, not show empty template sections.
- LLM synthesis produces **more readable narratives** that connect findings across data sources.
- The structured prompt ensures consistent output format while allowing adaptive content.

**Tradeoff**: Higher latency and token cost for the synthesis step (~2-3s). Acceptable given it runs once per query.

---

## 7. Streamlit over FastAPI + React

**Decision**: Use Streamlit for the frontend.

**Alternatives considered**:
- **FastAPI + React**: Production-grade separation of concerns, but doubles the codebase and adds JavaScript complexity.
- **Gradio**: Quick prototyping but less customizable than Streamlit.

**Rationale**:
- Streamlit is the **standard for data science portfolio projects** — reviewers know how to evaluate it.
- Single-language stack (Python only) keeps the project focused on ML/AI skills.
- Built-in chat interface, status indicators, and metric cards match our needs.
- Docker deployment is straightforward.

**Tradeoff**: Not production-grade for high-traffic applications. A real production system would use FastAPI + WebSocket for streaming.
