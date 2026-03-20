"""
Streamlit UI for the Multi-Agent Financial Research Assistant.

Provides:
- Chat-based interface for submitting research queries
- Real-time agent execution status tracking
- Formatted research memo display
- Agent trace/observability panel
"""

import asyncio
import json
import time

import streamlit as st

from src.graph import run_research, stream_research

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Research Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .agent-status {
        padding: 8px 16px;
        border-radius: 8px;
        margin: 4px 0;
        font-size: 14px;
    }
    .agent-success { background-color: #d4edda; color: #155724; }
    .agent-running { background-color: #fff3cd; color: #856404; }
    .agent-error { background-color: #f8d7da; color: #721c24; }
    .metric-card {
        background: #f8f9fa;
        padding: 16px;
        border-radius: 8px;
        border-left: 4px solid #2E5090;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏦 Research Assistant")
    st.markdown("---")

    st.subheader("Configuration")
    provider = st.selectbox("LLM Provider", ["OpenAI", "Anthropic"], index=0)
    model = st.text_input(
        "Model",
        value="gpt-4o" if provider == "OpenAI" else "claude-sonnet-4-20250514",
    )

    st.markdown("---")
    st.subheader("Example Queries")
    examples = [
        "Analyze Apple's financial health and market sentiment",
        "What are the key risks for Tesla based on recent filings?",
        "Compare NVIDIA's valuation metrics to the semiconductor sector",
        "Summarize Microsoft's latest 10-K and recent news outlook",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state["query_input"] = ex

    st.markdown("---")
    st.caption("Built with LangGraph, MCP, and LangSmith")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("Multi-Agent Financial Research Assistant")
st.markdown(
    "Enter a research query below. Specialized agents will collaborate to analyze "
    "SEC filings, news sentiment, and quantitative metrics, then synthesize a "
    "comprehensive research memo."
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_traces" not in st.session_state:
    st.session_state.agent_traces = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("is_report"):
            st.markdown(msg["content"])
        else:
            st.write(msg["content"])

# Chat input
query = st.chat_input(
    "Ask a financial research question...",
    key="chat_input",
)

# Handle pre-filled query from sidebar
if "query_input" in st.session_state:
    query = st.session_state.pop("query_input")

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Run research
    with st.chat_message("assistant"):
        status_container = st.container()
        report_container = st.container()

        with status_container:
            with st.status("Researching...", expanded=True) as status:
                st.write("🧠 Planning research strategy...")

                # Run the full research pipeline
                start_time = time.time()
                result = asyncio.run(run_research(query))
                elapsed = time.time() - start_time

                # Show agent execution trace
                traces = result.get("agent_trace", [])
                for trace in traces:
                    agent = trace.get("agent", "unknown")
                    agent_status = trace.get("status", "unknown")
                    duration = trace.get("duration_sec", 0)

                    icon = "✅" if agent_status == "success" else "⚠️" if agent_status == "no_data" else "❌"
                    st.write(f"{icon} **{agent}** — {agent_status} ({duration}s)")

                status.update(
                    label=f"Research complete ({elapsed:.1f}s)",
                    state="complete",
                )

        # Display the report
        with report_container:
            report = result.get("final_report", "No report generated.")
            st.markdown(report)

            # Show errors if any
            errors = result.get("errors", [])
            if errors:
                with st.expander("⚠️ Warnings & Errors"):
                    for err in errors:
                        st.warning(err)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": report,
            "is_report": True,
        })
        st.session_state.agent_traces.append(traces)

# ---------------------------------------------------------------------------
# Agent Trace Panel (bottom)
# ---------------------------------------------------------------------------
if st.session_state.agent_traces:
    with st.expander("📊 Agent Execution Trace", expanded=False):
        latest_trace = st.session_state.agent_traces[-1]

        cols = st.columns(len(latest_trace))
        for i, trace in enumerate(latest_trace):
            with cols[i]:
                agent = trace.get("agent", "unknown")
                status = trace.get("status", "unknown")
                duration = trace.get("duration_sec", 0)

                color = "green" if status == "success" else "orange" if status == "no_data" else "red"
                st.metric(label=agent.replace("_", " ").title(), value=f"{duration}s", delta=status)

        st.json(latest_trace)
