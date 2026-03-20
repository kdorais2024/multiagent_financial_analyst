"""
LangGraph orchestration for the multi-agent financial research workflow.

This module defines the state machine that coordinates agent execution:
  1. Planner  → parses user intent into a structured research plan
  2. Dispatch → fans out to SEC, News, and Quant agents in parallel
  3. Synthesizer → merges all agent outputs into a final research memo

The graph supports conditional routing (skip agents not needed),
parallel execution, and graceful error handling per agent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from src.agents.news_sentiment_agent import news_sentiment_node
from src.agents.planner import planner_node
from src.agents.quant_agent import quant_analysis_node
from src.agents.sec_filing_agent import sec_filing_node
from src.agents.synthesizer import synthesizer_node
from src.state import AgentState


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def should_dispatch_sec(state: AgentState) -> bool:
    """Check if the research plan requires SEC filing analysis."""
    plan = state.get("research_plan")
    if not plan:
        return False
    return "sec_filings" in plan.get("tasks", [])


def should_dispatch_news(state: AgentState) -> bool:
    """Check if the research plan requires news sentiment analysis."""
    plan = state.get("research_plan")
    if not plan:
        return False
    return "news_sentiment" in plan.get("tasks", [])


def should_dispatch_quant(state: AgentState) -> bool:
    """Check if the research plan requires quantitative analysis."""
    plan = state.get("research_plan")
    if not plan:
        return False
    return "quant_analysis" in plan.get("tasks", [])


def route_after_planning(state: AgentState) -> list[str]:
    """
    Determine which agent nodes to execute based on the research plan.
    Returns a list of node names for LangGraph's conditional fan-out.
    """
    plan = state.get("research_plan")
    if not plan:
        return ["synthesizer"]

    targets = []
    tasks = plan.get("tasks", [])

    if "sec_filings" in tasks:
        targets.append("sec_filing_agent")
    if "news_sentiment" in tasks:
        targets.append("news_sentiment_agent")
    if "quant_analysis" in tasks:
        targets.append("quant_agent")

    return targets if targets else ["synthesizer"]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_research_graph() -> StateGraph:
    """
    Construct the LangGraph state machine for financial research.

    Graph topology:
        planner → [sec_filing_agent, news_sentiment_agent, quant_agent] → synthesizer → END

    Agents are dispatched in parallel based on the planner's routing decision.
    Each agent writes to its own state field, avoiding write conflicts.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("sec_filing_agent", sec_filing_node)
    graph.add_node("news_sentiment_agent", news_sentiment_node)
    graph.add_node("quant_agent", quant_analysis_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Entry point
    graph.set_entry_point("planner")

    # Conditional fan-out from planner to agents
    graph.add_conditional_edges(
        "planner",
        route_after_planning,
        {
            "sec_filing_agent": "sec_filing_agent",
            "news_sentiment_agent": "news_sentiment_agent",
            "quant_agent": "quant_agent",
            "synthesizer": "synthesizer",
        },
    )

    # All agents converge to synthesizer
    graph.add_edge("sec_filing_agent", "synthesizer")
    graph.add_edge("news_sentiment_agent", "synthesizer")
    graph.add_edge("quant_agent", "synthesizer")

    # Synthesizer → END
    graph.add_edge("synthesizer", END)

    return graph


def compile_graph():
    """Build and compile the research graph, ready for invocation."""
    graph = build_research_graph()
    return graph.compile()


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

async def run_research(query: str) -> dict[str, Any]:
    """
    Execute a full research workflow for the given query.

    Args:
        query: Natural language research request
              (e.g., "Analyze Apple's financial health and market sentiment")

    Returns:
        Final state dict containing all agent outputs and the synthesized report.
    """
    app = compile_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "research_plan": None,
        "sec_data": None,
        "news_data": None,
        "quant_data": None,
        "final_report": None,
        "errors": [],
        "agent_trace": [],
    }

    start_time = time.time()
    result = await app.ainvoke(initial_state)
    elapsed = time.time() - start_time

    result["agent_trace"] = result.get("agent_trace", []) + [
        {"agent": "orchestrator", "event": "workflow_complete", "duration_sec": round(elapsed, 2)}
    ]

    return result


async def stream_research(query: str):
    """
    Stream research workflow events for real-time UI updates.

    Yields:
        Tuple of (node_name, state_update) as each agent completes.
    """
    app = compile_graph()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "research_plan": None,
        "sec_data": None,
        "news_data": None,
        "quant_data": None,
        "final_report": None,
        "errors": [],
        "agent_trace": [],
    }

    async for event in app.astream(initial_state, stream_mode="updates"):
        yield event


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Financial Research Assistant")
    parser.add_argument("--query", type=str, required=True, help="Research query")
    args = parser.parse_args()

    result = asyncio.run(run_research(args.query))

    print("\n" + "=" * 80)
    print("RESEARCH REPORT")
    print("=" * 80)
    print(result.get("final_report", "No report generated."))
    print("\n" + "=" * 80)

    if result.get("errors"):
        print("\nErrors encountered:")
        for err in result["errors"]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
