"""
Planner Agent — parses user intent and creates a structured research plan.

This is the first node in the LangGraph. It takes the user's natural language
query and produces a ResearchPlan that determines which downstream agents
to dispatch and what to focus on.
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from configs.settings import get_llm
from src.state import AgentState, ResearchPlan

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a financial research planning agent. Your job is to analyze the user's
research request and produce a structured plan.

Given a user query, extract:
1. **ticker**: The stock ticker symbol (e.g., "AAPL" for Apple). Infer from company name if needed.
2. **company_name**: Full company name.
3. **tasks**: Which research agents to dispatch. Choose from:
   - "sec_filings" — for fundamental analysis, financial statements, risk factors
   - "news_sentiment" — for recent news and market sentiment
   - "quant_analysis" — for price data, ratios, technical indicators
4. **focus_areas**: Specific topics the user is interested in (e.g., "revenue growth", "debt levels").
5. **time_horizon**: "recent" (last month), "quarterly" (last quarter), or "annual" (last year).

Respond ONLY with valid JSON matching this schema:
{{
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "tasks": ["sec_filings", "news_sentiment", "quant_analysis"],
    "focus_areas": ["revenue growth", "market sentiment"],
    "time_horizon": "quarterly"
}}

If the query is ambiguous, default to dispatching ALL agents.
If you cannot identify a company, set ticker to "UNKNOWN".
""",
        ),
        ("human", "{query}"),
    ]
)


async def planner_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Parse user query into a structured research plan.

    Reads: user_query
    Writes: research_plan, messages, agent_trace
    """
    start = time.time()
    query = state["user_query"]

    try:
        llm = get_llm()
        chain = PLANNER_PROMPT | llm

        response = await chain.ainvoke({"query": query})
        content = response.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()

        plan: ResearchPlan = json.loads(content)

        # Validate required fields
        if "ticker" not in plan or plan["ticker"] == "UNKNOWN":
            return {
                "research_plan": None,
                "errors": [f"Could not identify a company from query: '{query}'"],
                "messages": [AIMessage(content="I couldn't identify a company from your query. Please specify a company name or ticker.")],
                "agent_trace": [{"agent": "planner", "status": "failed", "reason": "unknown_ticker", "duration_sec": round(time.time() - start, 2)}],
            }

        # Default to all tasks if none specified
        if not plan.get("tasks"):
            plan["tasks"] = ["sec_filings", "news_sentiment", "quant_analysis"]

        return {
            "research_plan": plan,
            "messages": [AIMessage(content=f"Research plan created for {plan['company_name']} ({plan['ticker']}). Dispatching agents: {', '.join(plan['tasks'])}")],
            "agent_trace": [{"agent": "planner", "status": "success", "plan": plan, "duration_sec": round(time.time() - start, 2)}],
        }

    except json.JSONDecodeError as e:
        return {
            "research_plan": None,
            "errors": [f"Planner failed to produce valid JSON: {e}"],
            "messages": [AIMessage(content="I had trouble parsing the research plan. Please try rephrasing your query.")],
            "agent_trace": [{"agent": "planner", "status": "error", "error": str(e), "duration_sec": round(time.time() - start, 2)}],
        }
    except Exception as e:
        return {
            "research_plan": None,
            "errors": [f"Planner error: {e}"],
            "messages": [AIMessage(content=f"Planning failed: {e}")],
            "agent_trace": [{"agent": "planner", "status": "error", "error": str(e), "duration_sec": round(time.time() - start, 2)}],
        }
