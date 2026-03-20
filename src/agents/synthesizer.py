"""
Synthesizer Agent — merges all agent outputs into a final research memo.

This is the terminal node in the LangGraph. It receives structured data
from SEC, News, and Quant agents and produces a cohesive, executive-ready
investment research memo.
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from configs.settings import get_llm
from src.state import AgentState

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a senior financial research analyst. Synthesize the following
data from multiple research agents into a professional investment research memo.

## Report Structure:

### Executive Summary
A concise 2-3 sentence overview of the company's current position.

### Financial Health (from SEC filing data)
Key financial metrics and what they indicate. If no SEC data is available, note it.

### Market Sentiment (from news analysis)
Overall sentiment, key themes, and notable recent developments. If no news data, note it.

### Quantitative Metrics (from market data)
Valuation, price trends, and risk indicators. If no quant data, note it.

### Key Risk Factors
Top risks identified across all data sources.

### Conclusion
Overall assessment with a confidence level (High/Medium/Low) based on data coverage.

## Guidelines:
- Be specific: reference actual numbers, not vague qualifiers
- Be balanced: present both bullish and bearish signals
- Flag data gaps explicitly rather than hiding them
- Keep the total memo under 800 words
- Use markdown formatting
""",
        ),
        ("human", "Company: {company} ({ticker})\n\nSEC Filing Data:\n{sec_data}\n\nNews Sentiment Data:\n{news_data}\n\nQuantitative Data:\n{quant_data}\n\nResearch Focus Areas: {focus_areas}"),
    ]
)


def _format_data(data: dict | None, label: str) -> str:
    """Format agent data for the synthesis prompt."""
    if data is None:
        return f"No {label} data available (agent did not return results)."
    return json.dumps(data, indent=2, default=str)


async def synthesizer_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Synthesize all agent outputs into a research memo.

    Reads: research_plan, sec_data, news_data, quant_data
    Writes: final_report, messages, agent_trace
    """
    start = time.time()
    plan = state.get("research_plan")

    if not plan:
        report = (
            "## Research Report\n\n"
            "Unable to generate a report. The planning phase did not produce "
            "a valid research plan. Please try rephrasing your query with a "
            "specific company name or ticker symbol."
        )
        return {
            "final_report": report,
            "messages": [AIMessage(content=report)],
            "agent_trace": [{"agent": "synthesizer", "status": "no_plan", "duration_sec": round(time.time() - start, 2)}],
        }

    ticker = plan["ticker"]
    company = plan.get("company_name", ticker)
    focus_areas = ", ".join(plan.get("focus_areas", ["general analysis"]))

    try:
        llm = get_llm()
        chain = SYNTHESIS_PROMPT | llm

        response = await chain.ainvoke({
            "company": company,
            "ticker": ticker,
            "sec_data": _format_data(state.get("sec_data"), "SEC filing"),
            "news_data": _format_data(state.get("news_data"), "news sentiment"),
            "quant_data": _format_data(state.get("quant_data"), "quantitative"),
            "focus_areas": focus_areas,
        })

        report = response.content.strip()

        # Append metadata footer
        errors = state.get("errors", [])
        if errors:
            report += "\n\n---\n*Data gaps encountered:*\n"
            for err in errors:
                report += f"- {err}\n"

        return {
            "final_report": report,
            "messages": [AIMessage(content="Research memo generated successfully.")],
            "agent_trace": [{"agent": "synthesizer", "status": "success", "report_length": len(report), "duration_sec": round(time.time() - start, 2)}],
        }

    except Exception as e:
        fallback = (
            f"## Research Report for {company} ({ticker})\n\n"
            f"Report generation failed: {e}\n\n"
            "### Available Raw Data\n"
            f"- SEC Data: {'Available' if state.get('sec_data') else 'Not available'}\n"
            f"- News Data: {'Available' if state.get('news_data') else 'Not available'}\n"
            f"- Quant Data: {'Available' if state.get('quant_data') else 'Not available'}\n"
        )
        return {
            "final_report": fallback,
            "errors": [f"Synthesizer error: {e}"],
            "messages": [AIMessage(content=f"Report synthesis failed: {e}")],
            "agent_trace": [{"agent": "synthesizer", "status": "error", "error": str(e), "duration_sec": round(time.time() - start, 2)}],
        }
