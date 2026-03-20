"""
SEC Filing Agent — retrieves and analyzes SEC filings from EDGAR.

This agent fetches the most recent 10-K or 10-Q filing for a given ticker,
extracts key financial metrics, risk factors, and management discussion,
then structures the data for the synthesizer.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from configs.settings import get_llm
from src.state import AgentState, SECFilingData
from src.tools.edgar_tool import EdgarTool

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a financial analyst specializing in SEC filing analysis.
Given the raw text of a SEC filing, extract the following information as JSON:

{{
    "revenue": <float or null>,
    "net_income": <float or null>,
    "total_assets": <float or null>,
    "total_liabilities": <float or null>,
    "risk_factors": [<top 5 risk factors as short strings>],
    "management_discussion": "<2-3 sentence summary of MD&A section>"
}}

All dollar amounts should be in millions (e.g., 394328 for $394.328B).
If a value is not found, use null. Be precise with numbers.
""",
        ),
        ("human", "Filing type: {filing_type}\nFiling date: {filing_date}\n\nFiling text (truncated):\n{filing_text}"),
    ]
)


async def sec_filing_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Fetch and analyze SEC filings.

    Reads: research_plan (ticker, time_horizon)
    Writes: sec_data, messages, agent_trace, errors
    """
    start = time.time()
    plan = state.get("research_plan")

    if not plan:
        return {
            "sec_data": None,
            "errors": ["SEC Filing Agent: No research plan available"],
            "agent_trace": [{"agent": "sec_filing", "status": "skipped", "duration_sec": 0}],
        }

    ticker = plan["ticker"]
    company = plan.get("company_name", ticker)

    try:
        # Fetch filing from EDGAR
        edgar = EdgarTool()
        filing_type = "10-K" if plan.get("time_horizon") == "annual" else "10-Q"
        filing = await edgar.get_latest_filing(ticker, filing_type=filing_type)

        if not filing:
            return {
                "sec_data": None,
                "errors": [f"SEC Filing Agent: No {filing_type} filing found for {ticker}"],
                "messages": [AIMessage(content=f"Could not retrieve {filing_type} filing for {company}.")],
                "agent_trace": [{"agent": "sec_filing", "status": "no_data", "duration_sec": round(time.time() - start, 2)}],
            }

        # Use LLM to extract structured data from filing text
        llm = get_llm()
        chain = EXTRACTION_PROMPT | llm

        response = await chain.ainvoke({
            "filing_type": filing["filing_type"],
            "filing_date": filing["filing_date"],
            "filing_text": filing["text"][:15000],  # Truncate for context window
        })

        import json
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]

        extracted = json.loads(content.strip())

        sec_data: SECFilingData = {
            "filing_type": filing["filing_type"],
            "filing_date": filing["filing_date"],
            "revenue": extracted.get("revenue"),
            "net_income": extracted.get("net_income"),
            "total_assets": extracted.get("total_assets"),
            "total_liabilities": extracted.get("total_liabilities"),
            "risk_factors": extracted.get("risk_factors", []),
            "management_discussion": extracted.get("management_discussion", ""),
            "raw_text": filing["text"][:5000],
        }

        return {
            "sec_data": sec_data,
            "messages": [AIMessage(content=f"SEC filing analysis complete for {company} ({filing_type}).")],
            "agent_trace": [{"agent": "sec_filing", "status": "success", "filing_type": filing_type, "duration_sec": round(time.time() - start, 2)}],
        }

    except Exception as e:
        return {
            "sec_data": None,
            "errors": [f"SEC Filing Agent error: {e}"],
            "messages": [AIMessage(content=f"SEC filing analysis encountered an error: {e}")],
            "agent_trace": [{"agent": "sec_filing", "status": "error", "error": str(e), "duration_sec": round(time.time() - start, 2)}],
        }
