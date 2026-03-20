"""
Quantitative Analysis Agent — computes financial ratios and price analytics.

Uses Yahoo Finance data to calculate key metrics including valuation ratios,
technical indicators, and risk measures. This agent performs pure computation
without LLM calls for maximum reliability, using LLM only for the narrative.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from configs.settings import get_llm
from src.state import AgentState, QuantAnalysisData
from src.tools.yahoo_finance_tool import YahooFinanceTool

QUANT_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a quantitative financial analyst. Given the following metrics
for {company} ({ticker}), write a concise 2-3 sentence summary of the
quantitative outlook. Focus on what the numbers tell us about valuation,
momentum, and risk.

Be specific — reference actual numbers. Do not be generic.
""",
        ),
        ("human", "Metrics:\n{metrics_text}"),
    ]
)


async def quant_analysis_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Perform quantitative financial analysis.

    Reads: research_plan (ticker)
    Writes: quant_data, messages, agent_trace, errors
    """
    start = time.time()
    plan = state.get("research_plan")

    if not plan:
        return {
            "quant_data": None,
            "errors": ["Quant Agent: No research plan available"],
            "agent_trace": [{"agent": "quant_analysis", "status": "skipped", "duration_sec": 0}],
        }

    ticker = plan["ticker"]
    company = plan.get("company_name", ticker)

    try:
        yf_tool = YahooFinanceTool()

        # Fetch core data
        info = await yf_tool.get_stock_info(ticker)
        price_history = await yf_tool.get_price_history(ticker, period="6mo")

        if not info:
            return {
                "quant_data": None,
                "errors": [f"Quant Agent: Could not fetch data for {ticker}"],
                "messages": [AIMessage(content=f"Could not retrieve market data for {company}.")],
                "agent_trace": [{"agent": "quant_analysis", "status": "no_data", "duration_sec": round(time.time() - start, 2)}],
            }

        # Compute derived metrics
        volatility = price_history.get("volatility_30d") if price_history else None
        ma_50 = price_history.get("ma_50") if price_history else None
        ma_200 = price_history.get("ma_200") if price_history else None
        current_price = info.get("current_price")

        # Determine price trend
        price_trend = "Sideways"
        if ma_50 and ma_200:
            if ma_50 > ma_200:
                price_trend = "Uptrend"
            elif ma_50 < ma_200:
                price_trend = "Downtrend"

        quant_data: QuantAnalysisData = {
            "current_price": current_price,
            "market_cap": info.get("market_cap"),
            "pe_ratio": info.get("pe_ratio"),
            "eps": info.get("eps"),
            "fifty_two_week_high": info.get("fifty_two_week_high"),
            "fifty_two_week_low": info.get("fifty_two_week_low"),
            "moving_avg_50": ma_50,
            "moving_avg_200": ma_200,
            "volatility_30d": volatility,
            "debt_to_equity": info.get("debt_to_equity"),
            "return_on_equity": info.get("return_on_equity"),
            "price_trend": price_trend,
            "summary": "",  # Will be filled by LLM
        }

        # Generate narrative summary using LLM
        metrics_text = "\n".join(
            f"  {k}: {v}" for k, v in quant_data.items() if v is not None and k != "summary"
        )

        llm = get_llm()
        chain = QUANT_SUMMARY_PROMPT | llm
        summary_response = await chain.ainvoke({
            "company": company,
            "ticker": ticker,
            "metrics_text": metrics_text,
        })
        quant_data["summary"] = summary_response.content.strip()

        return {
            "quant_data": quant_data,
            "messages": [AIMessage(content=f"Quantitative analysis complete for {company}. Trend: {price_trend}.")],
            "agent_trace": [{"agent": "quant_analysis", "status": "success", "trend": price_trend, "duration_sec": round(time.time() - start, 2)}],
        }

    except Exception as e:
        return {
            "quant_data": None,
            "errors": [f"Quant Agent error: {e}"],
            "messages": [AIMessage(content=f"Quantitative analysis failed: {e}")],
            "agent_trace": [{"agent": "quant_analysis", "status": "error", "error": str(e), "duration_sec": round(time.time() - start, 2)}],
        }
