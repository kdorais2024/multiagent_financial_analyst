"""
Shared state schema for the multi-agent financial research workflow.

This module defines the TypedDict that flows through the LangGraph state machine.
Each agent reads from and writes to specific fields, enabling parallel execution
and clean data handoff between agents.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage


class ResearchPlan(TypedDict):
    """Structured research plan created by the planner agent."""

    ticker: str
    company_name: str
    tasks: list[str]  # Which agents to dispatch
    focus_areas: list[str]  # Specific areas of interest
    time_horizon: str  # "recent", "quarterly", "annual"


class SECFilingData(TypedDict):
    """Data returned by the SEC Filing Agent."""

    filing_type: str  # "10-K", "10-Q"
    filing_date: str
    revenue: float | None
    net_income: float | None
    total_assets: float | None
    total_liabilities: float | None
    risk_factors: list[str]
    management_discussion: str
    raw_text: str


class NewsSentimentData(TypedDict):
    """Data returned by the News Sentiment Agent."""

    articles_analyzed: int
    overall_sentiment: float  # -1.0 to 1.0
    sentiment_label: str  # "Bullish", "Bearish", "Neutral"
    key_themes: list[str]
    top_articles: list[dict[str, str]]  # title, source, sentiment, summary


class QuantAnalysisData(TypedDict):
    """Data returned by the Quantitative Analysis Agent."""

    current_price: float | None
    market_cap: float | None
    pe_ratio: float | None
    eps: float | None
    fifty_two_week_high: float | None
    fifty_two_week_low: float | None
    moving_avg_50: float | None
    moving_avg_200: float | None
    volatility_30d: float | None
    debt_to_equity: float | None
    return_on_equity: float | None
    price_trend: str  # "Uptrend", "Downtrend", "Sideways"
    summary: str


class AgentState(TypedDict):
    """
    Central state object flowing through the LangGraph.

    Fields are populated incrementally as agents execute.
    The `messages` field uses the `operator.add` reducer to
    append conversation history across nodes.
    """

    # Conversation history (appended via reducer)
    messages: Annotated[list[BaseMessage], operator.add]

    # User input
    user_query: str

    # Planning phase
    research_plan: ResearchPlan | None

    # Agent outputs
    sec_data: SECFilingData | None
    news_data: NewsSentimentData | None
    quant_data: QuantAnalysisData | None

    # Synthesis
    final_report: str | None

    # Metadata
    errors: Annotated[list[str], operator.add]
    agent_trace: Annotated[list[dict[str, Any]], operator.add]
