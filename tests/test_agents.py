"""
Tests for the multi-agent financial research system.

Covers:
- State schema validation
- Individual agent node execution
- Tool wrapper functionality
- Graph routing logic
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.state import AgentState, ResearchPlan


# ---------------------------------------------------------------------------
# State Schema Tests
# ---------------------------------------------------------------------------

class TestAgentState:
    """Test the shared state schema."""

    def test_initial_state_creation(self):
        """Verify a valid initial state can be constructed."""
        state: AgentState = {
            "messages": [],
            "user_query": "Analyze Apple stock",
            "research_plan": None,
            "sec_data": None,
            "news_data": None,
            "quant_data": None,
            "final_report": None,
            "errors": [],
            "agent_trace": [],
        }
        assert state["user_query"] == "Analyze Apple stock"
        assert state["research_plan"] is None
        assert state["errors"] == []

    def test_research_plan_structure(self):
        """Verify ResearchPlan TypedDict accepts valid data."""
        plan: ResearchPlan = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "tasks": ["sec_filings", "news_sentiment", "quant_analysis"],
            "focus_areas": ["revenue growth", "market sentiment"],
            "time_horizon": "quarterly",
        }
        assert plan["ticker"] == "AAPL"
        assert len(plan["tasks"]) == 3
        assert "sec_filings" in plan["tasks"]


# ---------------------------------------------------------------------------
# Routing Logic Tests
# ---------------------------------------------------------------------------

class TestRoutingLogic:
    """Test the graph routing functions."""

    def test_route_all_agents(self):
        """When all tasks are present, all agents should be dispatched."""
        from src.graph import route_after_planning

        state: AgentState = {
            "messages": [],
            "user_query": "test",
            "research_plan": {
                "ticker": "AAPL",
                "company_name": "Apple",
                "tasks": ["sec_filings", "news_sentiment", "quant_analysis"],
                "focus_areas": [],
                "time_horizon": "quarterly",
            },
            "sec_data": None,
            "news_data": None,
            "quant_data": None,
            "final_report": None,
            "errors": [],
            "agent_trace": [],
        }

        targets = route_after_planning(state)
        assert "sec_filing_agent" in targets
        assert "news_sentiment_agent" in targets
        assert "quant_agent" in targets

    def test_route_partial_agents(self):
        """Only requested agents should be dispatched."""
        from src.graph import route_after_planning

        state: AgentState = {
            "messages": [],
            "user_query": "test",
            "research_plan": {
                "ticker": "TSLA",
                "company_name": "Tesla",
                "tasks": ["news_sentiment"],
                "focus_areas": [],
                "time_horizon": "recent",
            },
            "sec_data": None,
            "news_data": None,
            "quant_data": None,
            "final_report": None,
            "errors": [],
            "agent_trace": [],
        }

        targets = route_after_planning(state)
        assert targets == ["news_sentiment_agent"]

    def test_route_no_plan_goes_to_synthesizer(self):
        """If planning fails, route directly to synthesizer."""
        from src.graph import route_after_planning

        state: AgentState = {
            "messages": [],
            "user_query": "test",
            "research_plan": None,
            "sec_data": None,
            "news_data": None,
            "quant_data": None,
            "final_report": None,
            "errors": [],
            "agent_trace": [],
        }

        targets = route_after_planning(state)
        assert targets == ["synthesizer"]


# ---------------------------------------------------------------------------
# Tool Tests
# ---------------------------------------------------------------------------

class TestNewsTool:
    """Test the News API tool wrapper."""

    @pytest.mark.asyncio
    async def test_mock_articles_returned_without_api_key(self):
        """Without an API key, mock articles should be returned."""
        from src.tools.news_tool import NewsTool

        tool = NewsTool()
        articles = await tool.search_news("Apple AAPL stock", max_results=3)

        assert len(articles) <= 3
        assert all("title" in a for a in articles)
        assert all("source" in a for a in articles)

    @pytest.mark.asyncio
    async def test_mock_articles_contain_company_name(self):
        """Mock articles should reference the queried company."""
        from src.tools.news_tool import NewsTool

        tool = NewsTool()
        articles = await tool.search_news("Tesla TSLA", max_results=5)

        # At least some articles should mention the company
        titles = " ".join(a["title"] for a in articles)
        assert "Tesla" in titles


class TestYahooFinanceTool:
    """Test the Yahoo Finance tool wrapper."""

    @pytest.mark.asyncio
    async def test_mock_stock_info(self):
        """Mock stock info should return all required fields."""
        from src.tools.yahoo_finance_tool import YahooFinanceTool

        tool = YahooFinanceTool()
        # This will use mock data if yfinance is not installed
        info = await tool.get_stock_info("AAPL")

        assert info is not None
        assert "current_price" in info
        assert "pe_ratio" in info
        assert "market_cap" in info

    @pytest.mark.asyncio
    async def test_mock_price_history(self):
        """Mock price history should return technical indicators."""
        from src.tools.yahoo_finance_tool import YahooFinanceTool

        tool = YahooFinanceTool()
        history = await tool.get_price_history("AAPL")

        assert history is not None
        assert "ma_50" in history
        assert "volatility_30d" in history


class TestMCPServer:
    """Test the MCP tool server."""

    def test_list_tools_returns_all_tools(self):
        """All registered tools should appear in list_tools."""
        from src.tools.mcp_server import MCPToolServer

        server = MCPToolServer()
        tools = server.list_tools()

        tool_names = [t["name"] for t in tools]
        assert "get_sec_filing" in tool_names
        assert "search_news" in tool_names
        assert "get_stock_data" in tool_names
        assert "get_price_history" in tool_names

    def test_tool_schemas_are_valid(self):
        """Each tool should have a valid inputSchema."""
        from src.tools.mcp_server import MCPToolServer

        server = MCPToolServer()
        tools = server.list_tools()

        for tool in tools:
            assert "inputSchema" in tool
            assert "properties" in tool["inputSchema"]
            assert "required" in tool["inputSchema"]

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_error(self):
        """Calling a non-existent tool should raise ValueError."""
        from src.tools.mcp_server import MCPToolServer

        server = MCPToolServer()
        with pytest.raises(ValueError, match="Unknown tool"):
            await server.call_tool("nonexistent_tool", {})


# ---------------------------------------------------------------------------
# Integration Test (requires API keys — skip in CI)
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Requires API keys — run manually")
class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_research_workflow(self):
        """Run a complete research workflow and verify output structure."""
        from src.graph import run_research

        result = await run_research("Analyze Apple's financial health")

        assert result.get("final_report") is not None
        assert len(result.get("final_report", "")) > 100
        assert len(result.get("agent_trace", [])) > 0
