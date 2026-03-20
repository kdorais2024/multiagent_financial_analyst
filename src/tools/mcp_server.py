"""
Model Context Protocol (MCP) Tool Server — standardized tool interface.

Exposes financial research tools via the MCP protocol, allowing any
MCP-compatible agent to discover and call tools with consistent schemas.

This module defines the MCP server that wraps our financial tools
(EDGAR, News, Yahoo Finance) into a standardized tool-calling interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from src.tools.edgar_tool import EdgarTool
from src.tools.news_tool import NewsTool
from src.tools.yahoo_finance_tool import YahooFinanceTool


@dataclass
class MCPTool:
    """Definition of a single MCP-compatible tool."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    handler: Callable[..., Awaitable[Any]]


class MCPToolServer:
    """
    MCP-compatible tool server exposing financial research tools.

    Follows MCP specification for tool discovery and invocation:
    - list_tools() → returns available tools with schemas
    - call_tool(name, args) → invokes the tool and returns results

    This enables standardized tool-calling across different LLM providers
    and agent frameworks.
    """

    def __init__(self):
        self._edgar = EdgarTool()
        self._news = NewsTool()
        self._yf = YahooFinanceTool()
        self._tools: dict[str, MCPTool] = {}
        self._register_tools()

    def _register_tools(self):
        """Register all available financial research tools."""

        self._tools["get_sec_filing"] = MCPTool(
            name="get_sec_filing",
            description="Fetch the most recent SEC filing (10-K or 10-Q) for a given stock ticker. Returns filing metadata and text content.",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'AAPL')",
                    },
                    "filing_type": {
                        "type": "string",
                        "enum": ["10-K", "10-Q", "8-K"],
                        "default": "10-K",
                        "description": "Type of SEC filing to retrieve",
                    },
                },
                "required": ["ticker"],
            },
            handler=self._handle_sec_filing,
        )

        self._tools["search_news"] = MCPTool(
            name="search_news",
            description="Search for recent financial news articles about a company. Returns article titles, sources, dates, and snippets.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'Apple AAPL earnings')",
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum number of articles to return",
                    },
                },
                "required": ["query"],
            },
            handler=self._handle_news_search,
        )

        self._tools["get_stock_data"] = MCPTool(
            name="get_stock_data",
            description="Fetch current stock information including price, P/E ratio, market cap, and other fundamental metrics.",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'AAPL')",
                    },
                },
                "required": ["ticker"],
            },
            handler=self._handle_stock_data,
        )

        self._tools["get_price_history"] = MCPTool(
            name="get_price_history",
            description="Fetch historical price data with computed technical indicators (moving averages, volatility).",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "period": {
                        "type": "string",
                        "enum": ["1mo", "3mo", "6mo", "1y", "2y"],
                        "default": "6mo",
                        "description": "Historical period to analyze",
                    },
                },
                "required": ["ticker"],
            },
            handler=self._handle_price_history,
        )

    def list_tools(self) -> list[dict[str, Any]]:
        """Return MCP-formatted tool definitions for discovery."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Invoke a tool by name with the given arguments.

        Args:
            name: Tool name (must match a registered tool)
            arguments: Tool arguments matching the tool's parameter schema

        Returns:
            Tool result as a dict

        Raises:
            ValueError: If the tool name is not found
        """
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}. Available: {list(self._tools.keys())}")

        return await tool.handler(**arguments)

    # --- Tool handlers ---

    async def _handle_sec_filing(self, ticker: str, filing_type: str = "10-K") -> dict:
        result = await self._edgar.get_latest_filing(ticker, filing_type)
        if result:
            # Truncate text for tool response
            result["text"] = result["text"][:10000]
        return result or {"error": f"No {filing_type} found for {ticker}"}

    async def _handle_news_search(self, query: str, max_results: int = 10) -> dict:
        articles = await self._news.search_news(query, max_results)
        return {"articles": articles, "count": len(articles)}

    async def _handle_stock_data(self, ticker: str) -> dict:
        info = await self._yf.get_stock_info(ticker)
        return info or {"error": f"Could not fetch data for {ticker}"}

    async def _handle_price_history(self, ticker: str, period: str = "6mo") -> dict:
        history = await self._yf.get_price_history(ticker, period)
        return history or {"error": f"Could not fetch history for {ticker}"}
