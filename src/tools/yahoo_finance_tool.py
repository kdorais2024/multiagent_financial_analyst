"""
Yahoo Finance Tool — fetches stock prices, fundamentals, and computes metrics.

Uses the yfinance library for market data retrieval.
Computations (moving averages, volatility) are done with pandas/numpy
for reliability — no LLM calls in the data pipeline.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from configs.settings import settings

# Thread pool for running sync yfinance calls in async context
_executor = ThreadPoolExecutor(max_workers=4)


class YahooFinanceTool:
    """Wrapper around yfinance for stock data retrieval and metric computation."""

    async def get_stock_info(self, ticker: str) -> dict[str, Any] | None:
        """
        Fetch fundamental stock information.

        Returns dict with: current_price, market_cap, pe_ratio, eps,
        fifty_two_week_high/low, debt_to_equity, return_on_equity
        """
        try:
            import yfinance as yf

            def _fetch():
                stock = yf.Ticker(ticker)
                info = stock.info
                return {
                    "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "eps": info.get("trailingEps"),
                    "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                    "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "return_on_equity": info.get("returnOnEquity"),
                    "dividend_yield": info.get("dividendYield"),
                    "beta": info.get("beta"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                }

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(_executor, _fetch)

        except Exception:
            return self._mock_stock_info(ticker)

    async def get_price_history(
        self,
        ticker: str,
        period: str = "6mo",
    ) -> dict[str, Any] | None:
        """
        Fetch historical price data and compute technical indicators.

        Args:
            ticker: Stock ticker
            period: yfinance period string ("1mo", "3mo", "6mo", "1y")

        Returns dict with: ma_50, ma_200, volatility_30d, price_change_pct
        """
        try:
            import yfinance as yf

            def _fetch():
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)

                if hist.empty:
                    return None

                close = hist["Close"]

                # Moving averages
                ma_50 = float(close.rolling(window=50).mean().iloc[-1]) if len(close) >= 50 else None
                ma_200 = float(close.rolling(window=200).mean().iloc[-1]) if len(close) >= 200 else None

                # 30-day annualized volatility
                returns = close.pct_change().dropna()
                vol_30d = None
                if len(returns) >= 30:
                    vol_30d = float(returns.tail(30).std() * np.sqrt(252) * 100)  # Annualized %

                # Price change
                price_change_pct = None
                if len(close) >= 2:
                    price_change_pct = float((close.iloc[-1] / close.iloc[0] - 1) * 100)

                return {
                    "ma_50": round(ma_50, 2) if ma_50 else None,
                    "ma_200": round(ma_200, 2) if ma_200 else None,
                    "volatility_30d": round(vol_30d, 2) if vol_30d else None,
                    "price_change_pct": round(price_change_pct, 2) if price_change_pct else None,
                    "data_points": len(close),
                }

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(_executor, _fetch)

        except Exception:
            return self._mock_price_history(ticker)

    @staticmethod
    def _mock_stock_info(ticker: str) -> dict[str, Any]:
        """Mock data for development without yfinance."""
        return {
            "current_price": 185.50,
            "market_cap": 2850000000000,
            "pe_ratio": 29.5,
            "forward_pe": 27.8,
            "eps": 6.29,
            "fifty_two_week_high": 199.62,
            "fifty_two_week_low": 155.98,
            "debt_to_equity": 176.3,
            "return_on_equity": 1.56,
            "dividend_yield": 0.0055,
            "beta": 1.24,
            "sector": "Technology",
            "industry": "Consumer Electronics",
        }

    @staticmethod
    def _mock_price_history(ticker: str) -> dict[str, Any]:
        """Mock price history for development."""
        return {
            "ma_50": 182.30,
            "ma_200": 175.60,
            "volatility_30d": 22.5,
            "price_change_pct": 12.3,
            "data_points": 126,
        }
