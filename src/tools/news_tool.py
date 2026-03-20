"""
News API Tool — fetches recent financial news articles.

Supports NewsAPI.org as primary source with fallback mock data
for development/testing without an API key.
"""

from __future__ import annotations

from typing import Any

import httpx

from configs.settings import settings


class NewsTool:
    """Wrapper around NewsAPI for fetching financial news."""

    BASE_URL = "https://newsapi.org/v2"

    async def search_news(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevancy",
    ) -> list[dict[str, str]]:
        """
        Search for recent news articles.

        Args:
            query: Search query (e.g., "Apple AAPL stock")
            max_results: Maximum articles to return
            sort_by: "relevancy", "publishedAt", or "popularity"

        Returns:
            List of article dicts with: title, source, date, snippet, url
        """
        api_key = settings.NEWS_API_KEY

        # If no API key, return mock data for development
        if not api_key or api_key == "your_newsapi_key_here":
            return self._mock_articles(query, max_results)

        params = {
            "q": query,
            "sortBy": sort_by,
            "language": "en",
            "pageSize": max_results,
            "apiKey": api_key,
        }

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f"{self.BASE_URL}/everything", params=params)

            if resp.status_code != 200:
                return self._mock_articles(query, max_results)

            data = resp.json()
            articles = data.get("articles", [])

            return [
                {
                    "title": a.get("title", ""),
                    "source": a.get("source", {}).get("name", "Unknown"),
                    "date": a.get("publishedAt", "")[:10],
                    "snippet": a.get("description", ""),
                    "url": a.get("url", ""),
                }
                for a in articles
                if a.get("title")
            ]

    @staticmethod
    def _mock_articles(query: str, max_results: int) -> list[dict[str, str]]:
        """
        Generate mock articles for development/testing.
        Replace with real API calls in production.
        """
        company = query.split()[0] if query else "Company"
        mock_data = [
            {
                "title": f"{company} Reports Strong Quarterly Earnings, Beating Analyst Estimates",
                "source": "Reuters",
                "date": "2025-01-15",
                "snippet": f"{company} exceeded Wall Street expectations with revenue growth of 8% year-over-year, driven by strong demand in core segments.",
                "url": "https://example.com/article1",
            },
            {
                "title": f"Analysts Upgrade {company} Stock After Product Launch",
                "source": "Bloomberg",
                "date": "2025-01-14",
                "snippet": f"Several major banks raised their price targets for {company} following the successful launch of new products.",
                "url": "https://example.com/article2",
            },
            {
                "title": f"{company} Faces Regulatory Scrutiny in European Markets",
                "source": "Financial Times",
                "date": "2025-01-13",
                "snippet": f"EU regulators are examining {company}'s market practices, which could impact operations in the region.",
                "url": "https://example.com/article3",
            },
            {
                "title": f"{company} Announces $10B Share Buyback Program",
                "source": "CNBC",
                "date": "2025-01-12",
                "snippet": f"{company}'s board approved a significant share repurchase program, signaling confidence in future growth.",
                "url": "https://example.com/article4",
            },
            {
                "title": f"Supply Chain Challenges May Impact {company}'s Q2 Outlook",
                "source": "Wall Street Journal",
                "date": "2025-01-10",
                "snippet": f"Industry analysts warn that ongoing supply chain disruptions could affect {company}'s margins next quarter.",
                "url": "https://example.com/article5",
            },
        ]
        return mock_data[:max_results]
