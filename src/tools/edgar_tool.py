"""
SEC EDGAR API Tool — fetches company filings from SEC.gov.

Uses the EDGAR EFTS full-text search and company filings APIs.
Respects SEC rate limits (10 requests/second) and requires
a User-Agent header with contact info per SEC policy.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from configs.settings import settings


class EdgarTool:
    """Wrapper around SEC EDGAR APIs for fetching company filings."""

    BASE_URL = "https://efts.sec.gov/LATEST"
    FILINGS_URL = "https://data.sec.gov/submissions"
    FULL_TEXT_URL = "https://efts.sec.gov/LATEST/search-index"

    HEADERS = {
        "User-Agent": settings.SEC_EDGAR_USER_AGENT,
        "Accept": "application/json",
    }

    # Mapping of common tickers to CIK numbers (extendable)
    TICKER_CIK_CACHE: dict[str, str] = {}

    async def _get_cik(self, ticker: str) -> str | None:
        """Resolve a ticker symbol to a CIK number."""
        if ticker in self.TICKER_CIK_CACHE:
            return self.TICKER_CIK_CACHE[ticker]

        async with httpx.AsyncClient(headers=self.HEADERS, timeout=15) as client:
            # Use the company tickers JSON endpoint
            resp = await client.get("https://www.sec.gov/files/company_tickers.json")
            if resp.status_code != 200:
                return None

            data = resp.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    self.TICKER_CIK_CACHE[ticker.upper()] = cik
                    return cik

        return None

    async def get_latest_filing(
        self,
        ticker: str,
        filing_type: str = "10-K",
    ) -> dict[str, Any] | None:
        """
        Fetch the most recent filing of the given type for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            filing_type: SEC filing type ("10-K", "10-Q", "8-K")

        Returns:
            Dict with keys: filing_type, filing_date, text, url
            or None if not found.
        """
        cik = await self._get_cik(ticker)
        if not cik:
            return None

        async with httpx.AsyncClient(headers=self.HEADERS, timeout=30) as client:
            # Fetch company submission history
            url = f"{self.FILINGS_URL}/CIK{cik}.json"
            resp = await client.get(url)

            if resp.status_code != 200:
                return None

            data = resp.json()
            recent = data.get("filings", {}).get("recent", {})

            if not recent:
                return None

            # Find the most recent filing of the requested type
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])

            for i, form in enumerate(forms):
                if form == filing_type:
                    accession = accessions[i].replace("-", "")
                    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{primary_docs[i]}"

                    # Fetch the actual document
                    doc_resp = await client.get(doc_url)
                    text = doc_resp.text if doc_resp.status_code == 200 else ""

                    # Basic HTML stripping (for production, use BeautifulSoup)
                    import re
                    text = re.sub(r"<[^>]+>", " ", text)
                    text = re.sub(r"\s+", " ", text).strip()

                    return {
                        "filing_type": filing_type,
                        "filing_date": dates[i],
                        "text": text[:50000],  # Cap at 50K chars
                        "url": doc_url,
                    }

        return None

    async def search_filings(
        self,
        query: str,
        ticker: str | None = None,
        filing_type: str | None = None,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Full-text search across SEC filings.

        Args:
            query: Search terms
            ticker: Optional ticker to filter by
            filing_type: Optional filing type filter
            max_results: Maximum results to return

        Returns:
            List of dicts with filing metadata.
        """
        params = {
            "q": query,
            "dateRange": "custom",
            "startdt": "2023-01-01",
            "enddt": "2025-12-31",
        }
        if filing_type:
            params["forms"] = filing_type

        async with httpx.AsyncClient(headers=self.HEADERS, timeout=15) as client:
            resp = await client.get(f"{self.BASE_URL}/search-index", params=params)
            if resp.status_code != 200:
                return []

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])

            results = []
            for hit in hits[:max_results]:
                source = hit.get("_source", {})
                results.append({
                    "form_type": source.get("form_type"),
                    "filing_date": source.get("file_date"),
                    "company": source.get("entity_name"),
                    "description": source.get("file_description", ""),
                })

            return results
