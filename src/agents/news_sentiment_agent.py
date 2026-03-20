"""
News Sentiment Agent — fetches recent financial news and performs sentiment analysis.

Uses NewsAPI (or fallback mock data) to retrieve articles, then leverages
an LLM for nuanced financial sentiment classification beyond simple
positive/negative polarity.
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from configs.settings import get_llm
from src.state import AgentState, NewsSentimentData
from src.tools.news_tool import NewsTool

SENTIMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a financial sentiment analyst. Analyze the following news articles
about {company} and produce a sentiment report as JSON:

{{
    "overall_sentiment": <float from -1.0 (very bearish) to 1.0 (very bullish)>,
    "sentiment_label": "<Bullish|Somewhat Bullish|Neutral|Somewhat Bearish|Bearish>",
    "key_themes": [<top 3-5 recurring themes across articles>],
    "article_sentiments": [
        {{
            "title": "<article title>",
            "source": "<source name>",
            "sentiment": "<Bullish|Neutral|Bearish>",
            "summary": "<1-sentence summary of relevance>"
        }}
    ]
}}

Consider financial context: earnings beats, analyst upgrades, regulatory actions,
product launches, lawsuits, and macroeconomic factors. Be specific about WHY
the sentiment is what it is.
""",
        ),
        ("human", "Articles:\n{articles_text}"),
    ]
)


async def news_sentiment_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: Fetch news and analyze sentiment.

    Reads: research_plan (ticker, company_name)
    Writes: news_data, messages, agent_trace, errors
    """
    start = time.time()
    plan = state.get("research_plan")

    if not plan:
        return {
            "news_data": None,
            "errors": ["News Sentiment Agent: No research plan available"],
            "agent_trace": [{"agent": "news_sentiment", "status": "skipped", "duration_sec": 0}],
        }

    ticker = plan["ticker"]
    company = plan.get("company_name", ticker)

    try:
        # Fetch recent news
        news_tool = NewsTool()
        articles = await news_tool.search_news(
            query=f"{company} {ticker} stock",
            max_results=10,
        )

        if not articles:
            return {
                "news_data": {
                    "articles_analyzed": 0,
                    "overall_sentiment": 0.0,
                    "sentiment_label": "Neutral",
                    "key_themes": ["No recent news found"],
                    "top_articles": [],
                },
                "messages": [AIMessage(content=f"No recent news articles found for {company}.")],
                "agent_trace": [{"agent": "news_sentiment", "status": "no_data", "duration_sec": round(time.time() - start, 2)}],
            }

        # Format articles for LLM analysis
        articles_text = "\n\n".join(
            f"Title: {a['title']}\nSource: {a['source']}\nDate: {a.get('date', 'N/A')}\nSnippet: {a.get('snippet', 'N/A')}"
            for a in articles
        )

        llm = get_llm()
        chain = SENTIMENT_PROMPT | llm
        response = await chain.ainvoke({
            "company": company,
            "articles_text": articles_text,
        })

        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]

        result = json.loads(content.strip())

        news_data: NewsSentimentData = {
            "articles_analyzed": len(articles),
            "overall_sentiment": result.get("overall_sentiment", 0.0),
            "sentiment_label": result.get("sentiment_label", "Neutral"),
            "key_themes": result.get("key_themes", []),
            "top_articles": result.get("article_sentiments", [])[:5],
        }

        return {
            "news_data": news_data,
            "messages": [AIMessage(content=f"Sentiment analysis complete: {news_data['sentiment_label']} ({news_data['articles_analyzed']} articles analyzed).")],
            "agent_trace": [{"agent": "news_sentiment", "status": "success", "articles_count": len(articles), "duration_sec": round(time.time() - start, 2)}],
        }

    except Exception as e:
        return {
            "news_data": None,
            "errors": [f"News Sentiment Agent error: {e}"],
            "messages": [AIMessage(content=f"News sentiment analysis failed: {e}")],
            "agent_trace": [{"agent": "news_sentiment", "status": "error", "error": str(e), "duration_sec": round(time.time() - start, 2)}],
        }
