# 🏦 Multi-Agent Financial Research Assistant

An **agentic AI system** where specialized agents collaborate to produce comprehensive financial research memos. Built with **LangGraph** for orchestration, **Model Context Protocol (MCP)** for standardized tool-calling, and **LangSmith** for full observability.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🎯 Overview

Traditional financial research requires analysts to manually gather data from multiple sources, perform quantitative analysis, assess market sentiment, and synthesize findings into a coherent report. This project automates that workflow using a **multi-agent architecture** where each agent specializes in a distinct research capability.

### What It Does

Given a natural language query like *"Analyze Apple's financial health and recent market sentiment"*, the system:

1. **Parses intent** and creates a research plan
2. **Dispatches specialized agents** in parallel
3. **Retrieves SEC filings** and extracts key financial metrics
4. **Analyzes news sentiment** across recent articles
5. **Performs quantitative analysis** on financial ratios and trends
6. **Synthesizes findings** into a structured investment research memo
7. **Traces every step** via LangSmith for full auditability

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                         │
│              (Chat + Research Dashboard)                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               LangGraph Orchestrator                    │
│         (State Machine + Routing Logic)                 │
│                                                         │
│   ┌──────────┐   ┌──────────┐   ┌───────────────┐      │
│   │ Planning  │──▶│ Dispatch │──▶│  Synthesize   │      │
│   │  Node     │   │  Node    │   │  Node         │      │
│   └──────────┘   └────┬─────┘   └───────────────┘      │
│                       │                                  │
│              ┌────────┼────────┐                         │
│              ▼        ▼        ▼                         │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│   │   SEC    │ │  News    │ │  Quant   │               │
│   │  Filing  │ │Sentiment │ │ Analysis │               │
│   │  Agent   │ │  Agent   │ │  Agent   │               │
│   └────┬─────┘ └────┬─────┘ └────┬─────┘               │
│        │             │            │                      │
└────────┼─────────────┼────────────┼──────────────────────┘
         │             │            │
         ▼             ▼            ▼
  ┌────────────┐ ┌──────────┐ ┌──────────────┐
  │ EDGAR API  │ │ News API │ │ Yahoo Finance│
  │ (SEC.gov)  │ │ + LLM    │ │ + Metrics    │
  │            │ │ Sentiment│ │ Engine       │
  └────────────┘ └──────────┘ └──────────────┘

  ◄──────── LangSmith Tracing (All Steps) ────────►
  ◄──────── FAISS Vector Store (Caching)  ────────►
```

### Agent Descriptions

| Agent | Responsibility | Tools Used |
|-------|---------------|------------|
| **Orchestrator** | Parses user query, creates research plan, routes to agents, synthesizes final report | LangGraph state machine |
| **SEC Filing Agent** | Fetches 10-K/10-Q filings from EDGAR, extracts financial statements and risk factors | EDGAR API, document parser |
| **News Sentiment Agent** | Retrieves recent financial news, performs sentiment analysis, identifies key themes | News API, LLM-based sentiment |
| **Quantitative Agent** | Calculates financial ratios, analyzes price trends, computes risk metrics | Yahoo Finance, pandas, numpy |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Agent Orchestration** | LangGraph |
| **LLM** | OpenAI GPT-4o / Anthropic Claude (configurable) |
| **Tool Protocol** | Model Context Protocol (MCP) |
| **Observability** | LangSmith |
| **Vector Store** | FAISS |
| **Data Sources** | SEC EDGAR API, Yahoo Finance, NewsAPI |
| **Frontend** | Streamlit |
| **Containerization** | Docker + Docker Compose |
| **Testing** | pytest |

---

## 📁 Project Structure

```
multi-agent-financial-analyst/
├── README.md
├── LICENSE
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── app.py                          # Streamlit entry point
├── configs/
│   └── settings.py                 # Centralized configuration
├── src/
│   ├── __init__.py
│   ├── graph.py                    # LangGraph orchestration (state machine)
│   ├── state.py                    # Shared state schema
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── planner.py              # Intent parsing + research plan
│   │   ├── sec_filing_agent.py     # SEC EDGAR data retrieval
│   │   ├── news_sentiment_agent.py # News fetching + sentiment
│   │   ├── quant_agent.py          # Financial ratio + price analysis
│   │   └── synthesizer.py          # Final report generation
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── edgar_tool.py           # SEC EDGAR API wrapper
│   │   ├── news_tool.py            # NewsAPI wrapper
│   │   ├── yahoo_finance_tool.py   # Yahoo Finance wrapper
│   │   └── mcp_server.py           # MCP tool server
│   └── utils/
│       ├── __init__.py
│       ├── embeddings.py           # FAISS vector store utilities
│       ├── prompts.py              # Prompt templates
│       └── formatters.py           # Output formatting helpers
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_tools.py
│   └── test_graph.py
├── docs/
│   ├── ARCHITECTURE.md
│   └── DESIGN_DECISIONS.md
└── data/
    └── .gitkeep
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key or Anthropic API key
- (Optional) NewsAPI key for live news
- (Optional) LangSmith API key for tracing

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multi-agent-financial-analyst.git
cd multi-agent-financial-analyst

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Run the App

```bash
# Start the Streamlit interface
streamlit run app.py

# Or run a research query from CLI
python -m src.graph --query "Analyze Tesla's financial health and market outlook"
```

### Docker

```bash
docker-compose up --build
# Access at http://localhost:8501
```

---

## 📊 Example Output

**Query:** *"Analyze Apple's financial position and recent market sentiment"*

The system produces a structured research memo containing:

- **Executive Summary** — One-paragraph synthesis of all findings
- **Financial Health** — Key metrics from latest 10-K/10-Q (revenue, margins, debt ratios)
- **Sentiment Analysis** — Aggregated sentiment score with key themes from recent news
- **Quantitative Metrics** — P/E ratio, EPS trend, volatility, moving averages
- **Risk Factors** — Top risk factors extracted from SEC filings
- **Conclusion** — Overall assessment with confidence level

---

## 🔍 LangSmith Tracing

Every research workflow is fully traced in LangSmith, providing:

- Step-by-step agent execution timeline
- Token usage per agent
- Latency breakdown by node
- Tool call inputs/outputs
- Error tracking and retry visibility

Set `LANGCHAIN_TRACING_V2=true` in your `.env` to enable.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_agents.py -v
```

---

## 🧠 Design Decisions

See [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) for detailed rationale on:

- Why LangGraph over CrewAI/AutoGen for orchestration
- Parallel vs. sequential agent dispatch tradeoffs
- FAISS caching strategy for repeated queries
- MCP tool-calling standardization approach
- Error handling and graceful degradation patterns

---

## 🗺️ Roadmap

- [ ] Add PDF parsing for uploaded financial documents
- [ ] Implement RAG over historical research memos
- [ ] Add streaming responses in Streamlit
- [ ] Support for comparing multiple tickers
- [ ] Add GraphRAG for entity relationship extraction
- [ ] Integrate real-time market data via WebSocket
- [ ] Add backtesting agent for historical validation

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- [SEC EDGAR](https://www.sec.gov/edgar) for public financial filings
- [Yahoo Finance](https://finance.yahoo.com/) for market data
- [LangSmith](https://smith.langchain.com/) for observability
