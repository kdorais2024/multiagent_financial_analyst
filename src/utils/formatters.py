"""
Output formatting helpers for research reports and agent data.

Provides consistent formatting for financial metrics, currency values,
and report sections across the application.
"""

from __future__ import annotations

from typing import Any


def format_currency(value: float | None, prefix: str = "$") -> str:
    """Format a number as currency with appropriate suffixes."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000_000_000:
        return f"{prefix}{value / 1_000_000_000_000:.2f}T"
    if abs(value) >= 1_000_000_000:
        return f"{prefix}{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{prefix}{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{prefix}{value / 1_000:.2f}K"
    return f"{prefix}{value:,.2f}"


def format_percentage(value: float | None, decimals: int = 2) -> str:
    """Format a decimal as a percentage string."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}%"


def format_ratio(value: float | None, decimals: int = 2) -> str:
    """Format a financial ratio."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}x"


def format_metric(label: str, value: Any, formatter: str = "auto") -> str:
    """
    Format a single metric as a label-value string.

    Args:
        label: Metric name
        value: Metric value
        formatter: "currency", "percentage", "ratio", or "auto"
    """
    if value is None:
        return f"**{label}**: N/A"

    if formatter == "currency":
        return f"**{label}**: {format_currency(value)}"
    elif formatter == "percentage":
        return f"**{label}**: {format_percentage(value)}"
    elif formatter == "ratio":
        return f"**{label}**: {format_ratio(value)}"
    else:
        # Auto-detect
        if isinstance(value, float) and abs(value) > 1_000_000:
            return f"**{label}**: {format_currency(value)}"
        elif isinstance(value, float) and abs(value) < 1:
            return f"**{label}**: {format_percentage(value * 100)}"
        return f"**{label}**: {value}"


def sentiment_badge(score: float) -> str:
    """Return a colored badge string for sentiment scores."""
    if score >= 0.3:
        return "🟢 Bullish"
    elif score >= 0.1:
        return "🟡 Slightly Bullish"
    elif score >= -0.1:
        return "⚪ Neutral"
    elif score >= -0.3:
        return "🟡 Slightly Bearish"
    else:
        return "🔴 Bearish"


def trend_arrow(trend: str) -> str:
    """Return an arrow indicator for price trends."""
    mapping = {
        "Uptrend": "📈",
        "Downtrend": "📉",
        "Sideways": "➡️",
    }
    return mapping.get(trend, "❓")
