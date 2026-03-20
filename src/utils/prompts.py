"""
Prompt templates and constants used across agents.

Centralizing prompts here makes them easy to iterate on,
version control, and A/B test.
"""

# System-level persona used across all agents
FINANCIAL_ANALYST_PERSONA = (
    "You are a senior financial research analyst at a top-tier investment bank. "
    "You are precise with numbers, balanced in your assessments, and always "
    "distinguish between facts and opinions. You cite specific data points "
    "rather than making vague claims."
)

# Output format instructions
JSON_OUTPUT_INSTRUCTION = (
    "Respond ONLY with valid JSON. Do not include markdown code fences, "
    "explanatory text, or any content outside the JSON object."
)
