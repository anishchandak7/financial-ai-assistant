"""LangChain/LangGraph-native Compliance and Regulatory Guardrail.

Implements an Output Guardrail Runnable enforcing:
1. Regulatory financial disclaimer attachment for investment analysis and market insight specialists.
2. Sanitization of high-risk unlicensed guarantees (e.g. 'guaranteed return', '100% risk-free').
3. LCEL Runnable interface for seamless composition in LangGraph workflows.
"""

import re
from typing import Optional, Dict, Any, Union
from langchain_core.runnables import RunnableSerializable, RunnableLambda
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)

FINANCIAL_DISCLAIMER = (
    "\n\n---\n*Disclaimer: Market data, technical indicators, and quantitative insights provided by "
    "this assistant are strictly for informational and educational purposes only. They do not constitute "
    "financial, investment, tax, or legal advice. Always conduct independent research or consult a "
    "licensed financial advisor before making investment decisions.*"
)

ADVICE_PROHIBITION_PATTERNS = [
    r"guaranteed\s+(return|profit|gains)",
    r"you\s+must\s+buy\s+this\s+now",
    r"100%\s+risk[ -]free",
]
COMPILED_ADVICE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in ADVICE_PROHIBITION_PATTERNS]


class ComplianceEnforcer(RunnableSerializable):
    """LangChain-native output compliance enforcer for financial assistant responses."""

    def attach_disclaimer(self, response_text: str, agent_name: str) -> str:
        """Appends official regulatory disclaimer to market analysis and technical reports."""
        if agent_name in ["Market Analysis Agent", "Stock Analyzer Agent", "Finance Q&A Agent"]:
            if FINANCIAL_DISCLAIMER.strip() not in response_text:
                return response_text + FINANCIAL_DISCLAIMER
        return response_text

    def sanitize_unlicensed_claims(self, response_text: str) -> str:
        """Sanitizes any hallucinated guarantees of profit or zero-risk assertions."""
        sanitized = response_text
        for pattern in COMPILED_ADVICE_PATTERNS:
            if pattern.search(sanitized):
                logger.warning("Compliance: Sanitized high-risk financial guarantee language from response.")
                sanitized = pattern.sub("[speculative market setup]", sanitized)
        return sanitized

    def enforce_on_text(self, text: str, agent_name: str) -> str:
        """Sanitizes text and attaches disclaimer."""
        sanitized = self.sanitize_unlicensed_claims(text)
        return self.attach_disclaimer(sanitized, agent_name)

    def invoke(self, input: Union[Dict[str, Any], str], config: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], str]:
        """LangChain Runnable interface support for either state dictionaries or direct text."""
        if isinstance(input, dict):
            # Process state dictionary containing 'agent_output'
            agent_output = input.get("agent_output", {})
            for agent_name, out in agent_output.items():
                if isinstance(out, dict) and "response" in out:
                    out["response"] = self.enforce_on_text(out["response"], agent_name)
                elif isinstance(out, str):
                    agent_output[agent_name] = self.enforce_on_text(out, agent_name)
            input["agent_output"] = agent_output
            return input
        elif isinstance(input, str):
            return self.enforce_on_text(input, "Market Analysis Agent")
        return input


# Singleton instance and LCEL Runnable export
compliance_enforcer = ComplianceEnforcer()
compliance_runnable = RunnableLambda(compliance_enforcer.invoke).with_config(run_name="output_compliance_runnable")
