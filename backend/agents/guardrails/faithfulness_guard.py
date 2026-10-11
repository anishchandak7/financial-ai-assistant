"""Online Runtime Citation & Faithfulness Guardrail.

Verifies that cited quotes and source attributions correspond to actual text
in the retrieved document chunks before emission to the client.
"""

from typing import List, Dict, Any, Tuple
from observability.logger import get_custom_logger

logger = get_custom_logger(name=__name__)


class FaithfulnessGuardrail:
    """In-flight verification for citation fidelity and groundedness."""

    def verify_citations(
        self,
        citations: List[Dict[str, Any]],
        retrieved_contexts: List[str]
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Verify that every citation quote matches text in the retrieved contexts.
        
        Returns:
            (verified_citations, citation_fidelity_score)
        """
        if not citations:
            return [], 1.0

        if not retrieved_contexts:
            logger.warning("Citations generated but 0 retrieved contexts available.")
            return [], 0.0

        verified: List[Dict[str, Any]] = []
        valid_count = 0

        for cit in citations:
            quote = (cit.get("quote") or "").strip().lower()
            source_idx = cit.get("source_index", 0)

            # Check primary source context
            is_valid = False
            if 0 <= source_idx < len(retrieved_contexts):
                target_text = retrieved_contexts[source_idx].lower()
                # Check verbatim or substantial substring match (first 40 chars of quote)
                sample = quote[:40] if len(quote) >= 40 else quote
                if sample in target_text:
                    is_valid = True

            # Fallback scan across all retrieved contexts
            if not is_valid and quote:
                sample = quote[:40] if len(quote) >= 40 else quote
                for ctx in retrieved_contexts:
                    if sample in ctx.lower():
                        is_valid = True
                        break

            if is_valid:
                valid_count += 1
                verified.append(cit)
            else:
                logger.warning(f"Unverified or ungrounded citation quote dropped: '{quote[:60]}...'")

        fidelity_score = round(valid_count / len(citations), 4)
        return verified, fidelity_score


faithfulness_guard = FaithfulnessGuardrail()
