from typing import Any

def extract_text_from_content(content: Any) -> str:
    """Extract text from diverse LLM message content types (str, list of dicts, etc.)."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_blocks = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_blocks.append(block["text"])
            elif isinstance(block, str):
                text_blocks.append(block)
        return "\n".join(text_blocks)
    return str(content) if content is not None else ""
