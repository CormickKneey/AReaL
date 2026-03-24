from __future__ import annotations

import copy
import html
import re
from typing import Any

from .types import Message, Part


def part_to_text(part: Part) -> str:
    if part.text is not None:
        return part.text
    if part.data is not None:
        return str(part.data)
    return ""


def message_to_text(message: Message) -> str:
    return "".join(part_to_text(p) for p in message.parts)


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, Message):
        return message_to_text(content)
    if isinstance(content, Part):
        return part_to_text(content)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, Part):
                texts.append(part_to_text(block))
            elif isinstance(block, dict):
                block_type = block.get("type")
                if block_type in {"text", "input_text", "output_text"}:
                    text = block.get("text")
                    if isinstance(text, str):
                        texts.append(text)
            elif isinstance(block, str):
                texts.append(block)
        return "".join(texts)
    return str(content)


def normalize_input_blocks(content: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"text", "input_text"}:
            normalized.append({"type": "input_text", "text": part.get("text", "")})
            continue
        if part_type == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                detail = image_url.get("detail") or part.get("detail", "auto")
            else:
                url = image_url
                detail = part.get("detail", "auto")
            if url:
                normalized.append(
                    {"type": "input_image", "image_url": url, "detail": detail}
                )
            continue
        if part_type == "input_image":
            normalized.append(copy.deepcopy(part))
            continue
        if "text" in part and isinstance(part["text"], str):
            normalized.append({"type": "input_text", "text": part["text"]})
            continue
        normalized.append(copy.deepcopy(part))
    return normalized


def extract_message_content(
    input_items: list[dict[str, Any]], instructions: str
) -> str | list[dict[str, Any]]:
    text_parts: list[str] = []
    content_parts: list[dict[str, Any]] = []
    has_non_text_content = False

    if instructions:
        text_parts.append(instructions)
        content_parts.append({"type": "input_text", "text": instructions})

    for item in input_items:
        if item.get("type") == "message":
            content = item.get("content", "")
            if isinstance(content, list):
                normalized = normalize_input_blocks(content)
                content_parts.extend(normalized)
                text_parts.append(content_to_text(normalized))
                has_non_text_content = has_non_text_content or any(
                    block.get("type") != "input_text" for block in normalized
                )
            elif isinstance(content, str):
                text_parts.append(content)
                content_parts.append({"type": "input_text", "text": content})
        elif item.get("type") == "function_call_output":
            tool_text = f"[tool result] {item.get('output', '')}"
            text_parts.append(tool_text)
            content_parts.append({"type": "input_text", "text": tool_text})

    if has_non_text_content:
        return content_parts
    return "\n".join(part for part in text_parts if part)


def extract_image_attachments(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, Message):
        blocks: list[dict[str, Any]] = []
        for part in content.parts:
            if part.file_url and part.media_type.startswith("image/"):
                blocks.append({"type": "input_image", "image_url": part.file_url})
        return blocks
    if not isinstance(content, list):
        return []
    attachments: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in {"image_url", "input_image"}:
            normalized = normalize_input_blocks([block])
            attachments.extend(normalized)
    return attachments


def html_to_text(content: str) -> str:
    if not content:
        return ""
    stripped = re.sub(
        r"<style.*?</style>", " ", content, flags=re.DOTALL | re.IGNORECASE
    )
    stripped = re.sub(
        r"<script.*?</script>", " ", stripped, flags=re.DOTALL | re.IGNORECASE
    )
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    stripped = html.unescape(stripped)
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.strip()
