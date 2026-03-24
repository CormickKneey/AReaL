"""Tests for multimodal message handling in the OpenResponses bridge."""

from __future__ import annotations

from areal.experimental.agent_service.gateway.bridge import OpenResponsesBridge


class TestExtractMessage:
    def test_text_only_content_collapses_to_string(self):
        message = OpenResponsesBridge._extract_message(
            [
                {
                    "type": "message",
                    "content": [{"type": "input_text", "text": "hello world"}],
                }
            ],
            instructions="",
        )
        assert message == "hello world"

    def test_multimodal_content_preserves_blocks(self):
        message = OpenResponsesBridge._extract_message(
            [
                {
                    "type": "message",
                    "content": [
                        {"type": "input_text", "text": "describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/cat.png"},
                        },
                    ],
                }
            ],
            instructions="",
        )
        assert isinstance(message, list)
        assert message[0] == {"type": "input_text", "text": "describe this image"}
        assert message[1]["type"] == "input_image"
        assert message[1]["image_url"] == "https://example.com/cat.png"

    def test_instructions_are_kept_for_multimodal_content(self):
        message = OpenResponsesBridge._extract_message(
            [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": "https://example.com/chart.png",
                        }
                    ],
                }
            ],
            instructions="analyze this chart",
        )
        assert isinstance(message, list)
        assert message[0] == {"type": "input_text", "text": "analyze this chart"}
        assert message[1]["type"] == "input_image"
