"""Retry policy for hosted VLM gateways.

499 is emitted by nginx-family reverse proxies in front of a hosted gateway when the
upstream connection drops mid-response. It is not a caller error: the same request
succeeds on retry. A rollout on the OpenArm rig died on it, so the classification is
pinned here.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from core.vlm.vlm_client import RETRYABLE_STATUS_CODES, VLMClient


def _client() -> VLMClient:
    return VLMClient(
        base_url="https://gateway.invalid/v1",
        model="m",
        api_key="k",
        timeout_s=5.0,
        max_tokens=16,
        temperature=0.0,
        provider="openai",
        max_retries=2,
        retry_base_delay_s=0.0,
        retry_max_delay_s=0.0,
    )


def _response(status: int, text: str = "closed") -> SimpleNamespace:
    return SimpleNamespace(status_code=status, text=text, headers={})


class VlmRetryClassificationTests(unittest.TestCase):
    def test_499_is_retryable(self) -> None:
        self.assertIn(499, RETRYABLE_STATUS_CODES)

    def test_call_completion_retries_a_499_then_succeeds(self) -> None:
        client = _client()
        ok = SimpleNamespace(
            status_code=200,
            text="",
            headers={},
            json=lambda: {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
            },
        )
        client.session = Mock()
        client.session.post.side_effect = [_response(499), ok]
        data = client._post_chat({"model": "m", "messages": []})
        self.assertIsInstance(data, dict)
        self.assertEqual(client.session.post.call_count, 2)

    def test_a_hard_4xx_still_fails_without_retrying(self) -> None:
        client = _client()
        client.session = Mock()
        client.session.post.return_value = _response(400, "bad request")
        with self.assertRaises(RuntimeError):
            client._post_chat({"model": "m", "messages": []})
        self.assertEqual(client.session.post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
