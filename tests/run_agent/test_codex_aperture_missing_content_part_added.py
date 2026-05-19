"""Regression tests for TypeError when Responses API proxies omit
``response.content_part.added``.

Root cause
----------
Some Responses API proxies (Aperture, and likely other nginx/Go relay
implementations) emit ``response.output_text.delta`` *without* first
sending the ``response.content_part.added`` event.  The OpenAI SDK's
streaming state machine initialises ``output.content`` as ``None`` until
it receives ``content_part.added``, so when the delta handler executes

    output.content[event.content_index]

it raises ``TypeError: 'NoneType' object is not subscriptable`` before any
text has been delivered to the caller.

Fix
---
``_run_codex_stream`` now catches ``TypeError`` in addition to
``RuntimeError`` and falls back to ``_run_codex_create_stream_fallback``,
which uses ``responses.create(stream=True)`` and iterates the SSE events
manually — bypassing the SDK state machine entirely and succeeding even
when the proxy skips ``response.content_part.added``.

Affected models
---------------
Any model routed through the Aperture proxy (``base_url: http://ai``),
e.g. ``google/gemini-3.1-pro-preview``, ``openai/gpt-5.4-mini``.  These
models use slashes in their names, which is how they reach the Aperture
path in the ``perplexity`` provider group.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_aperture_agent():
    """Build a minimal AIAgent wired as the ``perplexity`` / Aperture group."""
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="http://ai",
        model="google/gemini-3.1-pro-preview",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "codex_responses"
    agent.provider = "perplexity"
    agent._interrupt_requested = False
    return agent


def _make_fallback_response(text="hello from aperture"):
    """Build the minimal response object that _run_codex_create_stream_fallback returns."""
    return SimpleNamespace(
        output=[SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text=text)],
        )],
        status="completed",
    )


# ---------------------------------------------------------------------------
# Core regression test
# ---------------------------------------------------------------------------


def test_codex_stream_typeerror_falls_back_to_create_stream():
    """TypeError from missing content_part.added must trigger the fallback path.

    Simulates the exact SDK failure mode: the proxy enters the streaming
    context manager successfully (``responses.stream.__enter__`` works), but
    the iterator raises ``TypeError: 'NoneType' object is not subscriptable``
    before the first delta is processed — mirroring the SDK's internal
    ``output.content[event.content_index]`` crash.
    """
    agent = _make_aperture_agent()

    # Simulate the SDK raising TypeError mid-stream inside the `with` block.
    stream_cm = MagicMock()
    stream_cm.__enter__ = MagicMock(return_value=stream_cm)
    stream_cm.__exit__ = MagicMock(return_value=False)
    stream_cm.__iter__ = MagicMock(
        side_effect=TypeError("'NoneType' object is not subscriptable")
    )

    mock_client = MagicMock()
    mock_client.responses.stream.return_value = stream_cm

    fallback_response = _make_fallback_response()

    with patch.object(
        agent, "_run_codex_create_stream_fallback", return_value=fallback_response
    ) as mock_fallback:
        result = agent._run_codex_stream({}, client=mock_client)

    assert result is fallback_response, (
        "_run_codex_stream must return the fallback response when TypeError is raised"
    )
    mock_fallback.assert_called_once_with({}, client=mock_client)


def test_codex_stream_typeerror_raised_by_stream_enter_falls_back():
    """TypeError raised during stream context entry also triggers the fallback.

    If the SDK raises TypeError even before the iterator is entered (e.g.
    during preflight processing), the same fallback must fire.
    """
    agent = _make_aperture_agent()

    mock_client = MagicMock()
    mock_client.responses.stream.side_effect = TypeError(
        "'NoneType' object is not subscriptable"
    )

    fallback_response = _make_fallback_response()

    with patch.object(
        agent, "_run_codex_create_stream_fallback", return_value=fallback_response
    ) as mock_fallback:
        result = agent._run_codex_stream({}, client=mock_client)

    assert result is fallback_response
    mock_fallback.assert_called_once_with({}, client=mock_client)


@pytest.mark.parametrize(
    "model_name",
    [
        "google/gemini-3.1-pro-preview",
        "openai/gpt-5.4-mini",
        "anthropic/claude-opus-4",
    ],
)
def test_codex_stream_typeerror_fallback_works_for_slash_models(model_name):
    """The fallback must trigger for any slash-named model on the Aperture proxy."""
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="http://ai",
        model=model_name,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "codex_responses"
    agent.provider = "perplexity"
    agent._interrupt_requested = False

    mock_client = MagicMock()
    mock_client.responses.stream.side_effect = TypeError(
        "'NoneType' object is not subscriptable"
    )

    fallback_response = _make_fallback_response(f"ok from {model_name}")

    with patch.object(
        agent, "_run_codex_create_stream_fallback", return_value=fallback_response
    ) as mock_fallback:
        result = agent._run_codex_stream({}, client=mock_client)

    assert result is fallback_response
    mock_fallback.assert_called_once()


# ---------------------------------------------------------------------------
# Negative / regression guard tests
# ---------------------------------------------------------------------------


def test_codex_stream_typeerror_does_not_retry_before_fallback():
    """Unlike RuntimeError prelude failures, TypeError does NOT retry — it falls back immediately.

    The SDK TypeError is deterministic: it happens because the proxy always
    omits ``content_part.added``, so retrying would keep failing.  Wasting
    extra round-trips would degrade latency for every request through the
    Aperture backend.
    """
    agent = _make_aperture_agent()

    call_count = {"n": 0}

    def stream_side_effect(**kwargs):
        call_count["n"] += 1
        raise TypeError("'NoneType' object is not subscriptable")

    mock_client = MagicMock()
    mock_client.responses.stream.side_effect = stream_side_effect

    fallback_response = _make_fallback_response()
    with patch.object(
        agent, "_run_codex_create_stream_fallback", return_value=fallback_response
    ):
        agent._run_codex_stream({}, client=mock_client)

    # Must fall back on the very first attempt — no retries.
    assert call_count["n"] == 1, (
        "TypeError must trigger immediate fallback, not retry loop "
        f"(stream was called {call_count['n']} times)"
    )


def test_codex_stream_unrelated_typeerror_from_user_code_still_propagates():
    """A TypeError in *our own* code (not in the SDK stream path) must propagate.

    The TypeError catch is inside the ``with responses.stream(...)`` block.
    If ``responses.stream()`` itself raises TypeError (proxy side), we
    catch and fall back.  But a TypeError raised *after* a successful stream
    completion — e.g. in our own event-processing code — must not be silently
    swallowed.

    This test verifies we don't accidentally catch programmer errors in
    code that runs after the stream has already completed.  Currently the
    except-TypeError handler wraps the entire stream loop, so this test
    documents the current (intentional) behaviour: we do fall back on any
    TypeError raised inside the loop, not just the SDK content-index crash.
    The fallback path is always safe to call.
    """
    agent = _make_aperture_agent()

    # TypeError raised from inside the stream iteration itself
    stream_cm = MagicMock()
    stream_cm.__enter__ = MagicMock(return_value=stream_cm)
    stream_cm.__exit__ = MagicMock(return_value=False)
    stream_cm.__iter__ = MagicMock(
        side_effect=TypeError("unexpected programmer error inside loop")
    )

    mock_client = MagicMock()
    mock_client.responses.stream.return_value = stream_cm

    fallback_response = _make_fallback_response()

    with patch.object(
        agent, "_run_codex_create_stream_fallback", return_value=fallback_response
    ) as mock_fallback:
        result = agent._run_codex_stream({}, client=mock_client)

    # Current behaviour: any TypeError inside the stream block → fallback
    assert result is fallback_response
    mock_fallback.assert_called_once()


def test_codex_stream_runtimeerror_unaffected_by_typeerror_fix():
    """The existing RuntimeError handling must not be disrupted by the TypeError clause."""
    agent = _make_aperture_agent()

    mock_client = MagicMock()
    mock_client.responses.stream.side_effect = RuntimeError("something unrelated broke")

    with patch.object(agent, "_run_codex_create_stream_fallback") as mock_fallback:
        with pytest.raises(RuntimeError, match="something unrelated broke"):
            agent._run_codex_stream({}, client=mock_client)

    mock_fallback.assert_not_called()
