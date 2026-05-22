"""Test: post_tool_call plugin hook fires from both tool execution paths.

Background
----------
``agent/tool_executor.py`` was missing ``post_tool_call`` hook invocations.
The hook is documented in ``hermes_cli/plugins.py`` (VALID_HOOKS) and was
already fired from the legacy ``model_tools.handle_function_call`` path, but
the primary sequential and concurrent execution paths in ``tool_executor.py``
never called it. This meant plugins relying on ``post_tool_call``
(e.g. tts-autoplay, disk-cleanup inspection) were silently skipped for all
normal agent tool calls.

This suite verifies that both the sequential (execute_tool_calls_sequential)
and concurrent (execute_tool_calls_concurrent) paths fire the hook with the
correct kwargs after a successful tool execution.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch


def _make_agent(session_id="test-session"):
    """Return a minimal AIAgent stub sufficient for tool_executor tests."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    return agent


def _make_tool_call(name="terminal", args=None, call_id="call_abc"):
    tc = MagicMock()
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(args or {"command": "echo hi"})
    return tc


class _FakeAssistantMsg:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


def _make_assistant_message(tool_calls):
    return _FakeAssistantMsg(tool_calls)


class TestPostToolCallHookSequential:
    """post_tool_call fires in the sequential execution path."""

    def test_hook_fires_on_success(self):
        from agent.tool_executor import execute_tool_calls_sequential

        agent = _make_agent()
        tool_call = _make_tool_call(name="terminal", args={"command": "echo hi"})
        assistant_message = _make_assistant_message([tool_call])
        tool_result = json.dumps({"output": "hi"})
        fired = []

        with patch("hermes_cli.plugins.invoke_hook",
                   side_effect=lambda name, **kw: fired.append((name, kw))), \
             patch("run_agent.handle_function_call", return_value=tool_result), \
             patch.object(agent, "_record_file_mutation_result"), \
             patch.object(agent, "_touch_activity"), \
             patch.object(agent, "_append_guardrail_observation",
                          side_effect=lambda fn, fa, fr, failed: fr), \
             patch.object(agent, "_tool_result_content_for_active_model",
                          side_effect=lambda n, r: r), \
             patch.object(agent, "_subdirectory_hints",
                          MagicMock(check_tool_call=lambda *a: None)), \
             patch.object(agent, "_apply_pending_steer_to_tool_results"), \
             patch("agent.tool_executor._detect_tool_failure", return_value=(False, None)):
            messages = [{"role": "user", "content": "run echo hi"}, assistant_message]
            execute_tool_calls_sequential(agent, assistant_message, messages, "task-1")

        post_calls = [(n, kw) for (n, kw) in fired if n == "post_tool_call"]
        assert len(post_calls) == 1, \
            f"Expected 1 post_tool_call invocation, got {len(post_calls)}"
        _, kwargs = post_calls[0]
        assert kwargs["tool_name"] == "terminal"
        assert kwargs["args"] == {"command": "echo hi"}
        assert kwargs["result"] == tool_result
        assert kwargs["session_id"] == "test-session"
        assert "duration_ms" in kwargs

    def test_hook_does_not_fire_when_blocked(self):
        """Blocked tool calls must not fire post_tool_call."""
        from agent.tool_executor import execute_tool_calls_sequential

        agent = _make_agent()
        tool_call = _make_tool_call(name="terminal", args={"command": "rm -rf /"})
        assistant_message = _make_assistant_message([tool_call])
        fired = []

        with patch("hermes_cli.plugins.invoke_hook",
                   side_effect=lambda name, **kw: fired.append(name)), \
             patch("hermes_cli.plugins.get_pre_tool_call_block_message",
                   return_value="blocked by test"), \
             patch.object(agent, "_invoke_tool") as mock_invoke, \
             patch.object(agent, "_touch_activity"), \
             patch.object(agent, "_tool_result_content_for_active_model",
                          side_effect=lambda n, r: r), \
             patch.object(agent, "_subdirectory_hints",
                          MagicMock(check_tool_call=lambda *a: None)), \
             patch.object(agent, "_apply_pending_steer_to_tool_results"):
            messages = [{"role": "user", "content": "delete everything"}, assistant_message]
            execute_tool_calls_sequential(agent, assistant_message, messages, "task-1")

        mock_invoke.assert_not_called()
        assert "post_tool_call" not in fired


class TestPostToolCallHookConcurrent:
    """post_tool_call fires in the concurrent execution path (_run_tool worker)."""

    def test_hook_fires_on_success(self):
        from agent.tool_executor import execute_tool_calls_concurrent

        agent = _make_agent()
        tool_call = _make_tool_call(name="read_file", args={"path": "/tmp/foo.txt"})
        assistant_message = _make_assistant_message([tool_call])
        tool_result = json.dumps({"content": "hello"})
        fired = []

        with patch("hermes_cli.plugins.invoke_hook",
                   side_effect=lambda name, **kw: fired.append((name, kw))), \
             patch.object(agent, "_invoke_tool", return_value=tool_result), \
             patch.object(agent, "_record_file_mutation_result"), \
             patch.object(agent, "_touch_activity"), \
             patch.object(agent, "_append_guardrail_observation",
                          side_effect=lambda fn, fa, fr, failed: fr), \
             patch.object(agent, "_tool_result_content_for_active_model",
                          side_effect=lambda n, r: r), \
             patch.object(agent, "_subdirectory_hints",
                          MagicMock(check_tool_call=lambda *a: None)), \
             patch.object(agent, "_apply_pending_steer_to_tool_results"), \
             patch("agent.tool_executor._detect_tool_failure", return_value=(False, None)):
            messages = [{"role": "user", "content": "read foo"}, assistant_message]
            execute_tool_calls_concurrent(agent, assistant_message, messages, "task-1")

        post_calls = [(n, kw) for (n, kw) in fired if n == "post_tool_call"]
        assert len(post_calls) >= 1, \
            f"Expected at least 1 post_tool_call invocation, got {len(post_calls)}"
        _, kwargs = post_calls[0]
        assert kwargs["tool_name"] == "read_file"
        assert kwargs["result"] == tool_result
        assert kwargs["session_id"] == "test-session"
