# Copyright 2025 The game_arena Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for FreeCivLLMAgent error telemetry and structured error capture.

This test suite addresses PR feedback issue #3 (High Priority):
Missing error context in fallback (freeciv_llm_agent.py:209-226) where
error context is not captured for telemetry/debugging.

The fix adds structured error logging with:
- Exception type and message
- Stack trace capture
- Observation context (turn, player_id, state info)
- Fallback action details with reasoning
- Telemetry export for monitoring systems
"""

import traceback
import unittest
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any, List

from game_arena.harness import model_generation
from game_arena.harness.freeciv_llm_agent import FreeCivLLMAgent


class TestErrorTelemetryCapture(unittest.TestCase):
  """Test suite for error telemetry and structured error capture."""

  def setUp(self):
    """Set up test agent with mocked model."""
    self.mock_model = MagicMock(spec=model_generation.Model)
    self.mock_model.model_name = "test-model"

    self.agent = FreeCivLLMAgent(
        model=self.mock_model,
        strategy="balanced",
        use_rethinking=False,
        fallback_to_random=True,
    )

  def test_error_telemetry_captures_exception_type(self):
    """Test that error telemetry captures exception class name.

    When action generation fails, the telemetry should record:
    - Exception type (e.g., 'ValueError', 'RuntimeError', 'ConnectionError')
    - Full exception class path for debugging

    This helps identify patterns in failure modes and categorize errors
    for monitoring dashboards.
    """
    # Create observation with valid legal actions for fallback
    observation = {
        "playerID": 1,
        "turn": 5,
        "legalActions": [10, 20, 30],
        "serializedGameAndState": "test_state",
    }

    # Mock _get_action_async to raise specific exception types
    test_exceptions = [
        ValueError("Invalid game state"),
        RuntimeError("Model API timeout"),
        ConnectionError("Network error"),
    ]

    for exception in test_exceptions:
      with self.subTest(exception_type=type(exception).__name__):
        # Patch _get_action_async to raise exception
        with patch.object(self.agent, '_get_action_async', side_effect=exception):
          # Call agent (should use fallback)
          result = self.agent(observation, {})

          # Verify fallback action was returned
          self.assertIn("submission", result)
          self.assertIn(result["submission"], observation["legalActions"])

          # Verify telemetry captured exception type
          self.assertTrue(hasattr(self.agent, '_error_telemetry'))
          telemetry = self.agent._error_telemetry
          self.assertEqual(telemetry["error_type"], type(exception).__name__)
          self.assertEqual(telemetry["error_message"], str(exception))

  def test_error_telemetry_captures_stack_trace(self):
    """Test that error telemetry includes stack trace for debugging.

    Stack traces are essential for understanding:
    - Where the error originated
    - Call chain leading to the error
    - File and line numbers for debugging
    """
    observation = {
        "playerID": 1,
        "turn": 3,
        "legalActions": [1, 2, 3],
        "serializedGameAndState": "state",
    }

    # Create a nested error to generate interesting stack trace
    def inner_function():
      raise ValueError("Nested error in action generation")

    async def outer_function(obs):
      """Async function that calls inner function."""
      inner_function()

    with patch.object(self.agent, '_get_action_async', side_effect=outer_function):
      result = self.agent(observation, {})

      # Verify telemetry has stack trace
      self.assertTrue(hasattr(self.agent, '_error_telemetry'))
      telemetry = self.agent._error_telemetry

      self.assertIn("stack_trace", telemetry)
      self.assertIsInstance(telemetry["stack_trace"], str)
      self.assertGreater(len(telemetry["stack_trace"]), 0)

      # Verify stack trace contains useful info
      stack_trace = telemetry["stack_trace"]
      self.assertIn("inner_function", stack_trace)
      self.assertIn("outer_function", stack_trace)
      self.assertIn("Nested error", stack_trace)

  def test_error_telemetry_captures_observation_context(self):
    """Test that observation metadata is captured for debugging.

    When errors occur, we need context about:
    - Current turn number
    - Player ID
    - Number of legal actions available
    - Game phase (if available)

    This helps correlate errors with game state and identify patterns.
    """
    observation = {
        "playerID": 2,
        "turn": 15,
        "phase": "combat",
        "legalActions": [100, 101, 102, 103, 104],
        "serializedGameAndState": "complex_state",
        "units": [{"id": 1}, {"id": 2}],
    }

    error = RuntimeError("Action generation failed")

    with patch.object(self.agent, '_get_action_async', side_effect=error):
      result = self.agent(observation, {})

      # Verify observation context in telemetry
      self.assertTrue(hasattr(self.agent, '_error_telemetry'))
      telemetry = self.agent._error_telemetry

      self.assertIn("observation_context", telemetry)
      context = telemetry["observation_context"]

      # Check key fields are captured
      self.assertEqual(context["turn"], 15)
      self.assertEqual(context["player_id"], 2)
      self.assertEqual(context["num_legal_actions"], 5)
      self.assertEqual(context["phase"], "combat")

      # Observation keys should be logged (without full data to avoid bloat)
      self.assertIn("observation_keys", context)
      self.assertIn("playerID", context["observation_keys"])

  def test_fallback_action_logged_with_reason(self):
    """Test that fallback action selection is logged with reasoning.

    When falling back to random action, log should include:
    - Which action was chosen
    - Why it was chosen (random from legal actions)
    - Available legal actions at time of fallback
    - Whether rethinking was attempted
    """
    observation = {
        "playerID": 1,
        "turn": 7,
        "legalActions": [5, 6, 7],
        "serializedGameAndState": "state",
    }

    error = ValueError("Model timeout")

    # Patch random to make test deterministic
    with patch('random.Random.choice', return_value=6):
      with patch.object(self.agent, '_get_action_async', side_effect=error):
        result = self.agent(observation, {})

        # Verify fallback action
        self.assertEqual(result["submission"], 6)

        # Verify fallback telemetry
        self.assertTrue(hasattr(self.agent, '_error_telemetry'))
        telemetry = self.agent._error_telemetry

        self.assertIn("fallback_action", telemetry)
        fallback = telemetry["fallback_action"]

        self.assertEqual(fallback["action"], 6)
        self.assertEqual(fallback["reason"], "random_from_legal_actions")
        self.assertEqual(fallback["legal_actions"], [5, 6, 7])
        self.assertFalse(fallback["rethinking_attempted"])

  def test_error_telemetry_timestamp(self):
    """Test that error telemetry includes timestamp for time-series analysis."""
    observation = {
        "playerID": 1,
        "turn": 1,
        "legalActions": [1, 2, 3],
        "serializedGameAndState": "state",
    }

    error = RuntimeError("Test error")

    import time
    start_time = time.time()

    with patch.object(self.agent, '_get_action_async', side_effect=error):
      result = self.agent(observation, {})

    end_time = time.time()

    # Verify timestamp
    self.assertTrue(hasattr(self.agent, '_error_telemetry'))
    telemetry = self.agent._error_telemetry

    self.assertIn("timestamp", telemetry)
    self.assertIsInstance(telemetry["timestamp"], float)
    self.assertGreaterEqual(telemetry["timestamp"], start_time)
    self.assertLessEqual(telemetry["timestamp"], end_time)


class TestTelemetryExport(unittest.TestCase):
  """Test telemetry export functionality for monitoring systems."""

  def setUp(self):
    """Set up test agent."""
    self.mock_model = MagicMock(spec=model_generation.Model)
    self.mock_model.model_name = "test-model"

    self.agent = FreeCivLLMAgent(
        model=self.mock_model,
        use_rethinking=False,
        fallback_to_random=True,
    )

  def test_get_error_telemetry_returns_structured_data(self):
    """Test that error telemetry can be exported as structured data.

    Monitoring systems need to be able to export telemetry in a
    standardized format (e.g., JSON) for ingestion into metrics databases.
    """
    observation = {
        "playerID": 1,
        "turn": 10,
        "legalActions": [1, 2, 3],
        "serializedGameAndState": "state",
    }

    error = ValueError("Test error for export")

    with patch.object(self.agent, '_get_action_async', side_effect=error):
      self.agent(observation, {})

    # Export telemetry
    telemetry = self.agent.get_error_telemetry()

    # Verify structure
    self.assertIsInstance(telemetry, dict)
    self.assertIn("error_type", telemetry)
    self.assertIn("error_message", telemetry)
    self.assertIn("stack_trace", telemetry)
    self.assertIn("observation_context", telemetry)
    self.assertIn("fallback_action", telemetry)
    self.assertIn("timestamp", telemetry)

    # Verify it's JSON-serializable
    import json
    try:
      json_str = json.dumps(telemetry)
      self.assertIsInstance(json_str, str)
    except (TypeError, ValueError) as e:
      self.fail(f"Telemetry not JSON-serializable: {e}")

  def test_multiple_errors_tracked_separately(self):
    """Test that multiple errors are tracked in telemetry history.

    For production monitoring, we need to track patterns across multiple
    errors, not just the most recent one.
    """
    observation = {
        "playerID": 1,
        "turn": 1,
        "legalActions": [1, 2, 3],
        "serializedGameAndState": "state",
    }

    errors = [
        ValueError("First error"),
        RuntimeError("Second error"),
        ConnectionError("Third error"),
    ]

    for error in errors:
      with patch.object(self.agent, '_get_action_async', side_effect=error):
        self.agent(observation, {})

    # Get error history
    history = self.agent.get_error_history()

    # Verify all errors are tracked
    self.assertIsInstance(history, list)
    self.assertEqual(len(history), 3)

    # Verify error types
    error_types = [entry["error_type"] for entry in history]
    self.assertEqual(error_types, ["ValueError", "RuntimeError", "ConnectionError"])


class TestNoFallbackScenario(unittest.TestCase):
  """Test error handling when fallback is disabled."""

  def setUp(self):
    """Set up test agent with fallback disabled."""
    self.mock_model = MagicMock(spec=model_generation.Model)
    self.mock_model.model_name = "test-model"

    self.agent = FreeCivLLMAgent(
        model=self.mock_model,
        use_rethinking=False,
        fallback_to_random=False,  # Disable fallback
    )

  def test_error_telemetry_when_no_fallback(self):
    """Test that errors are still logged even when no fallback is used.

    When fallback is disabled, errors should still be captured for
    telemetry, then re-raised.
    """
    observation = {
        "playerID": 1,
        "turn": 1,
        "legalActions": [1, 2, 3],
        "serializedGameAndState": "state",
    }

    error = ValueError("Critical error with no fallback")

    with patch.object(self.agent, '_get_action_async', side_effect=error):
      # Should raise the error (no fallback)
      with self.assertRaises(ValueError):
        self.agent(observation, {})

    # But telemetry should still be captured
    self.assertTrue(hasattr(self.agent, '_error_telemetry'))
    telemetry = self.agent._error_telemetry

    self.assertEqual(telemetry["error_type"], "ValueError")
    self.assertEqual(telemetry["error_message"], "Critical error with no fallback")
    self.assertIn("stack_trace", telemetry)

    # Fallback action should indicate no fallback was used
    self.assertIn("fallback_action", telemetry)
    self.assertIsNone(telemetry["fallback_action"]["action"])
    self.assertEqual(telemetry["fallback_action"]["reason"], "no_fallback_configured")


if __name__ == "__main__":
  unittest.main()
