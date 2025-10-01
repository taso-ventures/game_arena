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

"""Game memory management system for FreeCiv LLM Agent.

This module provides memory management capabilities including:
- Action history tracking with configurable limits
- Context compression for token-aware prompt generation
- Important event extraction and summarization
- Performance-optimized caching using existing infrastructure

Example usage:
    >>> from game_arena.harness.freeciv_memory import GameMemory
    >>> from game_arena.harness.freeciv_state import FreeCivAction
    >>>
    >>> memory = GameMemory(max_size=10)
    >>> action = FreeCivAction("unit_move", 101, {"x": 2, "y": 3}, {}, "unit")
    >>> result = {"success": True, "score_delta": 5}
    >>> memory.record_action(action, result)
    >>>
    >>> context = memory.get_context(max_tokens=1000)
    >>> print(len(context))  # Context within token limit
"""

import hashlib
import json
import time
from collections import deque
from typing import Any, Dict, List, Optional

from absl import logging

from game_arena.harness.freeciv_cache import LRUCache
from game_arena.harness.freeciv_state import FreeCivAction


class TokenManager:
  """Manages token counting and truncation for different models."""

  # Approximate token limits for different models
  MODEL_LIMITS = {
      "gpt-4": 8192,
      "gpt-4.1": 16384,
      "claude-3": 100000,
      "claude-opus-4": 32000,
      "claude-sonnet-4": 64000,
      "gemini-2.5": 32768,
      "deepseek": 16384,
  }

  def __init__(self, model_name: str):
    """Initialize token manager for specific model.

    Args:
      model_name: Name of the model to manage tokens for
    """
    self.model_name = model_name
    self.limit = self._get_model_limit(model_name)
    self.tokenizer = self._load_tokenizer(model_name)

  def _get_model_limit(self, model_name: str) -> int:
    """Get token limit for model name.

    Args:
      model_name: Model name to lookup

    Returns:
      Token limit for the model
    """
    # Find matching model limit
    for pattern, limit in self.MODEL_LIMITS.items():
      if pattern in model_name.lower():
        return limit

    # Default conservative limit
    return 4096

  def _load_tokenizer(self, model_name: str):
    """Load appropriate tokenizer for the model.

    Args:
      model_name: Name of the model

    Returns:
      Tokenizer instance or None if not available
    """
    model_lower = model_name.lower()

    # Try to load OpenAI tokenizer for GPT models
    if "gpt" in model_lower:
      try:
        import tiktoken
        if "gpt-4" in model_lower:
          return tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model_lower:
          return tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
          # Default to GPT-4 for other GPT models
          return tiktoken.encoding_for_model("gpt-4")
      except ImportError:
        logging.warning("tiktoken not available for GPT tokenization")
      except Exception as e:
        logging.warning("Failed to load GPT tokenizer: %s", str(e))

    # Try to load Anthropic tokenizer for Claude models
    if "claude" in model_lower:
      try:
        import anthropic
        # Note: Anthropic doesn't provide a public tokenizer yet
        # We'll use a reasonable approximation
        logging.debug("Using approximation for Claude tokenization")
        return None
      except ImportError:
        logging.debug("anthropic package not available")

    # For Gemini models, use rough approximation
    if "gemini" in model_lower or "bison" in model_lower:
      # Google doesn't provide public tokenizers yet
      logging.debug("Using approximation for Gemini tokenization")
      return None

    # For DeepSeek models, use rough approximation
    if "deepseek" in model_lower:
      logging.debug("Using approximation for DeepSeek tokenization")
      return None

    return None

  def count_tokens(self, text: str) -> int:
    """Count tokens in text using model-specific tokenizer when available.

    Args:
      text: Text to count tokens for

    Returns:
      Token count (exact if tokenizer available, estimated otherwise)
    """
    if self.tokenizer is not None:
      try:
        # Use actual tokenizer for precise counting
        return len(self.tokenizer.encode(text))
      except Exception as e:
        logging.warning("Tokenizer failed, falling back to approximation: %s", str(e))

    # Fallback to improved approximation based on model type
    return self._approximate_token_count(text)

  def _approximate_token_count(self, text: str) -> int:
    """Improved token count approximation based on model characteristics.

    Args:
      text: Text to count tokens for

    Returns:
      Estimated token count
    """
    model_lower = self.model_name.lower()

    # Different models have different tokenization characteristics
    if "gpt" in model_lower:
      # GPT models: ~3.5-4 characters per token on average
      # More accurate for English text with punctuation
      return int(len(text) / 3.7)
    elif "claude" in model_lower:
      # Claude models: similar to GPT but slightly different
      return int(len(text) / 3.8)
    elif "gemini" in model_lower:
      # Gemini models: roughly similar to GPT
      return int(len(text) / 3.6)
    elif "deepseek" in model_lower:
      # DeepSeek: similar tokenization to GPT
      return int(len(text) / 3.7)
    else:
      # Conservative default
      return int(len(text) / 4.0)

  def truncate_to_limit(self, text: str, reserve: int = 1000) -> str:
    """Truncate text to fit within token limit with reserve.

    Args:
      text: Text to truncate
      reserve: Tokens to reserve for other prompt parts

    Returns:
      Truncated text that fits within limit
    """
    max_tokens = self.limit - reserve
    current_tokens = self.count_tokens(text)

    if current_tokens <= max_tokens:
      return text

    # Use binary search for precise truncation when tokenizer available
    if self.tokenizer is not None:
      return self._binary_search_truncate(text, max_tokens)

    # Fallback to character-based approximation
    return self._approximate_truncate(text, max_tokens, current_tokens)

  def _binary_search_truncate(self, text: str, max_tokens: int) -> str:
    """Use binary search to find optimal truncation point with exact tokenizer.

    Args:
      text: Text to truncate
      max_tokens: Maximum tokens allowed

    Returns:
      Truncated text with exact token count
    """
    if not text:
      return text

    left, right = 0, len(text)
    best_end = 0

    while left <= right:
      mid = (left + right) // 2
      candidate = text[:mid]

      try:
        tokens = len(self.tokenizer.encode(candidate))
        if tokens <= max_tokens:
          best_end = mid
          left = mid + 1
        else:
          right = mid - 1
      except Exception:
        # Fallback if tokenizer fails
        return self._approximate_truncate(text, max_tokens, self.count_tokens(text))

    truncated = text[:best_end]
    if best_end < len(text):
      # Add ellipsis if we truncated
      # Make sure ellipsis doesn't exceed token limit
      test_with_ellipsis = truncated + "..."
      try:
        if len(self.tokenizer.encode(test_with_ellipsis)) <= max_tokens:
          truncated = test_with_ellipsis
      except Exception:
        pass

    logging.debug(
        "Precisely truncated text from %d to %d tokens",
        self.count_tokens(text),
        self.count_tokens(truncated),
    )
    return truncated

  def _approximate_truncate(self, text: str, max_tokens: int, current_tokens: int) -> str:
    """Truncate using character-based approximation.

    Args:
      text: Text to truncate
      max_tokens: Maximum tokens allowed
      current_tokens: Current token count

    Returns:
      Truncated text
    """
    # Calculate approximate character ratio
    char_per_token = len(text) / current_tokens if current_tokens > 0 else 4.0
    target_chars = int(max_tokens * char_per_token)

    if target_chars >= len(text):
      return text

    # Leave room for ellipsis
    target_chars = max(0, target_chars - 3)
    truncated = text[:target_chars] + "..."

    logging.debug(
        "Approximately truncated text from %d to %d tokens",
        current_tokens,
        self.count_tokens(truncated),
    )
    return truncated

  def get_tokenizer_info(self) -> Dict[str, Any]:
    """Get information about the current tokenizer setup.

    Returns:
      Dictionary with tokenizer information
    """
    return {
        "model_name": self.model_name,
        "token_limit": self.limit,
        "has_tokenizer": self.tokenizer is not None,
        "tokenizer_type": type(self.tokenizer).__name__ if self.tokenizer else None,
        "approximation_ratio": self._get_approximation_ratio(),
    }

  def _get_approximation_ratio(self) -> float:
    """Get the character-to-token ratio used for approximation.

    Returns:
      Characters per token ratio
    """
    model_lower = self.model_name.lower()
    if "gpt" in model_lower:
      return 3.7
    elif "claude" in model_lower:
      return 3.8
    elif "gemini" in model_lower:
      return 3.6
    elif "deepseek" in model_lower:
      return 3.7
    else:
      return 4.0

  def validate_token_count(self, text: str, expected_tokens: int, tolerance: float = 0.1) -> bool:
    """Validate token count against expected value (for testing).

    Args:
      text: Text to count tokens for
      expected_tokens: Expected token count
      tolerance: Tolerance as fraction (0.1 = 10%)

    Returns:
      True if token count is within tolerance
    """
    actual_tokens = self.count_tokens(text)
    lower_bound = expected_tokens * (1 - tolerance)
    upper_bound = expected_tokens * (1 + tolerance)
    return lower_bound <= actual_tokens <= upper_bound


class MemorySummarizer:
  """Summarizes game memory for efficient context generation."""

  def summarize_action(
      self, action: FreeCivAction, result: Dict[str, Any]
  ) -> str:
    """Summarize a single action and its result.

    Args:
      action: FreeCiv action that was taken
      result: Result of the action execution

    Returns:
      Concise summary string
    """
    # Create concise action summary
    if action.action_type == "unit_move":
      summary = f"Moved {action.source} to {action.target}"
    elif action.action_type == "unit_attack":
      summary = f"Attacked with {action.source}"
    elif action.action_type == "city_production":
      summary = (
          f"Set {action.source} to produce"
          f" {action.target.get('value', 'unknown')}"
      )
    elif action.action_type == "unit_fortify":
      summary = f"Fortified {action.source}"
    else:
      summary = f"{action.action_type} on {action.source}"

    # Add result context if available
    if result.get("success"):
      if "score_delta" in result:
        summary += f" (+{result['score_delta']} points)"
    else:
      summary += " (failed)"

    return summary

  def summarize_turn_sequence(self, turns: List[Dict[str, Any]]) -> str:
    """Summarize a sequence of turns into key events.

    Args:
      turns: List of turn records

    Returns:
      Summary of important events and patterns
    """
    if not turns:
      return ""

    summaries = []

    # Group by turn number
    turns_by_number = {}
    for turn_data in turns:
      turn_num = turn_data.get("turn", 0)
      if turn_num not in turns_by_number:
        turns_by_number[turn_num] = []
      turns_by_number[turn_num].append(turn_data)

    # Summarize each turn
    for turn_num in sorted(turns_by_number.keys()):
      turn_actions = turns_by_number[turn_num]
      actions_summary = ", ".join([
          self.summarize_action(td["action"], td["result"])
          for td in turn_actions
          if "action" in td and "result" in td
      ])

      if actions_summary:
        summaries.append(f"Turn {turn_num}: {actions_summary}")

    return "; ".join(summaries)

  def extract_important_events(self, turns: List[Dict[str, Any]]) -> List[str]:
    """Extract important events from turn history.

    Args:
      turns: List of turn records

    Returns:
      List of important event descriptions
    """
    important_events = []

    for turn_data in turns:
      result = turn_data.get("result", {})
      action = turn_data.get("action")

      # Mark as important if significant score change
      if result.get("score_delta", 0) > 20:
        event = f"Major gain: {self.summarize_action(action, result)}"
        important_events.append(event)

      # Mark combat actions as important
      if action and action.action_type in ["unit_attack", "city_capture"]:
        event = f"Combat: {self.summarize_action(action, result)}"
        important_events.append(event)

      # Mark failed actions as important for learning
      if not result.get("success", True):
        event = f"Failed: {self.summarize_action(action, result)}"
        important_events.append(event)

    return important_events


class GameMemory:
  """Game memory system for tracking action history and context.

  Manages game history with configurable size limits, provides context
  summarization for prompt generation, and includes performance optimizations
  using caching.

  Attributes:
    max_size: Maximum number of turns to keep in memory
    history: Deque containing recent action history
    cache: LRU cache for expensive operations
    summarizer: Component for generating summaries
    token_manager: Optional token manager for context limits
  """

  def __init__(
      self, max_size: int = 10, token_manager: Optional[TokenManager] = None
  ):
    """Initialize game memory system.

    Args:
      max_size: Maximum number of turns to keep in memory
      token_manager: Optional token manager for context limits
    """
    self.max_size = max_size
    self.history = deque(maxlen=max_size)
    # Use configurable cache settings
    from game_arena.harness.freeciv_proxy_client import (
        DEFAULT_SIMILARITY_CACHE_SIZE, DEFAULT_MEMORY_CACHE_TTL
    )
    self.cache = LRUCache[str, str](
        max_size=DEFAULT_SIMILARITY_CACHE_SIZE, ttl_seconds=DEFAULT_MEMORY_CACHE_TTL
    )
    self.summarizer = MemorySummarizer()
    self.token_manager = token_manager

    logging.debug("GameMemory initialized with max_size=%d", max_size)

  def record_action(
      self, action: FreeCivAction, result: Dict[str, Any]
  ) -> None:
    """Record an action and its result in memory.

    Args:
      action: FreeCiv action that was executed
      result: Dictionary containing action results and metadata
    """
    turn_record = {
        "timestamp": time.time(),
        "action": action,
        "result": result,
        "summary": self.summarizer.summarize_action(action, result),
    }

    self.history.append(turn_record)

    # Clear cache when new data is added
    self.cache.clear()

    logging.debug(
        "Recorded action: %s (memory size: %d)",
        turn_record["summary"],
        len(self.history),
    )

  def get_context(self, max_tokens: int = 2000) -> str:
    """Generate context string for prompt inclusion.

    Args:
      max_tokens: Maximum tokens for context (will be truncated if needed)

    Returns:
      Formatted context string for prompt inclusion
    """
    if not self.history:
      return ""

    # Create cache key based on history state and token limit
    history_hash = self._hash_history()
    cache_key = f"context_{history_hash}_{max_tokens}"

    # Check cache first
    cached_context = self.cache.get(cache_key)
    if cached_context is not None:
      logging.debug("Context cache hit")
      return cached_context

    # Generate context
    context = self._generate_context_string()

    # Truncate if token manager is available
    if self.token_manager:
      context = self.token_manager.truncate_to_limit(
          context, reserve=max_tokens
      )
    else:
      # Rough truncation if no token manager
      if len(context) > max_tokens * 4:  # ~4 chars per token
        context = context[: max_tokens * 4 - 3] + "..."

    # Cache the result
    self.cache.set(cache_key, context)

    logging.debug("Generated context: %d chars", len(context))
    return context

  def _generate_context_string(self) -> str:
    """Generate the full context string from memory.

    Returns:
      Formatted context string
    """
    if not self.history:
      return ""

    context_parts = []

    # Add recent action summary
    recent_turns = list(self.history)[-5:]  # Last 5 turns
    if recent_turns:
      recent_summary = self.summarizer.summarize_turn_sequence(recent_turns)
      context_parts.append(f"Recent actions: {recent_summary}")

    # Add important events
    important_events = self.summarizer.extract_important_events(
        list(self.history)
    )
    if important_events:
      events_str = "; ".join(important_events[-3:])  # Last 3 important events
      context_parts.append(f"Key events: {events_str}")

    # Add performance summary
    successful_actions = sum(
        1
        for turn in self.history
        if turn.get("result", {}).get("success", True)
    )
    success_rate = successful_actions / len(self.history) if self.history else 0
    context_parts.append(f"Success rate: {success_rate:.1%}")

    return " | ".join(context_parts)

  def _hash_history(self) -> str:
    """Create hash of current history state for caching.

    Returns:
      SHA256 hash of history state
    """
    # Create deterministic representation of history
    history_data = []
    for turn in self.history:
      # Only include essential data for hashing
      turn_data = {
          "action_type": (
              turn["action"].action_type if turn.get("action") else None
          ),
          "actor_id": turn["action"].actor_id if turn.get("action") else None,
          "result_success": turn.get("result", {}).get("success"),
          "timestamp": turn.get("timestamp"),
      }
      history_data.append(turn_data)

    history_str = json.dumps(history_data, sort_keys=True)
    return hashlib.sha256(history_str.encode()).hexdigest()[:16]

  def get_cache_statistics(self) -> Dict[str, Any]:
    """Get cache performance statistics.

    Returns:
      Dictionary with cache metrics
    """
    return self.cache.statistics

  def clear_history(self) -> None:
    """Clear all history and cache."""
    self.history.clear()
    self.cache.clear()
    logging.debug("Memory cleared")

  def get_recent_actions(self, count: int = 5) -> List[Dict[str, Any]]:
    """Get recent actions from memory.

    Args:
      count: Number of recent actions to retrieve

    Returns:
      List of recent action records
    """
    recent = list(self.history)[-count:] if self.history else []
    return recent

  def get_action_patterns(self) -> Dict[str, int]:
    """Analyze action patterns in memory.

    Returns:
      Dictionary mapping action types to frequency counts
    """
    patterns = {}
    for turn in self.history:
      action = turn.get("action")
      if action:
        action_type = action.action_type
        patterns[action_type] = patterns.get(action_type, 0) + 1

    return patterns

  def get_performance_summary(self) -> Dict[str, Any]:
    """Get performance summary from memory.

    Returns:
      Dictionary with performance metrics
    """
    if not self.history:
      return {"total_actions": 0, "success_rate": 0.0}

    total_actions = len(self.history)
    successful = sum(
        1
        for turn in self.history
        if turn.get("result", {}).get("success", True)
    )

    total_score_delta = sum(
        turn.get("result", {}).get("score_delta", 0) for turn in self.history
    )

    return {
        "total_actions": total_actions,
        "success_rate": successful / total_actions,
        "total_score_delta": total_score_delta,
        "avg_score_per_action": total_score_delta / total_actions,
        "action_patterns": self.get_action_patterns(),
    }
