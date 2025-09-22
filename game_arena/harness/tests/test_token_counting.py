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

"""Tests for improved token counting accuracy."""

import unittest
from unittest.mock import MagicMock, patch

from game_arena.harness.freeciv_memory import TokenManager


class TestTokenCounting(unittest.TestCase):
  """Test cases for accurate token counting."""

  def setUp(self):
    """Set up test fixtures."""
    self.sample_texts = [
      "Hello world!",  # Simple text
      "FreeCiv is a strategy game where players build civilizations.",  # Game text
      "Move warrior from (10,14) to (11,14). Attack enemy unit at position (12,15).",  # Action text
      "This is a longer text that contains multiple sentences. It includes various punctuation marks, numbers like 123, and technical terms. The goal is to test how different tokenizers handle diverse content types.",  # Complex text
    ]

  def test_gpt_tokenizer_loading(self):
    """Test loading GPT tokenizer when available."""
    try:
      import tiktoken
      # Test with real tokenizer if available
      token_manager = TokenManager("gpt-4")
      self.assertIsNotNone(token_manager.tokenizer)

      # Test token counting
      text = "Hello, world! This is a test."
      tokens = token_manager.count_tokens(text)
      self.assertIsInstance(tokens, int)
      self.assertGreater(tokens, 0)

      # Should be more accurate than simple division
      approx_tokens = len(text) // 4
      self.assertTrue(abs(tokens - approx_tokens) <= approx_tokens * 0.5)  # Within 50%

    except ImportError:
      # Skip if tiktoken not available
      self.skipTest("tiktoken not available")

  def test_approximation_accuracy(self):
    """Test that approximations are more accurate than basic division."""
    models_to_test = ["gpt-4", "claude-opus-4", "gemini-2.5", "deepseek"]

    for model_name in models_to_test:
      with self.subTest(model=model_name):
        token_manager = TokenManager(model_name)

        for text in self.sample_texts:
          tokens = token_manager.count_tokens(text)
          basic_estimate = len(text) // 4

          # Should be reasonable (not zero, not excessive)
          self.assertGreater(tokens, 0)
          self.assertLess(tokens, len(text))  # Should be less than character count

          # For longer texts, our approximation should be different from basic division
          if len(text) > 50:  # Only test difference for longer texts
            # Our improved approximation should be different from naive approach
            ratio_diff = abs(tokens / len(text) - 0.25)  # 0.25 = 1/4
            self.assertGreater(ratio_diff, 0.005)  # Should have some difference

  def test_model_specific_ratios(self):
    """Test that different models use different approximation ratios."""
    gpt_manager = TokenManager("gpt-4")
    claude_manager = TokenManager("claude-opus-4")
    gemini_manager = TokenManager("gemini-2.5")

    test_text = "This is a test text for comparing tokenization ratios."

    gpt_tokens = gpt_manager.count_tokens(test_text)
    claude_tokens = claude_manager.count_tokens(test_text)
    gemini_tokens = gemini_manager.count_tokens(test_text)

    # Should be slightly different due to different approximation ratios
    # Not necessarily different (text might tokenize similarly), but ratios should differ
    self.assertNotEqual(gpt_manager._get_approximation_ratio(),
                       claude_manager._get_approximation_ratio())
    self.assertNotEqual(gpt_manager._get_approximation_ratio(),
                       gemini_manager._get_approximation_ratio())

  def test_truncation_accuracy(self):
    """Test that truncation respects token limits."""
    token_manager = TokenManager("gpt-4")

    # Create text that will definitely need truncation
    long_text = " ".join(["This is a test sentence with multiple words."] * 1000)
    original_tokens = token_manager.count_tokens(long_text)

    # Force a low limit that requires truncation
    original_limit = token_manager.limit
    token_manager.limit = 200  # Force a low limit

    try:
      truncated = token_manager.truncate_to_limit(long_text, reserve=10)
      truncated_tokens = token_manager.count_tokens(truncated)

      # Should be within limit (accounting for reserve)
      max_allowed = token_manager.limit - 10
      self.assertLessEqual(truncated_tokens, max_allowed)

      # Should be shorter than original
      self.assertLess(len(truncated), len(long_text))
      self.assertLess(truncated_tokens, original_tokens)

    finally:
      # Restore original limit
      token_manager.limit = original_limit

  def test_binary_search_truncation(self):
    """Test binary search truncation with mock tokenizer."""
    # Create mock tokenizer that returns predictable token counts
    mock_tokenizer = MagicMock()

    # Mock encode method to return length based on text length
    def mock_encode(text):
      # Simple mock: 1 token per 4 characters
      return ["token"] * (len(text) // 4)

    mock_tokenizer.encode = mock_encode

    token_manager = TokenManager("gpt-4")
    token_manager.tokenizer = mock_tokenizer

    test_text = "x" * 400  # 400 characters = 100 tokens with our mock
    truncated = token_manager._binary_search_truncate(test_text, 50)  # Limit to 50 tokens

    # Should be approximately 200 characters (50 tokens * 4 chars/token)
    self.assertLessEqual(len(truncated), 204)  # Allow for ellipsis
    self.assertGreaterEqual(len(truncated), 190)  # Should be close to target

  def test_tokenizer_info(self):
    """Test tokenizer information retrieval."""
    token_manager = TokenManager("gpt-4")
    info = token_manager.get_tokenizer_info()

    self.assertIn("model_name", info)
    self.assertIn("token_limit", info)
    self.assertIn("has_tokenizer", info)
    self.assertIn("approximation_ratio", info)

    self.assertEqual(info["model_name"], "gpt-4")
    self.assertIsInstance(info["token_limit"], int)
    self.assertIsInstance(info["has_tokenizer"], bool)
    self.assertIsInstance(info["approximation_ratio"], float)

  def test_token_validation(self):
    """Test token count validation function."""
    token_manager = TokenManager("gpt-4")

    # Test with exact match
    self.assertTrue(token_manager.validate_token_count("test", 1, tolerance=0.5))

    # Test with tolerance
    text = "This is a longer test text"
    actual_tokens = token_manager.count_tokens(text)

    # Should validate against itself
    self.assertTrue(token_manager.validate_token_count(text, actual_tokens, tolerance=0.01))

    # Should fail with very different expected count
    self.assertFalse(token_manager.validate_token_count(text, actual_tokens * 10, tolerance=0.1))

  def test_different_text_types(self):
    """Test token counting on different types of text content."""
    token_manager = TokenManager("gpt-4")

    test_cases = [
      ("code", "def hello_world():\n    print('Hello, world!')\n    return True"),
      ("json", '{"name": "FreeCiv", "type": "game", "players": [1, 2, 3, 4]}'),
      ("action", "unit_move_warrior(101)_to(11,14)"),
      ("prose", "The quick brown fox jumps over the lazy dog."),
      ("numbers", "1234567890 98765432.10 -456.789"),
      ("punctuation", "Hello! How are you? I'm fine... What about you?"),
    ]

    for text_type, text in test_cases:
      with self.subTest(text_type=text_type):
        tokens = token_manager.count_tokens(text)
        self.assertGreater(tokens, 0, f"Failed for {text_type}")

        # Reasonable bounds check
        min_expected = len(text) // 10  # Very conservative lower bound
        max_expected = len(text)  # Upper bound is character count
        self.assertGreaterEqual(tokens, min_expected)
        self.assertLessEqual(tokens, max_expected)

  def test_memory_integration(self):
    """Test token manager integration with GameMemory."""
    from game_arena.harness.freeciv_memory import GameMemory

    # Create memory with token manager
    token_manager = TokenManager("gpt-4")
    memory = GameMemory(max_size=5, token_manager=token_manager)

    # Memory should use the token manager
    self.assertEqual(memory.token_manager, token_manager)

    # Test context generation with token limits
    context = memory.get_context(max_tokens=100)

    # Should return string
    self.assertIsInstance(context, str)

    # Should respect token limits (roughly)
    if context:  # If not empty
      tokens = token_manager.count_tokens(context)
      # Should be reasonable (not exactly 100, but in the ballpark)
      self.assertLessEqual(tokens, 150)  # Allow some overhead


if __name__ == '__main__':
  unittest.main()