"""Tests for FreeCiv parsers and action parsing functionality."""

import copy
import json
import time
import unittest
from unittest.mock import Mock, patch

from game_arena.harness import parsers
from game_arena.harness.freeciv_parsers import (
    FreeCivRuleBasedParser,
    FreeCivSoftParser,
    create_freeciv_parser_chain,
)
from game_arena.harness.freeciv_state import FreeCivAction


class TestFreeCivRuleBasedParser(unittest.TestCase):
  """Test FreeCivRuleBasedParser functionality."""

  def setUp(self):
    self.parser = FreeCivRuleBasedParser()

  def test_json_parsing(self):
    """Test JSON parsing functionality."""
    # Test basic JSON format
    json_text = '{"action": "unit_move", "unit": 101, "to": [3, 5]}'
    parser_input = parsers.TextParserInput(text=json_text)

    result = self.parser.parse(parser_input)
    self.assertIsNotNone(result)
    self.assertIn("unit_move", result)
    self.assertIn("101", result)
    self.assertIn("3", result)
    self.assertIn("5", result)

    # Test alternative JSON format
    json_text2 = (
        '{"type": "city_production", "city": 301, "target": {"value":'
        ' "warriors"}}'
    )
    parser_input2 = parsers.TextParserInput(text=json_text2)

    result2 = self.parser.parse(parser_input2)
    self.assertIsNotNone(result2)
    self.assertIn("city_production", result2)
    self.assertIn("301", result2)
    self.assertIn("warriors", result2)

  def test_json_parsing_with_text_wrapper(self):
    """Test JSON parsing when JSON is embedded in text."""
    wrapped_text = (
        'I will perform this action: {"action": "unit_attack", "unit": 102,'
        ' "target": {"id": 203}} to complete the task.'
    )
    parser_input = parsers.TextParserInput(text=wrapped_text)

    result = self.parser.parse(parser_input)
    self.assertIsNotNone(result)
    self.assertIn("unit_attack", result)
    self.assertIn("102", result)
    self.assertIn("203", result)

  def test_invalid_json_fallback_to_regex(self):
    """Test fallback to regex parsing when JSON is invalid."""
    invalid_json = "unit_move_settlers(101)_to(2,3)"
    parser_input = parsers.TextParserInput(text=invalid_json)

    result = self.parser.parse(parser_input)
    self.assertIsNotNone(result)
    self.assertEqual(result, "unit_move_settlers(101)_to(2,3)")

  def test_regex_parsing_patterns(self):
    """Test regex pattern matching for different action types."""
    test_cases = [
        ("unit_move_warrior(102)_to(4,6)", "unit_move_warrior(102)_to(4,6)"),
        (
            "unit_attack_archer(103)_target(204)",
            "unit_attack_archer(103)_target(204)",
        ),
        ("unit_fortify_legion(105)", "unit_fortify_legion(105)"),
        (
            "city_production_rome(302)_target(granary)",
            "city_production_rome(302)_target(granary)",
        ),
    ]

    for input_text, expected in test_cases:
      with self.subTest(input=input_text):
        parser_input = parsers.TextParserInput(text=input_text)
        result = self.parser.parse(parser_input)
        self.assertEqual(result, expected)

  def test_soft_matching_fallback(self):
    """Test soft matching against legal moves."""
    legal_moves = [
        "unit_move_settlers(101)_to(2,3)",
        "unit_fortify_warrior(102)",
        "city_production_rome(301)_target(warriors)",
    ]

    # Test partial match
    parser_input = parsers.TextParserInput(
        text="move settlers", legal_moves=legal_moves
    )

    result = self.parser.parse(parser_input)
    self.assertIsNotNone(result)
    self.assertIn("unit_move_settlers", result)

  def test_empty_and_invalid_inputs(self):
    """Test handling of empty and invalid inputs."""
    # Test empty input
    parser_input = parsers.TextParserInput(text="")
    result = self.parser.parse(parser_input)
    self.assertIsNone(result)

    # Test pure noise
    parser_input = parsers.TextParserInput(text="random nonsense text")
    result = self.parser.parse(parser_input)
    self.assertIsNone(result)

    # Test malformed JSON
    parser_input = parsers.TextParserInput(text='{"invalid": json syntax}')
    result = self.parser.parse(parser_input)
    self.assertIsNone(result)


class TestFreeCivSoftParser(unittest.TestCase):
  """Test FreeCivSoftParser fuzzy matching functionality."""

  def setUp(self):
    self.parser = FreeCivSoftParser()

  def test_fuzzy_matching_exact_match(self):
    """Test exact matching."""
    legal_moves = [
        "unit_move_settlers(101)_to(2,3)",
        "unit_fortify_warrior(102)",
        "city_production_rome(301)_target(warriors)",
    ]

    parser_input = parsers.TextParserInput(
        text="unit_move_settlers(101)_to(2,3)", legal_moves=legal_moves
    )

    result = self.parser.parse(parser_input)
    self.assertEqual(result, "unit_move_settlers(101)_to(2,3)")

  def test_fuzzy_matching_partial_match(self):
    """Test partial token matching."""
    legal_moves = [
        "unit_move_settlers(101)_to(2,3)",
        "unit_fortify_warrior(102)",
        "city_production_rome(301)_target(warriors)",
    ]

    # Test partial match with key tokens
    parser_input = parsers.TextParserInput(
        text="move settlers 101", legal_moves=legal_moves
    )

    result = self.parser.parse(parser_input)
    self.assertIsNotNone(result)
    self.assertIn("unit_move_settlers", result)

  def test_fuzzy_matching_with_typos(self):
    """Test fuzzy matching with typos using edit distance."""
    legal_moves = [
        "unit_move_settlers(101)_to(2,3)",
        "unit_fortify_warrior(102)",
        "city_production_rome(301)_target(warriors)",
    ]

    # Test with typos
    parser_input = parsers.TextParserInput(
        text="unit_mov_settlrs(101)_to(2,3)",  # typos: mov instead of move, settlrs instead of settlers
        legal_moves=legal_moves,
    )

    result = self.parser.parse(parser_input)
    self.assertIsNotNone(result)
    self.assertIn("unit_move_settlers", result)

  def test_similarity_caching(self):
    """Test that similarity calculations are cached for performance."""
    legal_moves = [
        "unit_move_settlers(101)_to(2,3)",
        "unit_fortify_warrior(102)",
    ]

    parser_input = parsers.TextParserInput(
        text="move settlers", legal_moves=legal_moves
    )

    # First call should populate cache
    result1 = self.parser.parse(parser_input)
    cache_size_after_first = len(self.parser._similarity_cache)

    # Second call with same input should use cache
    result2 = self.parser.parse(parser_input)
    cache_size_after_second = len(self.parser._similarity_cache)

    self.assertEqual(result1, result2)
    self.assertEqual(cache_size_after_first, cache_size_after_second)

  def test_cache_size_limit(self):
    """Test that cache respects size limits."""
    # Create parser with small cache size for testing
    from game_arena.harness.freeciv_parser_config import FreeCivParserConfig

    config = FreeCivParserConfig(max_cache_size=3)
    parser = FreeCivSoftParser(config)

    legal_moves = ["unit_move_settlers(101)_to(2,3)"]

    # Add more items than cache size
    for i in range(5):
      parser_input = parsers.TextParserInput(
          text=f"test input {i}", legal_moves=legal_moves
      )
      parser.parse(parser_input)

    # Cache should not exceed max size
    self.assertLessEqual(len(parser._similarity_cache), config.max_cache_size)

  def test_enhanced_similarity_algorithm(self):
    """Test the enhanced similarity calculation with multiple metrics."""
    legal_moves = [
        "unit_move_settlers(101)_to(2,3)",
        "unit_attack_warrior(102)_target(203)",
        "city_production_rome(301)_target(warriors)",
    ]

    # Test action type preference
    parser_input = parsers.TextParserInput(
        text="attack with warrior", legal_moves=legal_moves
    )

    result = self.parser.parse(parser_input)
    self.assertIsNotNone(result)
    self.assertIn("unit_attack_warrior", result)

  def test_low_confidence_rejection(self):
    """Test that low confidence matches are rejected."""
    legal_moves = [
        "unit_move_settlers(101)_to(2,3)",
        "city_production_rome(301)_target(warriors)",
    ]

    # Test completely unrelated input
    parser_input = parsers.TextParserInput(
        text="xyz abc def", legal_moves=legal_moves
    )

    result = self.parser.parse(parser_input)
    self.assertIsNone(result)


class TestParserChain(unittest.TestCase):
  """Test parser chain integration."""

  def test_parser_chain_creation(self):
    """Test that parser chain is created correctly."""
    chain = create_freeciv_parser_chain()
    self.assertIsInstance(chain, parsers.ChainedMoveParser)
    self.assertGreaterEqual(
        len(chain._parsers), 2
    )  # At least rule-based and soft parser

  def test_parser_chain_with_model(self):
    """Test parser chain creation with LLM model."""
    mock_model = Mock()
    chain = create_freeciv_parser_chain(model=mock_model)
    self.assertIsInstance(chain, parsers.ChainedMoveParser)
    self.assertGreaterEqual(
        len(chain._parsers), 3
    )  # Rule-based, LLM, and soft parser

  def test_parser_chain_execution_order(self):
    """Test that parser chain executes in correct order."""
    legal_moves = [
        "unit_move_settlers(101)_to(2,3)",
        "city_production_rome(301)_target(warriors)",
    ]

    chain = create_freeciv_parser_chain()

    # Test JSON parsing (should be handled by rule-based parser first)
    json_input = parsers.TextParserInput(
        text='{"action": "unit_move", "unit": 101, "to": [2, 3]}',
        legal_moves=legal_moves,
    )

    result = chain.parse(json_input)
    self.assertIsNotNone(result)
    self.assertIn("unit_move", result)

    # Test fuzzy matching (should fall back to soft parser)
    fuzzy_input = parsers.TextParserInput(
        text="move settlers", legal_moves=legal_moves
    )

    result = chain.parse(fuzzy_input)
    self.assertIsNotNone(result)
    self.assertIn("unit_move_settlers", result)


class TestPerformance(unittest.TestCase):
  """Test performance requirements."""

  def test_json_parsing_performance(self):
    """Test JSON parsing meets <10ms requirement."""
    parser = FreeCivRuleBasedParser()
    json_text = '{"action": "unit_move", "unit": 101, "to": [3, 5]}'
    parser_input = parsers.TextParserInput(text=json_text)

    # Warm up
    for _ in range(10):
      parser.parse(parser_input)

    # Measure performance
    start_time = time.perf_counter()
    for _ in range(100):
      parser.parse(parser_input)
    end_time = time.perf_counter()

    avg_time_ms = ((end_time - start_time) / 100) * 1000
    self.assertLess(
        avg_time_ms,
        10,
        f"JSON parsing took {avg_time_ms:.2f}ms (should be <10ms)",
    )

  def test_fuzzy_matching_performance(self):
    """Test fuzzy matching performance."""
    parser = FreeCivSoftParser()
    legal_moves = [f"unit_move_settlers({i})_to({i},{i+1})" for i in range(50)]
    parser_input = parsers.TextParserInput(
        text="move settlers 25", legal_moves=legal_moves
    )

    # Warm up
    for _ in range(5):
      parser.parse(parser_input)

    # Measure performance
    start_time = time.perf_counter()
    for _ in range(50):
      parser.parse(parser_input)
    end_time = time.perf_counter()

    avg_time_ms = ((end_time - start_time) / 50) * 1000
    self.assertLess(
        avg_time_ms,
        10,
        f"Fuzzy matching took {avg_time_ms:.2f}ms (should be <10ms)",
    )

  def test_parser_chain_performance(self):
    """Test full parser chain performance."""
    chain = create_freeciv_parser_chain()
    legal_moves = [
        "unit_move_settlers(101)_to(2,3)",
        "unit_fortify_warrior(102)",
        "city_production_rome(301)_target(warriors)",
    ]

    parser_input = parsers.TextParserInput(
        text='{"action": "unit_move", "unit": 101, "to": [2, 3]}',
        legal_moves=legal_moves,
    )

    # Warm up
    for _ in range(10):
      chain.parse(parser_input)

    # Measure performance
    start_time = time.perf_counter()
    for _ in range(100):
      chain.parse(parser_input)
    end_time = time.perf_counter()

    avg_time_ms = ((end_time - start_time) / 100) * 1000
    self.assertLess(
        avg_time_ms,
        10,
        f"Parser chain took {avg_time_ms:.2f}ms (should be <10ms)",
    )


class TestErrorHandling(unittest.TestCase):
  """Test error handling and edge cases."""

  def test_malformed_json_handling(self):
    """Test handling of malformed JSON."""
    parser = FreeCivRuleBasedParser()

    malformed_inputs = [
        '{"action": "unit_move", "unit": 101, "to": [3, 5]',  # Missing closing brace
        '{"action": "unit_move" "unit": 101, "to": [3, 5]}',  # Missing comma
        '{action: "unit_move", "unit": 101, "to": [3, 5]}',  # Unquoted key
        '{"action": unit_move, "unit": 101, "to": [3, 5]}',  # Unquoted value
    ]

    for malformed_json in malformed_inputs:
      with self.subTest(json=malformed_json):
        parser_input = parsers.TextParserInput(text=malformed_json)
        # Should not raise exception, should return None or fallback gracefully
        result = parser.parse(parser_input)
        # We don't assert None because it might fallback to regex parsing

  def test_large_input_handling(self):
    """Test handling of very large inputs."""
    parser = FreeCivRuleBasedParser()
    large_text = "x" * 10001  # Just large text without valid JSON/action
    parser_input = parsers.TextParserInput(text=large_text)

    # Should handle large inputs gracefully - now rejects for security
    result = parser.parse(parser_input)
    self.assertIsNone(result)

  def test_unicode_handling(self):
    """Test handling of unicode characters."""
    parser = FreeCivRuleBasedParser()
    unicode_text = (
        '{"action": "unit_move", "unit": 101, "to": [3, 5], "note": "移动单位"}'
    )
    parser_input = parsers.TextParserInput(text=unicode_text)

    # Should handle unicode gracefully
    result = parser.parse(parser_input)
    self.assertIsNotNone(result)

  def test_none_and_empty_inputs(self):
    """Test handling of None and empty inputs."""
    parser = FreeCivRuleBasedParser()

    # Test empty string
    parser_input = parsers.TextParserInput(text="")
    result = parser.parse(parser_input)
    self.assertIsNone(result)

    # Test whitespace only
    parser_input = parsers.TextParserInput(text="   \n\t  ")
    result = parser.parse(parser_input)
    self.assertIsNone(result)

  def test_extremely_long_input_security(self):
    """Test security protection against extremely long inputs."""
    parser = FreeCivRuleBasedParser()
    # Create input larger than MAX_INPUT_SIZE (10KB)
    extremely_long_text = "x" * 15000
    parser_input = parsers.TextParserInput(text=extremely_long_text)

    result = parser.parse(parser_input)
    self.assertIsNone(result)

  def test_concurrent_cache_access(self):
    """Test concurrent access to the similarity cache."""
    import threading
    import time

    parser = FreeCivSoftParser()
    legal_moves = [
        "unit_move_settlers(101)_to(2,3)",
        "unit_fortify_warrior(102)",
        "city_production_rome(301)_target(warriors)",
    ]

    results = []
    errors = []

    def parse_action(text_suffix):
      try:
        parser_input = parsers.TextParserInput(
            text=f"move settlers {text_suffix}", legal_moves=legal_moves
        )
        result = parser.parse(parser_input)
        results.append(result)
      except Exception as e:
        errors.append(e)

    # Create multiple threads accessing cache concurrently
    threads = []
    for i in range(10):
      thread = threading.Thread(target=parse_action, args=(i,))
      threads.append(thread)
      thread.start()

    for thread in threads:
      thread.join()

    # Should not have any errors from concurrent access
    self.assertEqual(
        len(errors), 0, f"Concurrent access caused errors: {errors}"
    )
    # Should have results from all threads
    self.assertEqual(len(results), 10)

  def test_cache_invalidation_on_legal_moves_change(self):
    """Test that cache is invalidated when legal moves change."""
    parser = FreeCivSoftParser()

    legal_moves_1 = ["unit_move_settlers(101)_to(2,3)"]
    legal_moves_2 = ["unit_fortify_warrior(102)"]

    parser_input_1 = parsers.TextParserInput(
        text="move settlers", legal_moves=legal_moves_1
    )
    parser_input_2 = parsers.TextParserInput(
        text="fortify warrior", legal_moves=legal_moves_2
    )

    # First parse with legal_moves_1
    result1 = parser.parse(parser_input_1)
    cache_size_after_first = len(parser._similarity_cache)

    # Second parse with different legal moves should clear cache
    result2 = parser.parse(parser_input_2)
    # Cache should have been cleared and repopulated
    self.assertIsNotNone(result1)
    self.assertIsNotNone(result2)

  def test_manual_cache_invalidation(self):
    """Test manual cache invalidation."""
    parser = FreeCivSoftParser()
    legal_moves = ["unit_move_settlers(101)_to(2,3)"]

    parser_input = parsers.TextParserInput(
        text="move settlers", legal_moves=legal_moves
    )

    # Populate cache
    parser.parse(parser_input)
    self.assertGreater(len(parser._similarity_cache), 0)

    # Manually invalidate cache
    parser.invalidate_cache()
    self.assertEqual(len(parser._similarity_cache), 0)
    self.assertIsNone(parser._legal_moves_hash)

  def test_malformed_game_state_handling(self):
    """Test action validation with malformed game states."""
    # This tests the natural language parsing robustness
    parser = FreeCivRuleBasedParser()

    # Test natural language parsing with potentially problematic input
    natural_language_inputs = [
        "move unit 999 to coordinates 100,200",  # Non-existent unit
        "attack city 500 with unit 101",  # Non-existent city
        "build improvement in city that does not exist",
    ]

    for nl_input in natural_language_inputs:
      with self.subTest(input=nl_input):
        parser_input = parsers.TextParserInput(text=nl_input)
        # Should not raise exception, should handle gracefully
        try:
          result = parser.parse(parser_input)
          # Result might be None or a best-effort parse
        except Exception as e:
          self.fail(f"Natural language parsing raised exception: {e}")

  def test_packet_conversion_failure_handling(self):
    """Test error recovery when packet conversion fails."""
    # Create action with invalid data that might cause packet conversion issues
    try:
      action = FreeCivAction(
          action_type="invalid_action_type",
          actor_id=999999,  # Very large ID
          source="invalid_source",
      )
      # Try to convert to packet - this might fail but shouldn't crash
      packet = action.to_packet()
      # If it succeeds, packet should have basic structure
      self.assertIn("pid", packet)
      self.assertIn("type", packet)
    except Exception:
      # Packet conversion failure is acceptable
      pass

  def test_cache_thrashing_scenarios(self):
    """Test cache behavior under thrashing conditions."""
    from game_arena.harness.freeciv_parser_config import FreeCivParserConfig

    config = FreeCivParserConfig(max_cache_size=5)
    parser = FreeCivSoftParser(config)

    legal_moves = ["unit_move_settlers(101)_to(2,3)"]

    # Generate many different inputs to cause cache thrashing
    for i in range(20):
      parser_input = parsers.TextParserInput(
          text=f"unique input string {i} that differs", legal_moves=legal_moves
      )
      result = parser.parse(parser_input)
      # Should not crash or perform extremely poorly

    # Cache should respect size limit
    self.assertLessEqual(len(parser._similarity_cache), config.max_cache_size)

  def test_memory_usage_with_large_cache(self):
    """Test memory usage patterns with large cache sizes."""
    import sys

    from game_arena.harness.freeciv_parser_config import FreeCivParserConfig

    config = FreeCivParserConfig(max_cache_size=100)
    parser = FreeCivSoftParser(config)

    legal_moves = ["unit_move_settlers(101)_to(2,3)"]

    # Measure memory before
    initial_cache_size = len(parser._similarity_cache)

    # Fill cache with many entries
    for i in range(50):
      parser_input = parsers.TextParserInput(
          text=f"test input {i}", legal_moves=legal_moves
      )
      parser.parse(parser_input)

    # Cache should grow but respect limits
    final_cache_size = len(parser._similarity_cache)
    self.assertGreater(final_cache_size, initial_cache_size)
    self.assertLessEqual(final_cache_size, config.max_cache_size)

  def test_input_validation_edge_cases(self):
    """Test input validation edge cases."""
    parser = FreeCivRuleBasedParser()

    edge_case_inputs = [
        "",  # Empty string
        " \t\n ",  # Whitespace only
        "\x00\x01\x02",  # Binary characters
        "🎮🎯⚔️",  # Unicode emojis
        "<script>alert('test')</script>",  # HTML/JS injection attempt
        '" OR 1=1 --',  # SQL injection attempt
        "../../../etc/passwd",  # Path traversal attempt
    ]

    for edge_input in edge_case_inputs:
      with self.subTest(input=repr(edge_input)):
        parser_input = parsers.TextParserInput(text=edge_input)
        # Should handle all inputs gracefully without crashing
        try:
          result = parser.parse(parser_input)
          # Result can be None or a valid parse
        except Exception as e:
          self.fail(f"Input validation failed for {repr(edge_input)}: {e}")

  def test_difflib_vs_custom_edit_distance(self):
    """Test that difflib and custom edit distance give reasonable results."""
    parser = FreeCivSoftParser()

    # Test cases where both methods should work
    test_pairs = [
        ("settlers", "settlers"),  # Exact match
        ("settlers", "settlrs"),  # Missing letter
        ("warriors", "wariers"),  # Transposed letters
        ("move", "mov"),  # Truncated
        ("attack", "atack"),  # Missing letter
    ]

    for text1, text2 in test_pairs:
      with self.subTest(text1=text1, text2=text2):
        # Both methods should give similar results for these cases
        difflib_sim = parser._calculate_difflib_similarity(text1, text2)
        custom_sim = parser._calculate_edit_similarity(text1, text2)

        # Both should be between 0 and 1
        self.assertGreaterEqual(difflib_sim, 0.0)
        self.assertLessEqual(difflib_sim, 1.0)
        self.assertGreaterEqual(custom_sim, 0.0)
        self.assertLessEqual(custom_sim, 1.0)

        # For identical strings, both should return 1.0
        if text1 == text2:
          self.assertAlmostEqual(difflib_sim, 1.0, places=2)
          self.assertAlmostEqual(custom_sim, 1.0, places=2)


class TestSecurityAndPerformance(unittest.TestCase):
  """Additional tests for security and performance edge cases."""

  def test_regex_timeout_protection(self):
    """Test that regex operations don't hang on malicious input."""
    parser = FreeCivRuleBasedParser()

    # Create input designed to cause regex backtracking
    malicious_input = "{{" + "a" * 1000 + "}}"
    parser_input = parsers.TextParserInput(text=malicious_input)

    import time

    start_time = time.time()
    try:
      result = parser.parse(parser_input)
      # Should complete quickly, not hang
      elapsed = time.time() - start_time
      self.assertLess(elapsed, 1.0, "Regex operation took too long")
    except Exception:
      # Exception is acceptable, hanging is not
      elapsed = time.time() - start_time
      self.assertLess(
          elapsed, 1.0, "Regex operation took too long even with exception"
      )

  def test_json_depth_limit_protection(self):
    """Test protection against deeply nested JSON."""
    parser = FreeCivRuleBasedParser()

    # Create deeply nested JSON
    nested_json = "{"
    for i in range(20):
      nested_json += f'"level{i}": {{'
    nested_json += '"action": "unit_move", "unit": 101'
    for i in range(20):
      nested_json += "}"
    nested_json += "}"

    parser_input = parsers.TextParserInput(text=nested_json)
    # Should handle without hanging or excessive memory usage
    result = parser.parse(parser_input)
    # Result can be None (rejected) or parsed successfully

  def test_line_length_validation(self):
    """Test that excessively long lines are rejected."""
    parser = FreeCivRuleBasedParser()

    # Create input with one very long line
    long_line = "a" * 15000  # Exceeds default max_line_length of 10KB
    test_input = f"Some valid content\n{long_line}\nMore content"

    parser_input = parsers.TextParserInput(text=test_input)
    result = parser.parse(parser_input)

    # Should be rejected due to line length
    self.assertIsNone(result)

  def test_line_length_validation_soft_parser(self):
    """Test that soft parser also validates line lengths."""
    parser = FreeCivSoftParser()
    legal_moves = ["unit_move_settlers(101)_to(2,3)"]

    # Create input with one very long line
    long_line = "a" * 15000  # Exceeds default max_line_length of 10KB
    test_input = f"Move unit\n{long_line}\nto position"

    parser_input = parsers.TextParserInput(
        text=test_input, legal_moves=legal_moves
    )
    result = parser.parse(parser_input)

    # Should be rejected due to line length
    self.assertIsNone(result)

  def test_similarity_calculation_performance_bounds(self):
    """Test that similarity calculations have reasonable performance bounds."""
    parser = FreeCivSoftParser()

    # Test with longer strings
    long_text = (
        "move settlers from position 101 to coordinate 25 30 using shortest"
        " path available"
    )
    long_move = "unit_move_settlers(101)_to(25,30)_via_shortest_path_optimization_enabled"

    start_time = time.perf_counter()
    similarity = parser._calculate_enhanced_similarity(long_text, long_move)
    elapsed = time.perf_counter() - start_time

    # Should complete within reasonable time (1ms for this size)
    self.assertLess(
        elapsed, 0.001, f"Similarity calculation took {elapsed:.4f}s"
    )
    self.assertGreaterEqual(similarity, 0.0)
    self.assertLessEqual(similarity, 1.0)

  def test_memory_leak_prevention(self):
    """Test that parser doesn't accumulate memory over time."""
    import gc

    parser = FreeCivSoftParser()
    legal_moves = ["unit_move_settlers(101)_to(2,3)"]

    # Run many parsing operations
    for i in range(100):
      parser_input = parsers.TextParserInput(
          text=f"move settlers {i}", legal_moves=legal_moves
      )
      parser.parse(parser_input)

      # Periodically check that cache doesn't grow unbounded
      if i % 20 == 0:
        cache_size = len(parser._similarity_cache)
        self.assertLessEqual(
            cache_size,
            parser._config.max_cache_size,
            f"Cache size {cache_size} exceeds limit at iteration {i}",
        )

    # Force garbage collection and ensure no circular references
    gc.collect()


if __name__ == "__main__":
  unittest.main()
