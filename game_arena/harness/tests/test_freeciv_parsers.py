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
        json_text2 = '{"type": "city_production", "city": 301, "target": {"value": "warriors"}}'
        parser_input2 = parsers.TextParserInput(text=json_text2)

        result2 = self.parser.parse(parser_input2)
        self.assertIsNotNone(result2)
        self.assertIn("city_production", result2)
        self.assertIn("301", result2)
        self.assertIn("warriors", result2)

    def test_json_parsing_with_text_wrapper(self):
        """Test JSON parsing when JSON is embedded in text."""
        wrapped_text = 'I will perform this action: {"action": "unit_attack", "unit": 102, "target": {"id": 203}} to complete the task.'
        parser_input = parsers.TextParserInput(text=wrapped_text)

        result = self.parser.parse(parser_input)
        self.assertIsNotNone(result)
        self.assertIn("unit_attack", result)
        self.assertIn("102", result)
        self.assertIn("203", result)

    def test_invalid_json_fallback_to_regex(self):
        """Test fallback to regex parsing when JSON is invalid."""
        invalid_json = 'unit_move_settlers(101)_to(2,3)'
        parser_input = parsers.TextParserInput(text=invalid_json)

        result = self.parser.parse(parser_input)
        self.assertIsNotNone(result)
        self.assertEqual(result, "unit_move_settlers(101)_to(2,3)")

    def test_regex_parsing_patterns(self):
        """Test regex pattern matching for different action types."""
        test_cases = [
            ("unit_move_warrior(102)_to(4,6)", "unit_move_warrior(102)_to(4,6)"),
            ("unit_attack_archer(103)_target(204)", "unit_attack_archer(103)_target(204)"),
            ("unit_fortify_legion(105)", "unit_fortify_legion(105)"),
            ("city_production_rome(302)_target(granary)", "city_production_rome(302)_target(granary)"),
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
            "city_production_rome(301)_target(warriors)"
        ]

        # Test partial match
        parser_input = parsers.TextParserInput(
            text="move settlers",
            legal_moves=legal_moves
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
            "city_production_rome(301)_target(warriors)"
        ]

        parser_input = parsers.TextParserInput(
            text="unit_move_settlers(101)_to(2,3)",
            legal_moves=legal_moves
        )

        result = self.parser.parse(parser_input)
        self.assertEqual(result, "unit_move_settlers(101)_to(2,3)")

    def test_fuzzy_matching_partial_match(self):
        """Test partial token matching."""
        legal_moves = [
            "unit_move_settlers(101)_to(2,3)",
            "unit_fortify_warrior(102)",
            "city_production_rome(301)_target(warriors)"
        ]

        # Test partial match with key tokens
        parser_input = parsers.TextParserInput(
            text="move settlers 101",
            legal_moves=legal_moves
        )

        result = self.parser.parse(parser_input)
        self.assertIsNotNone(result)
        self.assertIn("unit_move_settlers", result)

    def test_fuzzy_matching_with_typos(self):
        """Test fuzzy matching with typos using edit distance."""
        legal_moves = [
            "unit_move_settlers(101)_to(2,3)",
            "unit_fortify_warrior(102)",
            "city_production_rome(301)_target(warriors)"
        ]

        # Test with typos
        parser_input = parsers.TextParserInput(
            text="unit_mov_settlrs(101)_to(2,3)",  # typos: mov instead of move, settlrs instead of settlers
            legal_moves=legal_moves
        )

        result = self.parser.parse(parser_input)
        self.assertIsNotNone(result)
        self.assertIn("unit_move_settlers", result)

    def test_similarity_caching(self):
        """Test that similarity calculations are cached for performance."""
        legal_moves = [
            "unit_move_settlers(101)_to(2,3)",
            "unit_fortify_warrior(102)"
        ]

        parser_input = parsers.TextParserInput(
            text="move settlers",
            legal_moves=legal_moves
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
        # Set a small cache size for testing
        self.parser._max_cache_size = 3

        legal_moves = ["unit_move_settlers(101)_to(2,3)"]

        # Add more items than cache size
        for i in range(5):
            parser_input = parsers.TextParserInput(
                text=f"test input {i}",
                legal_moves=legal_moves
            )
            self.parser.parse(parser_input)

        # Cache should not exceed max size
        self.assertLessEqual(len(self.parser._similarity_cache), self.parser._max_cache_size)

    def test_enhanced_similarity_algorithm(self):
        """Test the enhanced similarity calculation with multiple metrics."""
        legal_moves = [
            "unit_move_settlers(101)_to(2,3)",
            "unit_attack_warrior(102)_target(203)",
            "city_production_rome(301)_target(warriors)"
        ]

        # Test action type preference
        parser_input = parsers.TextParserInput(
            text="attack with warrior",
            legal_moves=legal_moves
        )

        result = self.parser.parse(parser_input)
        self.assertIsNotNone(result)
        self.assertIn("unit_attack_warrior", result)

    def test_low_confidence_rejection(self):
        """Test that low confidence matches are rejected."""
        legal_moves = [
            "unit_move_settlers(101)_to(2,3)",
            "city_production_rome(301)_target(warriors)"
        ]

        # Test completely unrelated input
        parser_input = parsers.TextParserInput(
            text="xyz abc def",
            legal_moves=legal_moves
        )

        result = self.parser.parse(parser_input)
        self.assertIsNone(result)


class TestParserChain(unittest.TestCase):
    """Test parser chain integration."""

    def test_parser_chain_creation(self):
        """Test that parser chain is created correctly."""
        chain = create_freeciv_parser_chain()
        self.assertIsInstance(chain, parsers.ChainedMoveParser)
        self.assertGreaterEqual(len(chain._parsers), 2)  # At least rule-based and soft parser

    def test_parser_chain_with_model(self):
        """Test parser chain creation with LLM model."""
        mock_model = Mock()
        chain = create_freeciv_parser_chain(model=mock_model)
        self.assertIsInstance(chain, parsers.ChainedMoveParser)
        self.assertGreaterEqual(len(chain._parsers), 3)  # Rule-based, LLM, and soft parser

    def test_parser_chain_execution_order(self):
        """Test that parser chain executes in correct order."""
        legal_moves = [
            "unit_move_settlers(101)_to(2,3)",
            "city_production_rome(301)_target(warriors)"
        ]

        chain = create_freeciv_parser_chain()

        # Test JSON parsing (should be handled by rule-based parser first)
        json_input = parsers.TextParserInput(
            text='{"action": "unit_move", "unit": 101, "to": [2, 3]}',
            legal_moves=legal_moves
        )

        result = chain.parse(json_input)
        self.assertIsNotNone(result)
        self.assertIn("unit_move", result)

        # Test fuzzy matching (should fall back to soft parser)
        fuzzy_input = parsers.TextParserInput(
            text="move settlers",
            legal_moves=legal_moves
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
        self.assertLess(avg_time_ms, 10, f"JSON parsing took {avg_time_ms:.2f}ms (should be <10ms)")

    def test_fuzzy_matching_performance(self):
        """Test fuzzy matching performance."""
        parser = FreeCivSoftParser()
        legal_moves = [
            f"unit_move_settlers({i})_to({i},{i+1})" for i in range(50)
        ]
        parser_input = parsers.TextParserInput(
            text="move settlers 25",
            legal_moves=legal_moves
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
        self.assertLess(avg_time_ms, 10, f"Fuzzy matching took {avg_time_ms:.2f}ms (should be <10ms)")

    def test_parser_chain_performance(self):
        """Test full parser chain performance."""
        chain = create_freeciv_parser_chain()
        legal_moves = [
            "unit_move_settlers(101)_to(2,3)",
            "unit_fortify_warrior(102)",
            "city_production_rome(301)_target(warriors)"
        ]

        parser_input = parsers.TextParserInput(
            text='{"action": "unit_move", "unit": 101, "to": [2, 3]}',
            legal_moves=legal_moves
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
        self.assertLess(avg_time_ms, 10, f"Parser chain took {avg_time_ms:.2f}ms (should be <10ms)")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def test_malformed_json_handling(self):
        """Test handling of malformed JSON."""
        parser = FreeCivRuleBasedParser()

        malformed_inputs = [
            '{"action": "unit_move", "unit": 101, "to": [3, 5]',  # Missing closing brace
            '{"action": "unit_move" "unit": 101, "to": [3, 5]}',  # Missing comma
            '{action: "unit_move", "unit": 101, "to": [3, 5]}',   # Unquoted key
            '{"action": unit_move, "unit": 101, "to": [3, 5]}',   # Unquoted value
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
        large_text = "x" * 10000 + '{"action": "unit_move", "unit": 101, "to": [3, 5]}'
        parser_input = parsers.TextParserInput(text=large_text)

        # Should handle large inputs gracefully
        result = parser.parse(parser_input)
        self.assertIsNotNone(result)

    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        parser = FreeCivRuleBasedParser()
        unicode_text = '{"action": "unit_move", "unit": 101, "to": [3, 5], "note": "移动单位"}'
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


if __name__ == "__main__":
    unittest.main()