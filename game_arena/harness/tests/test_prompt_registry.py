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

"""Tests for the prompt builder registry system."""

import unittest
from game_arena.harness.prompts import (
    PromptBuilderRegistry,
    get_prompt_builder,
    FreeCivPromptBuilder,
    ChessPromptBuilder,
    GoPromptBuilder
)


class TestPromptBuilderRegistry(unittest.TestCase):
    """Test cases for the prompt builder registry system."""

    def test_supported_games(self):
        """Test that all expected games are supported."""
        supported_games = PromptBuilderRegistry.list_supported_games()
        self.assertIn("freeciv", supported_games)
        self.assertIn("chess", supported_games)
        self.assertIn("go", supported_games)

    def test_is_supported(self):
        """Test the is_supported method."""
        self.assertTrue(PromptBuilderRegistry.is_supported("freeciv"))
        self.assertTrue(PromptBuilderRegistry.is_supported("chess"))
        self.assertTrue(PromptBuilderRegistry.is_supported("go"))
        self.assertFalse(PromptBuilderRegistry.is_supported("unknown_game"))

    def test_case_insensitive_support(self):
        """Test that game names are case insensitive."""
        self.assertTrue(PromptBuilderRegistry.is_supported("FREECIV"))
        self.assertTrue(PromptBuilderRegistry.is_supported("Chess"))
        self.assertTrue(PromptBuilderRegistry.is_supported("GO"))

    def test_get_builder_class(self):
        """Test getting builder classes."""
        freeciv_class = PromptBuilderRegistry.get_builder("freeciv")
        self.assertEqual(freeciv_class, FreeCivPromptBuilder)

        chess_class = PromptBuilderRegistry.get_builder("chess")
        self.assertEqual(chess_class, ChessPromptBuilder)

        go_class = PromptBuilderRegistry.get_builder("go")
        self.assertEqual(go_class, GoPromptBuilder)

        unknown_class = PromptBuilderRegistry.get_builder("unknown")
        self.assertIsNone(unknown_class)

    def test_create_builder_instances(self):
        """Test creating builder instances."""
        freeciv_builder = PromptBuilderRegistry.create_builder("freeciv")
        self.assertIsInstance(freeciv_builder, FreeCivPromptBuilder)

        chess_builder = PromptBuilderRegistry.create_builder("chess")
        self.assertIsInstance(chess_builder, ChessPromptBuilder)

        go_builder = PromptBuilderRegistry.create_builder("go")
        self.assertIsInstance(go_builder, GoPromptBuilder)

        unknown_builder = PromptBuilderRegistry.create_builder("unknown")
        self.assertIsNone(unknown_builder)

    def test_factory_function(self):
        """Test the get_prompt_builder factory function."""
        freeciv_builder = get_prompt_builder("freeciv")
        self.assertIsInstance(freeciv_builder, FreeCivPromptBuilder)

        chess_builder = get_prompt_builder("chess")
        self.assertIsInstance(chess_builder, ChessPromptBuilder)

        go_builder = get_prompt_builder("go")
        self.assertIsInstance(go_builder, GoPromptBuilder)

        unknown_builder = get_prompt_builder("unknown")
        self.assertIsNone(unknown_builder)

    def test_freeciv_builder_functionality(self):
        """Test that FreeCiv builder works through registry."""
        builder = get_prompt_builder("freeciv")
        self.assertIsNotNone(builder)

        # Test prompt generation
        observation = {
            "turn": 50,
            "players": {1: {"score": 100, "name": "Romans"}},
            "units": [],
            "cities": []
        }

        prompt = builder.build_enhanced_prompt(observation, [], "gpt-5")
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)
        self.assertIn("GAME ANALYSIS & STRATEGY", prompt)
        self.assertIn("ENTERTAINMENT DIRECTIVE", prompt)

    def test_chess_and_go_use_original_system(self):
        """Test that chess and go correctly redirect to original system."""
        chess_builder = get_prompt_builder("chess")
        go_builder = get_prompt_builder("go")

        self.assertIsNotNone(chess_builder)
        self.assertIsNotNone(go_builder)

        # These should raise NotImplementedError with specific message about using original system
        with self.assertRaises(NotImplementedError) as cm:
            chess_builder.build_enhanced_prompt({}, [], "gpt-5")
        self.assertIn("original PromptGeneratorText", str(cm.exception))

        with self.assertRaises(NotImplementedError) as cm:
            go_builder.build_enhanced_prompt({}, [], "gpt-5")
        self.assertIn("original PromptGeneratorText", str(cm.exception))


if __name__ == "__main__":
    unittest.main()