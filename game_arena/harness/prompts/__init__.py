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

"""Prompt builders for game-specific prompt generation.

This module provides a registry-based approach for managing prompt builders
across different games, ensuring proper separation of concerns and easy
extensibility for new games.
"""

from typing import Dict, Type, Optional
from game_arena.harness.prompts.base import BasePromptBuilder


class PromptBuilderRegistry:
    """Registry for game-specific prompt builders.

    This registry maintains a mapping of game names to their corresponding
    prompt builder classes, enabling clean separation and easy extensibility.
    """

    _builders: Dict[str, Type[BasePromptBuilder]] = {}

    @classmethod
    def register(cls, game_name: str, builder_class: Type[BasePromptBuilder]) -> None:
        """Register a prompt builder for a specific game.

        Args:
            game_name: Name of the game (e.g., 'freeciv', 'chess', 'go')
            builder_class: Prompt builder class for the game
        """
        cls._builders[game_name.lower()] = builder_class

    @classmethod
    def get_builder(cls, game_name: str) -> Optional[Type[BasePromptBuilder]]:
        """Get a prompt builder class for a specific game.

        Args:
            game_name: Name of the game

        Returns:
            Prompt builder class if registered, None otherwise
        """
        return cls._builders.get(game_name.lower())

    @classmethod
    def create_builder(cls, game_name: str) -> Optional[BasePromptBuilder]:
        """Create an instance of a prompt builder for a specific game.

        Args:
            game_name: Name of the game

        Returns:
            Prompt builder instance if registered, None otherwise
        """
        builder_class = cls.get_builder(game_name)
        if builder_class:
            return builder_class()
        return None

    @classmethod
    def list_supported_games(cls) -> list[str]:
        """List all supported games.

        Returns:
            List of supported game names
        """
        return list(cls._builders.keys())

    @classmethod
    def is_supported(cls, game_name: str) -> bool:
        """Check if a game is supported.

        Args:
            game_name: Name of the game

        Returns:
            True if the game is supported, False otherwise
        """
        return game_name.lower() in cls._builders


# Import and register game-specific prompt builders
from game_arena.harness.prompts.freeciv_prompts import FreeCivPromptBuilder
from game_arena.harness.prompts.chess_prompts import ChessPromptBuilder
from game_arena.harness.prompts.go_prompts import GoPromptBuilder

# Register prompt builders for all supported games
PromptBuilderRegistry.register("freeciv", FreeCivPromptBuilder)
PromptBuilderRegistry.register("chess", ChessPromptBuilder)
PromptBuilderRegistry.register("go", GoPromptBuilder)

# Convenience exports for backward compatibility
FreeCivPromptBuilder = FreeCivPromptBuilder
ChessPromptBuilder = ChessPromptBuilder
GoPromptBuilder = GoPromptBuilder

# Factory function for easy access
def get_prompt_builder(game_name: str) -> Optional[BasePromptBuilder]:
    """Factory function to get a prompt builder for a specific game.

    Args:
        game_name: Name of the game

    Returns:
        Prompt builder instance if supported, None otherwise

    Example:
        >>> builder = get_prompt_builder('freeciv')
        >>> if builder:
        ...     prompt = builder.build_enhanced_prompt(obs, actions, 'gpt-5')
    """
    return PromptBuilderRegistry.create_builder(game_name)


__all__ = [
    "BasePromptBuilder",
    "PromptBuilderRegistry",
    "FreeCivPromptBuilder",
    "ChessPromptBuilder",
    "GoPromptBuilder",
    "get_prompt_builder"
]
