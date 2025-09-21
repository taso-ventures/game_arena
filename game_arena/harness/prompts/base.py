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

"""Base classes and utilities for game-specific prompt builders.

This module provides the foundation for creating consistent, entertaining,
and strategically sound prompts across different games and models.
"""

import os
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

# Type variable for game-specific action types
ActionT = TypeVar('ActionT')


class ConfigLoader:
    """Utility class for loading YAML configuration files."""

    _instance = None
    _configs = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the config loader with default paths."""
        if not hasattr(self, '_initialized'):
            self._config_dir = Path(__file__).parent.parent.parent.parent / "config" / "prompts"
            self._initialized = True

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a configuration file by name.

        Args:
            config_name: Name of the config file (without .yaml extension)

        Returns:
            Dictionary containing the configuration data

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        if config_name in self._configs:
            return self._configs[config_name]

        config_path = self._config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            self._configs[config_name] = config_data
            return config_data
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")


class MemoryContext:
    """Manages memory context for long-term strategic reasoning."""

    def __init__(self, max_turns: int = 10):
        """Initialize memory context with turn limit.

        Args:
            max_turns: Maximum number of turns to remember
        """
        self.max_turns = max_turns
        self.turn_history: List[Dict[str, Any]] = []
        self.strategic_notes: List[str] = []
        self.key_events: List[str] = []

    def add_turn(self, turn_data: Dict[str, Any]) -> None:
        """Add a turn's data to memory.

        Args:
            turn_data: Dictionary containing turn information
        """
        self.turn_history.append(turn_data)
        if len(self.turn_history) > self.max_turns:
            self.turn_history.pop(0)

    def add_strategic_note(self, note: str) -> None:
        """Add a strategic note for future reference.

        Args:
            note: Strategic observation or decision rationale
        """
        self.strategic_notes.append(note)
        if len(self.strategic_notes) > self.max_turns:
            self.strategic_notes.pop(0)

    def add_key_event(self, event: str) -> None:
        """Add a significant game event to memory.

        Args:
            event: Description of an important game event
        """
        self.key_events.append(event)
        if len(self.key_events) > self.max_turns:
            self.key_events.pop(0)

    def get_context_summary(self) -> str:
        """Generate a summary of recent context for prompt inclusion.

        Returns:
            Formatted string summarizing recent strategic context
        """
        if not (self.turn_history or self.strategic_notes or self.key_events):
            return "No previous context available. This is the beginning of your strategic journey."

        summary_parts = []

        if self.key_events:
            summary_parts.append("🔥 KEY EVENTS:")
            for event in self.key_events[-3:]:  # Last 3 events
                summary_parts.append(f"• {event}")

        if self.strategic_notes:
            summary_parts.append("\n📝 STRATEGIC INSIGHTS:")
            for note in self.strategic_notes[-3:]:  # Last 3 notes
                summary_parts.append(f"• {note}")

        if self.turn_history:
            summary_parts.append(f"\n📊 RECENT PERFORMANCE:")
            recent_turns = self.turn_history[-3:]
            for turn_data in recent_turns:
                turn_num = turn_data.get('turn', '?')
                action = turn_data.get('action', 'unknown')
                outcome = turn_data.get('outcome', 'pending')
                summary_parts.append(f"• Turn {turn_num}: {action} → {outcome}")

        return "\n".join(summary_parts)


class BasePromptBuilder(ABC):
    """Abstract base class for game-specific prompt builders.

    This class provides the common interface and utilities for building
    entertaining, strategically sound prompts across different games.
    All game-specific prompt builders should inherit from this class.
    """

    def __init__(self, game_name: str):
        """Initialize the base prompt builder.

        Args:
            game_name: Name of the game (e.g., 'freeciv', 'chess', 'go')
        """
        self.game_name = game_name
        self.config_loader = ConfigLoader()
        self.model_configs = self.config_loader.load_config('model_configs')
        self.game_templates = self.config_loader.load_config('game_templates')
        self.memory_context = MemoryContext()

    @abstractmethod
    def build_enhanced_prompt(
        self,
        observation: Dict[str, Any],
        legal_actions: List[ActionT],
        model_name: str,
        **kwargs
    ) -> str:
        """Build an enhanced prompt for the specific game.

        Args:
            observation: Current game state observation
            legal_actions: List of legal actions available
            model_name: Target model name for formatting
            **kwargs: Additional game-specific parameters

        Returns:
            Formatted prompt string optimized for the target model

        Raises:
            ValueError: If model_name is not supported
            KeyError: If required observation data is missing
        """
        pass

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Model configuration dictionary

        Raises:
            ValueError: If model is not supported
        """
        if model_name not in self.model_configs['models']:
            supported_models = list(self.model_configs['models'].keys())
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Supported models: {supported_models}"
            )
        return self.model_configs['models'][model_name]

    def format_for_model(self, content: str, model_name: str) -> str:
        """Format content according to model preferences.

        Args:
            content: Raw content to format
            model_name: Target model name

        Returns:
            Formatted content string
        """
        model_config = self.get_model_config(model_name)
        style = model_config.get('style', 'structured')

        # Apply style-specific formatting
        style_config = self.model_configs.get('styles', {}).get(style, {})

        if style_config.get('use_headers', True):
            # Headers are already included in templates
            pass

        if not style_config.get('bullet_points', True):
            # Convert bullet points to numbered lists or plain text
            content = content.replace('•', '-')

        return content

    def get_response_format_instruction(self, model_name: str) -> str:
        """Get response format instruction for a model.

        Args:
            model_name: Target model name

        Returns:
            Format instruction string
        """
        model_config = self.get_model_config(model_name)
        style = model_config.get('style', 'structured')
        return self.game_templates['response_formats'].get(style, '')

    def determine_game_phase(self, observation: Dict[str, Any]) -> str:
        """Determine the current game phase from observation.

        This is a default implementation that can be overridden by
        game-specific builders for more sophisticated phase detection.

        Args:
            observation: Current game state observation

        Returns:
            Game phase string ('early_game', 'mid_game', 'late_game')
        """
        turn = observation.get('turn', 0)
        if turn < 50:
            return 'early_game'
        elif turn < 150:
            return 'mid_game'
        else:
            return 'late_game'

    def get_long_term_strategy(self, observation: Dict[str, Any]) -> str:
        """Generate long-term strategy guidance.

        Args:
            observation: Current game state observation

        Returns:
            Long-term strategy description
        """
        # This is a base implementation - games should override for specifics
        strategy_frameworks = self.game_templates.get('strategy_frameworks', {})

        # Simple heuristic - games should implement more sophisticated logic
        score = observation.get('score', 0)
        military_strength = observation.get('military_strength', 0.5)

        if military_strength > 0.7:
            return strategy_frameworks.get('aggressive_expansion',
                'Focus on aggressive expansion and territorial control.')
        elif score > 1000:
            return strategy_frameworks.get('cultural_dominance',
                'Build a magnificent civilization that inspires others.')
        else:
            return strategy_frameworks.get('balanced_growth',
                'Pursue steady, balanced development across all areas.')

    def validate_model_name(self, model_name: str) -> None:
        """Validate that the model name is supported.

        Args:
            model_name: Model name to validate

        Raises:
            ValueError: If model name is invalid or unsupported
        """
        if not isinstance(model_name, str):
            raise ValueError(f"Model name must be a string, got {type(model_name)}")

        if model_name not in self.model_configs['models']:
            supported_models = list(self.model_configs['models'].keys())
            raise ValueError(
                f"Unsupported model '{model_name}'. "
                f"Supported models: {supported_models}"
            )

    def update_memory_context(
        self,
        turn_data: Dict[str, Any],
        strategic_note: Optional[str] = None,
        key_event: Optional[str] = None
    ) -> None:
        """Update the memory context with new information.

        Args:
            turn_data: Data from the current turn
            strategic_note: Optional strategic note to remember
            key_event: Optional key event to remember
        """
        self.memory_context.add_turn(turn_data)

        if strategic_note:
            self.memory_context.add_strategic_note(strategic_note)

        if key_event:
            self.memory_context.add_key_event(key_event)