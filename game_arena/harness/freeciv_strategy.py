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

"""Strategy management system for FreeCiv LLM Agent.

This module provides strategy configurations and adaptation logic for different
gameplay approaches. Strategies can be dynamically adjusted based on game phase,
relative performance, and other contextual factors.

Example usage:
    >>> from game_arena.harness.freeciv_strategy import StrategyManager
    >>>
    >>> manager = StrategyManager()
    >>> config = manager.get_strategy_config("aggressive")
    >>> print(config["prioritize"])  # ['military_units', 'territory_expansion']
    >>>
    >>> # Adapt strategy based on performance
    >>> new_strategy = manager.adapt_strategy(
    ...     current_strategy="balanced",
    ...     game_phase="late",
    ...     relative_score=-0.3
    ... )
    >>> print(new_strategy)  # "economic_focus"
"""

from typing import Any, Dict, List, Optional

from absl import logging


# Strategy configuration constants
STRATEGY_CONFIGS = {
    'aggressive_expansion': {
        'name': 'aggressive_expansion',
        'description': (
            'Focus on rapid territorial expansion and military dominance'
        ),
        'prioritize': [
            'settler_production',
            'military_units',
            'territory_control',
            'city_founding',
            'unit_movement',
        ],
        'avoid': [
            'wonder_building',
            'cultural_development',
            'defensive_improvements',
        ],
        'risk_tolerance': 0.8,
        'exploration_weight': 0.9,
        'military_weight': 0.9,
        'economic_weight': 0.4,
        'diplomatic_weight': 0.2,
        'early_game_modifier': 1.2,
        'mid_game_modifier': 1.0,
        'late_game_modifier': 0.8,
    },
    'economic_focus': {
        'name': 'economic_focus',
        'description': 'Emphasize economic development and infrastructure',
        'prioritize': [
            'city_improvements',
            'trade_routes',
            'science_development',
            'infrastructure',
            'resource_exploitation',
        ],
        'avoid': ['early_warfare', 'unit_spam', 'aggressive_expansion'],
        'risk_tolerance': 0.3,
        'exploration_weight': 0.6,
        'military_weight': 0.4,
        'economic_weight': 0.9,
        'diplomatic_weight': 0.7,
        'early_game_modifier': 0.9,
        'mid_game_modifier': 1.2,
        'late_game_modifier': 1.1,
    },
    'balanced': {
        'name': 'balanced',
        'description': 'Balanced approach adapting to circumstances',
        'prioritize': [
            'adaptive_response',
            'situational_awareness',
            'flexible_development',
        ],
        'avoid': ['extreme_specialization', 'neglecting_defense'],
        'risk_tolerance': 0.5,
        'exploration_weight': 0.7,
        'military_weight': 0.6,
        'economic_weight': 0.7,
        'diplomatic_weight': 0.5,
        'early_game_modifier': 1.0,
        'mid_game_modifier': 1.0,
        'late_game_modifier': 1.0,
    },
    'defensive_turtle': {
        'name': 'defensive_turtle',
        'description': 'Focus on strong defense and careful development',
        'prioritize': [
            'city_defenses',
            'defensive_units',
            'secure_borders',
            'infrastructure_development',
            'technological_advancement',
        ],
        'avoid': ['aggressive_expansion', 'early_conflicts', 'risky_ventures'],
        'risk_tolerance': 0.2,
        'exploration_weight': 0.4,
        'military_weight': 0.8,  # High, but defensive
        'economic_weight': 0.8,
        'diplomatic_weight': 0.6,
        'early_game_modifier': 1.1,
        'mid_game_modifier': 1.0,
        'late_game_modifier': 0.9,
    },
    'science_victory': {
        'name': 'science_victory',
        'description': 'Focus on technological superiority and science victory',
        'prioritize': [
            'research_facilities',
            'science_specialists',
            'technological_wonders',
            'resource_management',
            'peaceful_development',
        ],
        'avoid': [
            'military_conflicts',
            'territorial_disputes',
            'culture_focus',
        ],
        'risk_tolerance': 0.4,
        'exploration_weight': 0.5,
        'military_weight': 0.3,
        'economic_weight': 0.8,
        'diplomatic_weight': 0.7,
        'early_game_modifier': 0.8,
        'mid_game_modifier': 1.1,
        'late_game_modifier': 1.3,
    },
    'opportunistic': {
        'name': 'opportunistic',
        'description': 'Adapt quickly to exploit opportunities and weaknesses',
        'prioritize': [
            'quick_adaptation',
            'exploit_weaknesses',
            'flexible_military',
            'rapid_response',
            'situational_advantage',
        ],
        'avoid': [
            'long_term_commitments',
            'rigid_planning',
            'predictable_patterns',
        ],
        'risk_tolerance': 0.7,
        'exploration_weight': 0.8,
        'military_weight': 0.7,
        'economic_weight': 0.6,
        'diplomatic_weight': 0.4,
        'early_game_modifier': 1.0,
        'mid_game_modifier': 1.1,
        'late_game_modifier': 1.0,
    },
}


class StrategyManager:
  """Manages strategy configurations and adaptive strategy selection.

  This class provides strategy configurations for different gameplay approaches
  and implements logic for dynamically adapting strategies based on game
  conditions and performance.

  Attributes:
    strategies: Dictionary of available strategy configurations
    adaptation_history: List of strategy adaptations for analysis
  """

  def __init__(self):
    """Initialize strategy manager."""
    self.strategies = STRATEGY_CONFIGS.copy()
    self.adaptation_history: List[Dict[str, Any]] = []

    logging.debug(
        'StrategyManager initialized with %d strategies', len(self.strategies)
    )

  def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
    """Get configuration for a specific strategy.

    Args:
      strategy_name: Name of the strategy to retrieve

    Returns:
      Strategy configuration dictionary

    Raises:
      ValueError: If strategy name is not found
    """
    if strategy_name not in self.strategies:
      available = list(self.strategies.keys())
      raise ValueError(
          f"Strategy '{strategy_name}' not found. Available: {available}"
      )

    config = self.strategies[strategy_name].copy()

    logging.debug('Retrieved strategy config: %s', strategy_name)
    return config

  def adapt_strategy(
      self,
      current_strategy: str,
      game_phase: str,
      relative_score: float,
      turn_number: Optional[int] = None,
      context: Optional[Dict[str, Any]] = None,
  ) -> str:
    """Adapt strategy based on game conditions and performance.

    Args:
      current_strategy: Current strategy name
      game_phase: Game phase ("early", "mid", "late")
      relative_score: Relative score from -1.0 (far behind) to 1.0 (far ahead)
      turn_number: Optional turn number for additional context
      context: Optional additional context information

    Returns:
      Recommended strategy name (may be same as current)
    """
    adaptation_reasons = []

    # Get current strategy config
    current_config = self.get_strategy_config(current_strategy)

    # Calculate adaptation score for each strategy
    strategy_scores = {}
    for strategy_name, strategy_config in self.strategies.items():
      score = self._calculate_strategy_score(
          strategy_config=strategy_config,
          game_phase=game_phase,
          relative_score=relative_score,
          current_strategy=current_strategy,
      )
      strategy_scores[strategy_name] = score

    # Find best strategy
    best_strategy = max(
        strategy_scores.keys(), key=lambda x: strategy_scores[x]
    )
    best_score = strategy_scores[best_strategy]
    current_score = strategy_scores[current_strategy]

    # Only change if improvement is significant
    adaptation_threshold = 0.1
    if best_score > current_score + adaptation_threshold:
      adaptation_reasons.append(
          f'Better fit for current conditions (score: {best_score:.2f} vs'
          f' {current_score:.2f})'
      )

      # Log adaptation
      self._log_adaptation(
          old_strategy=current_strategy,
          new_strategy=best_strategy,
          game_phase=game_phase,
          relative_score=relative_score,
          reasons=adaptation_reasons,
          turn_number=turn_number,
      )

      return best_strategy
    else:
      logging.debug(
          'Keeping current strategy %s (score: %.2f, best alternative: %s'
          ' %.2f)',
          current_strategy,
          current_score,
          best_strategy,
          best_score,
      )
      return current_strategy

  def _calculate_strategy_score(
      self,
      strategy_config: Dict[str, Any],
      game_phase: str,
      relative_score: float,
      current_strategy: str,
  ) -> float:
    """Calculate fitness score for a strategy given current conditions.

    Args:
      strategy_config: Strategy configuration to evaluate
      game_phase: Current game phase
      relative_score: Relative performance score
      current_strategy: Current strategy name for continuity bonus

    Returns:
      Fitness score for the strategy
    """
    base_score = 0.5

    # Apply game phase modifiers
    phase_modifiers = {
        'early': strategy_config.get('early_game_modifier', 1.0),
        'mid': strategy_config.get('mid_game_modifier', 1.0),
        'late': strategy_config.get('late_game_modifier', 1.0),
    }
    phase_modifier = phase_modifiers.get(game_phase, 1.0)
    base_score *= phase_modifier

    # Adjust based on relative performance
    if relative_score < -0.5:  # Far behind
      # Favor aggressive or opportunistic strategies
      if strategy_config['name'] in ['aggressive_expansion', 'opportunistic']:
        base_score += 0.3
      elif strategy_config['name'] in ['defensive_turtle', 'science_victory']:
        base_score -= 0.2

    elif relative_score > 0.3:  # Ahead
      # Favor defensive or consolidation strategies
      if strategy_config['name'] in ['defensive_turtle', 'economic_focus']:
        base_score += 0.2
      elif strategy_config['name'] == 'aggressive_expansion':
        base_score -= 0.1

    else:  # Close game
      # Favor balanced or adaptive strategies
      if strategy_config['name'] in ['balanced', 'opportunistic']:
        base_score += 0.1

    # Risk tolerance adjustment
    risk_tolerance = strategy_config.get('risk_tolerance', 0.5)
    if relative_score < 0:  # Behind, may need more risk
      risk_bonus = risk_tolerance * 0.2
    else:  # Ahead, may want less risk
      risk_bonus = (1.0 - risk_tolerance) * 0.1
    base_score += risk_bonus

    # Continuity bonus (slight preference for current strategy)
    if strategy_config['name'] == current_strategy:
      base_score += 0.05

    return max(0.0, min(1.0, base_score))

  def _log_adaptation(
      self,
      old_strategy: str,
      new_strategy: str,
      game_phase: str,
      relative_score: float,
      reasons: List[str],
      turn_number: Optional[int] = None,
  ) -> None:
    """Log strategy adaptation for analysis.

    Args:
      old_strategy: Previous strategy name
      new_strategy: New strategy name
      game_phase: Current game phase
      relative_score: Relative performance score
      reasons: List of adaptation reasons
      turn_number: Optional turn number
    """
    adaptation_record = {
        'old_strategy': old_strategy,
        'new_strategy': new_strategy,
        'game_phase': game_phase,
        'relative_score': relative_score,
        'reasons': reasons,
        'turn_number': turn_number,
        'timestamp': None,  # Would use time.time() in full implementation
    }

    self.adaptation_history.append(adaptation_record)

    logging.info(
        'Strategy adapted: %s -> %s (phase: %s, score: %.2f, reasons: %s)',
        old_strategy,
        new_strategy,
        game_phase,
        relative_score,
        '; '.join(reasons),
    )

  def get_available_strategies(self) -> List[str]:
    """Get list of available strategy names.

    Returns:
      List of strategy names
    """
    return list(self.strategies.keys())

  def get_strategy_description(self, strategy_name: str) -> str:
    """Get human-readable description of a strategy.

    Args:
      strategy_name: Name of the strategy

    Returns:
      Strategy description string

    Raises:
      ValueError: If strategy name is not found
    """
    config = self.get_strategy_config(strategy_name)
    return config.get('description', f'Strategy: {strategy_name}')

  def compare_strategies(
      self, strategy1: str, strategy2: str
  ) -> Dict[str, Any]:
    """Compare two strategies across different dimensions.

    Args:
      strategy1: First strategy name
      strategy2: Second strategy name

    Returns:
      Dictionary with comparison results

    Raises:
      ValueError: If either strategy name is not found
    """
    config1 = self.get_strategy_config(strategy1)
    config2 = self.get_strategy_config(strategy2)

    comparison = {
        'strategies': [strategy1, strategy2],
        'risk_tolerance': [
            config1.get('risk_tolerance', 0.5),
            config2.get('risk_tolerance', 0.5),
        ],
        'military_focus': [
            config1.get('military_weight', 0.5),
            config2.get('military_weight', 0.5),
        ],
        'economic_focus': [
            config1.get('economic_weight', 0.5),
            config2.get('economic_weight', 0.5),
        ],
        'exploration_focus': [
            config1.get('exploration_weight', 0.5),
            config2.get('exploration_weight', 0.5),
        ],
    }

    return comparison

  def recommend_strategy_for_situation(
      self,
      game_phase: str,
      relative_score: float,
      military_threat: bool = False,
      resource_scarcity: bool = False,
      diplomatic_isolation: bool = False,
  ) -> str:
    """Recommend strategy for a specific situation.

    Args:
      game_phase: Current game phase
      relative_score: Relative performance score
      military_threat: Whether facing military threats
      resource_scarcity: Whether resources are scarce
      diplomatic_isolation: Whether diplomatically isolated

    Returns:
      Recommended strategy name
    """
    # Start with situation-specific recommendations
    if military_threat:
      if relative_score < 0:
        return 'opportunistic'  # Need to find weaknesses
      else:
        return 'defensive_turtle'  # Protect lead

    if resource_scarcity:
      if game_phase == 'early':
        return 'aggressive_expansion'  # Grab resources
      else:
        return 'economic_focus'  # Optimize what we have

    if diplomatic_isolation:
      return 'defensive_turtle'  # Turtle until relations improve

    # General recommendations based on phase and score
    if game_phase == 'early':
      if relative_score >= 0:
        return 'aggressive_expansion'
      else:
        return 'balanced'

    elif game_phase == 'mid':
      if relative_score > 0.3:
        return 'economic_focus'
      elif relative_score < -0.3:
        return 'opportunistic'
      else:
        return 'balanced'

    else:  # late game
      if relative_score > 0.2:
        return 'defensive_turtle'
      elif relative_score < -0.4:
        return 'aggressive_expansion'
      else:
        return 'science_victory'

  def get_adaptation_history(self) -> List[Dict[str, Any]]:
    """Get history of strategy adaptations.

    Returns:
      List of adaptation records
    """
    return self.adaptation_history.copy()

  def clear_adaptation_history(self) -> None:
    """Clear adaptation history."""
    self.adaptation_history.clear()
    logging.debug('Strategy adaptation history cleared')

  def validate_strategy_config(self, config: Dict[str, Any]) -> bool:
    """Validate a strategy configuration.

    Args:
      config: Strategy configuration to validate

    Returns:
      True if configuration is valid

    Raises:
      ValueError: If configuration is invalid
    """
    required_fields = [
        'name',
        'description',
        'prioritize',
        'avoid',
        'risk_tolerance',
    ]

    for field in required_fields:
      if field not in config:
        raise ValueError(f'Missing required field: {field}')

    # Validate risk tolerance
    risk = config['risk_tolerance']
    if not isinstance(risk, (int, float)) or not 0 <= risk <= 1:
      raise ValueError(f'Invalid risk_tolerance: {risk} (must be 0-1)')

    # Validate lists
    for list_field in ['prioritize', 'avoid']:
      if not isinstance(config[list_field], list):
        raise ValueError(f'{list_field} must be a list')

    return True
