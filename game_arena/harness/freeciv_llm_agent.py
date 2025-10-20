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

"""FreeCiv LLM Agent implementation following Game Arena patterns.

This module implements the main LLM agent for FreeCiv gameplay, integrating
with the existing Game Arena infrastructure including model generation,
rethinking sampler, and FreeCiv-specific components.

Example usage:
    >>> from game_arena.harness import model_registry
    >>> from game_arena.harness.freeciv_llm_agent import FreeCivLLMAgent
    >>>
    >>> model = model_registry.ModelRegistry.GEMINI_2_5_FLASH.build(
    ...     api_key="your_api_key"
    ... )
    >>> agent = FreeCivLLMAgent(
    ...     model=model,
    ...     strategy="balanced",
    ...     use_rethinking=True
    ... )
    >>>
    >>> observation = {"serializedGameAndState": "...", "legalActions": [1, 2, 3]}
    >>> result = agent(observation, {})
    >>> print(result["submission"])  # Selected action integer
"""

import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Mapping, Optional, TypedDict

from absl import logging as absl_logging

from game_arena.harness import agent, model_generation, rethink, tournament_util
from game_arena.harness.freeciv_action_converter import FreeCivActionConverter
from game_arena.harness.freeciv_memory import GameMemory
from game_arena.harness.freeciv_parsers import FreeCivRuleBasedParser
from game_arena.harness.freeciv_prompt_adapter import FreeCivPromptGeneratorAdapter
from game_arena.harness.freeciv_proxy_client import FreeCivProxyClient
from game_arena.harness.freeciv_state import FreeCivAction, FreeCivState
from game_arena.harness.freeciv_state_sync import FreeCivStateSynchronizer
from game_arena.harness.freeciv_strategy import StrategyManager
from game_arena.harness.prompts.freeciv_prompts import FreeCivPromptBuilder

logger = logging.getLogger(__name__)


# Type definitions for FreeCiv agent
class GameObservationDict(TypedDict, total=False):
  """Type definition for game observation."""
  serializedGameAndState: str
  legalActions: List[int]
  playerID: int


class ActionSubmissionDict(TypedDict):
  """Type definition for action submission."""
  submission: int


class RethinkObservationDict(TypedDict):
  """Type definition for rethinking sampler observation."""
  serializedGameAndState: str
  legalActions: List[int]
  playerID: int


class FreeCivLLMAgent(
    agent.KaggleSpielAgent[agent.KaggleSpielActionWithExtras]
):
  """FreeCiv LLM Agent that orchestrates game decisions using language models.

  This agent integrates with the existing Game Arena infrastructure while
  providing FreeCiv-specific functionality including:
  - Multiple LLM provider support (GPT-5, Claude, DeepSeek, etc.)
  - Memory management with token-aware context compression
  - Strategy adaptation based on game phase and performance
  - Robust error recovery with fallback mechanisms
  - Integration with rethinking sampler for illegal move handling

  Attributes:
    model: The language model for generating responses
    strategy: Current strategy configuration name
    prompt_builder: FreeCiv-specific prompt generator
    action_parser: Parser for LLM responses to FreeCiv actions
    action_converter: Converter between FreeCivAction and action integers
    state_synchronizer: Manages state sync between Game Arena and FreeCiv3D
    memory: Game memory system for context management
    sampler: Optional rethinking sampler for illegal move handling
    strategy_manager: Manages strategy configurations and adaptation
    fallback_to_random: Whether to fall back to random action on failures
  """

  def __init__(
      self,
      model: model_generation.Model,
      strategy: str = "balanced",
      use_rethinking: bool = True,
      max_rethinks: int = 3,
      memory_size: int = 10,
      fallback_to_random: bool = True,
      seed: Optional[int] = None,
  ):
    """Initialize FreeCiv LLM Agent.

    Args:
      model: Language model for generating responses
      strategy: Strategy name ("balanced", "aggressive", "economic")
      use_rethinking: Whether to use rethinking sampler for illegal moves
      max_rethinks: Maximum number of rethinking attempts
      memory_size: Maximum number of turns to keep in memory
      fallback_to_random: Whether to fall back to random actions on failure
      seed: Random seed for reproducible behavior
    """
    super().__init__()

    self.model = model
    self.strategy = strategy
    self.fallback_to_random = fallback_to_random

    # Initialize FreeCiv-specific components
    self.prompt_builder = FreeCivPromptBuilder()
    self.action_parser = FreeCivRuleBasedParser()
    self.action_converter = FreeCivActionConverter()
    self.state_synchronizer = FreeCivStateSynchronizer()
    self.memory = GameMemory(max_size=memory_size)
    self.strategy_manager = StrategyManager()

    # Initialize rethinking sampler if enabled
    if use_rethinking:
      self.sampler = rethink.RethinkSampler(
          model=model,
          strategy=tournament_util.SamplerChoice.RETHINK_WITH_ENV,
          num_max_rethinks=max_rethinks,
          move_parser=self.action_parser,
          legality_parser=self.action_parser,
          game_short_name="freeciv",
          prompt_generator=FreeCivPromptGeneratorAdapter(),
          rethink_template=None,
      )
    else:
      self.sampler = None

    # Initialize random number generator
    self._rng = random.Random(seed)

    # Performance tracking
    self._num_model_calls = 0
    self._total_response_time = 0.0

    absl_logging.info(
        "FreeCivLLMAgent initialized: model=%s, strategy=%s, rethinking=%s",
        model.model_name,
        strategy,
        use_rethinking,
    )

  def __call__(
      self, observation: Mapping[str, Any], info: Mapping[str, Any]
  ) -> agent.KaggleSpielActionWithExtras:
    """Execute agent action selection following KaggleSpielAgent protocol.

    This is the main entry point for the agent, called by the tournament
    framework. It synchronizes the game state, generates an action using
    the LLM, and returns the result in the expected format.

    Args:
      observation: Game observation containing state and legal actions
      info: Additional game information (unused)

    Returns:
      Dictionary with "submission" key containing selected action integer

    Raises:
      Exception: If action generation fails and fallback is disabled
    """
    start_time = time.time()

    try:
      # Run async action generation
      action_int = asyncio.run(self._get_action_async(observation))

      # Track performance
      execution_time = time.time() - start_time
      self._total_response_time += execution_time

      absl_logging.info(
          "Action generated: action=%d, time=%.2fs, calls=%d",
          action_int,
          execution_time,
          self._num_model_calls,
      )

      return {"submission": action_int}

    except Exception as e:
      execution_time = time.time() - start_time
      absl_logging.error(
          "Action generation failed after %.2fs: %s", execution_time, str(e)
      )

      if self.fallback_to_random:
        # Fall back to random legal action
        legal_actions = observation.get("legalActions", [])
        if legal_actions:
          fallback_action = self._rng.choice(legal_actions)
          absl_logging.warning(
              "Falling back to random action: %d", fallback_action
          )
          return {"submission": fallback_action}
        else:
          absl_logging.error("No legal actions available for fallback")
          raise e
      else:
        raise e

  async def _get_action_async(self, observation: Mapping[str, Any]) -> int:
    """Generate action asynchronously with full pipeline.

    Args:
      observation: Game observation from tournament framework

    Returns:
      Action integer for submission
    """
    # Validate observation input for security
    from game_arena.harness.freeciv_proxy_client import (
        validate_observation_size, validate_array_length, validate_player_id
    )

    # Convert to dict if needed and validate size
    observation_dict = dict(observation)
    observation_dict = validate_observation_size(observation_dict)

    # Validate legal actions array
    legal_actions = observation_dict.get("legalActions", [])
    legal_actions = validate_array_length(legal_actions, "legalActions")

    # Validate player ID if present
    if "playerID" in observation_dict:
      player_id = validate_player_id(observation_dict["playerID"])
      observation_dict["playerID"] = player_id

    # Extract game state from observation
    state = await self._extract_game_state(observation)

    # Get legal actions for current player
    player_id = self._extract_player_id(observation)
    legal_freeciv_actions = state.get_legal_actions(player_id)

    if not legal_freeciv_actions:
      raise ValueError(f"No legal actions available for player {player_id}")

    # Generate action using LLM
    selected_action = await self._generate_action_with_llm(
        observation, state, legal_freeciv_actions
    )

    # Convert FreeCivAction to action integer
    action_int = self.action_converter.action_to_int(selected_action, state, player_id)

    # Record action in memory
    self._record_action_in_memory(selected_action, observation, state)

    return action_int

  async def _extract_game_state(
      self, observation: Mapping[str, Any]
  ) -> FreeCivState:
    """Extract FreeCivState from observation.

    Args:
      observation: Game observation containing state data

    Returns:
      FreeCivState object for current game state
    """
    # For now, create state directly from observation
    # In full implementation, this would use state_synchronizer
    # with WebSocket communication to FreeCiv3D server

    state_data = {
        "turn": observation.get("turn", 1),
        "phase": observation.get("phase", "move"),
        "players": observation.get("players", {}),
        "units": observation.get("units", []),
        "cities": observation.get("cities", []),
        "map": observation.get("map", {"tiles": []}),
    }

    return FreeCivState(state_data)

  def _extract_player_id(self, observation: Mapping[str, Any]) -> int:
    """Extract player ID from observation using multiple fallback strategies.

    Args:
      observation: Game observation

    Returns:
      Player ID for current agent

    Raises:
      ValueError: If player ID cannot be determined
    """
    # Try different common observation keys
    for key in ["playerID", "player_id", "current_player", "agent_id"]:
      if key in observation:
        player_id = observation[key]
        if isinstance(player_id, int) and player_id >= 0:
          return player_id

    # Try to extract from state information
    if "state" in observation:
      state_data = observation["state"]
      if isinstance(state_data, dict):
        for key in ["current_player", "active_player", "turn_player"]:
          if key in state_data:
            player_id = state_data[key]
            if isinstance(player_id, int) and player_id >= 0:
              return player_id

    # Check if stored from previous calls
    if hasattr(self, '_cached_player_id'):
      absl_logging.warning(
          "Using cached player ID %d (could not extract from observation)",
          self._cached_player_id
      )
      return self._cached_player_id

    # Last resort: try to infer from available data or use default
    absl_logging.warning(
        "Could not extract player ID from observation, defaulting to 1. "
        "Observation keys: %s", list(observation.keys())
    )

    # Cache the default for consistency
    self._cached_player_id = 1
    return 1

  async def _generate_action_with_llm(
      self,
      observation: Mapping[str, Any],
      state: FreeCivState,
      legal_actions: List[FreeCivAction],
  ) -> FreeCivAction:
    """Generate action using LLM with context and strategy.

    Args:
      observation: Game observation
      state: Current FreeCiv game state
      legal_actions: List of legal FreeCiv actions

    Returns:
      Selected FreeCivAction
    """
    # Log available legal actions for debugging
    absl_logging.debug(
        "Generating action with %d legal options. Examples: %s",
        len(legal_actions),
        [self.action_converter.action_to_string(a) for a in legal_actions[:3]]
    )

    # Build context-aware prompt
    prompt = self._build_context_aware_prompt(observation, state, legal_actions)

    # Generate response using model (synchronous method)
    self._num_model_calls += 1
    response = self.model.generate_with_text_input(prompt)

    # Parse response to FreeCivAction
    selected_action = self._parse_llm_response(
        response.main_response, legal_actions
    )

    # Validate action is legal with detailed logging
    if selected_action not in legal_actions:
      # Log what went wrong for debugging
      absl_logging.warning(
          "LLM generated illegal action: %s (type=%s, actor=%s, target=%s)",
          self.action_converter.action_to_string(selected_action),
          selected_action.action_type,
          selected_action.actor_id,
          selected_action.target
      )
      absl_logging.debug(
          "Legal actions available (%d): %s",
          len(legal_actions),
          [self.action_converter.action_to_string(a) for a in legal_actions[:5]]
      )
      absl_logging.debug("LLM response (first 500 chars): %s", response.main_response[:500])

      # Use rethinking sampler if available
      if self.sampler:
        absl_logging.info("Illegal action generated, using rethinking sampler")
        try:
          # Create observation compatible with rethinking sampler
          rethink_observation = {
              "serializedGameAndState": observation.get("serializedGameAndState", ""),
              "legalActions": [self.action_converter.action_to_int(action, state) for action in legal_actions],
              "playerID": observation.get("playerID", 0)
          }

          # Use rethinking sampler to get a valid action
          rethink_result = await asyncio.to_thread(
              self.sampler,
              rethink_observation,
              {}  # Environment info
          )

          # Convert back to FreeCivAction
          if "submission" in rethink_result:
            action_int = rethink_result["submission"]
            selected_action = self.action_converter.int_to_action(action_int, state)
            absl_logging.info("Rethinking sampler provided valid action")
          else:
            raise ValueError("Rethinking sampler did not return valid submission")

        except Exception as e:
          absl_logging.warning(f"Rethinking sampler failed: {e}, falling back to first legal action")
          selected_action = legal_actions[0]
      else:
        # Fall back to first legal action
        absl_logging.warning(
            "Generated action not legal, falling back to first option"
        )
        selected_action = legal_actions[0]

    return selected_action

  def _build_context_aware_prompt(
      self,
      observation: Mapping[str, Any],
      state: FreeCivState,
      legal_actions: List[FreeCivAction],
  ) -> tournament_util.ModelTextInput:
    """Build context-aware prompt for LLM.

    Args:
      observation: Game observation
      state: Current game state
      legal_actions: Legal actions available

    Returns:
      Formatted prompt for model input
    """
    # Get memory context
    memory_context = self.memory.get_context(max_tokens=1000)

    # Get strategy configuration
    strategy_config = self.strategy_manager.get_strategy_config(self.strategy)

    # Build enhanced prompt
    prompt_text = self.prompt_builder.build_enhanced_prompt(
        observation=observation,
        legal_actions=legal_actions,
        model_name=self.model.model_name,
        strategy_context=strategy_config,
        memory_context=memory_context,
    )

    return tournament_util.ModelTextInput(prompt_text=prompt_text)

  def _parse_llm_response(
      self, response_text: str, legal_actions: List[FreeCivAction]
  ) -> FreeCivAction:
    """Parse LLM response to FreeCivAction.

    Args:
      response_text: Raw LLM response text
      legal_actions: List of legal actions for validation

    Returns:
      Parsed FreeCivAction
    """
    from game_arena.harness import parsers

    # Convert legal actions to string format for parser
    legal_action_strings = [
        self.action_converter.action_to_string(action)
        for action in legal_actions
    ]

    # Create parser input
    parser_input = parsers.TextParserInput(
        text=response_text, legal_moves=legal_action_strings
    )

    # Parse using FreeCiv parser
    parsed_action_string = self.action_parser.parse(parser_input)

    if parsed_action_string:
      # Validate canonical format before converting
      if not self._is_valid_canonical_format(parsed_action_string):
        absl_logging.warning(
            "Parser returned potentially invalid format: %s",
            parsed_action_string
        )
        absl_logging.debug("LLM response excerpt: %s", response_text[:300])

      # Convert back to FreeCivAction
      return self.action_converter.string_to_action(parsed_action_string)
    else:
      # Fallback to first legal action
      absl_logging.warning("Failed to parse LLM response, using fallback")
      absl_logging.debug("LLM response: %s", response_text[:500])
      return legal_actions[0]

  def _is_valid_canonical_format(self, action_string: str) -> bool:
    """Validate that action string follows canonical format.

    Args:
      action_string: Action string to validate

    Returns:
      True if valid canonical format, False otherwise
    """
    import re

    # Basic structure check: word_word(number) with optional _target(...) or _to(...)
    canonical_pattern = r'^[a-z_]+(?:_[a-z]+)?\([^)]+\)(?:_(?:to|target)\([^)]+\))?$'

    if not re.match(canonical_pattern, action_string):
      return False

    # Additional validation: common action types
    valid_prefixes = [
        'tech_research_player',
        'unit_move_',
        'unit_attack_',
        'unit_fortify_',
        'unit_explore_',
        'city_production_',
        'city_build_improvement_',
    ]

    return any(action_string.startswith(prefix) for prefix in valid_prefixes)

  def _record_action_in_memory(
      self,
      action: FreeCivAction,
      observation: Mapping[str, Any],
      state: FreeCivState,
  ) -> None:
    """Record action and result in memory system.

    Args:
      action: Selected FreeCiv action
      observation: Game observation
      state: Current game state
    """
    result = {
        "turn": state.turn,
        "success": True,  # Assume success for now
        "strategy": self.strategy,
        "model_calls": self._num_model_calls,
    }

    self.memory.record_action(action, result)

  def update_strategy(self, game_phase: str, score_relative: float) -> None:
    """Update strategy based on game phase and relative performance.

    Args:
      game_phase: Current game phase ("early", "mid", "late")
      score_relative: Relative score (-1.0 to 1.0, where negative means behind)
    """
    new_strategy = self.strategy_manager.adapt_strategy(
        current_strategy=self.strategy,
        game_phase=game_phase,
        relative_score=score_relative,
    )

    if new_strategy != self.strategy:
      absl_logging.info(
          "Strategy adapted: %s -> %s (phase=%s, score=%.2f)",
          self.strategy,
          new_strategy,
          game_phase,
          score_relative,
      )
      self.strategy = new_strategy

  async def get_action_async(
      self, observation: Mapping[str, Any], proxy_client: FreeCivProxyClient
  ) -> FreeCivAction:
    """Generate action asynchronously with WebSocket client.

    This method is used for direct integration testing and provides
    access to the full FreeCivAction object rather than just the integer.

    Args:
      observation: Game observation
      proxy_client: WebSocket client for FreeCiv3D communication

    Returns:
      Selected FreeCivAction object
    """
    # Synchronize state with FreeCiv3D server
    state = await self.state_synchronizer.sync_state(proxy_client, observation)

    # Extract player ID
    player_id = self._extract_player_id(observation)

    # Get legal actions
    legal_actions = state.get_legal_actions(player_id)

    # If no legal actions, wait and retry once (game might still be initializing)
    if not legal_actions:
      absl_logging.warning(
          "No legal actions for player %d, waiting 2s for game initialization...",
          player_id,
      )
      await asyncio.sleep(2.0)

      # Re-sync state after waiting
      state = await self.state_synchronizer.sync_state(proxy_client, observation)
      legal_actions = state.get_legal_actions(player_id)

      if not legal_actions:
        # After retry, if still no actions, log details and raise
        absl_logging.error(
            "Still no legal actions for player %d after retry. Units: %d, Cities: %d",
            player_id,
            len([u for u in state.units.values() if u.owner == player_id]),
            len([c for c in state.cities.values() if c.owner == player_id]),
        )
        raise ValueError(f"No legal actions for player {player_id}")

    # Generate action
    selected_action = await self._generate_action_with_llm(
        observation, state, legal_actions
    )

    # Record in memory
    self._record_action_in_memory(selected_action, observation, state)

    return selected_action

  def get_performance_stats(self) -> Dict[str, Any]:
    """Get performance statistics for monitoring.

    Returns:
      Dictionary with performance metrics
    """
    avg_response_time = self._total_response_time / max(
        self._num_model_calls, 1
    )

    return {
        "model_calls": self._num_model_calls,
        "total_response_time": self._total_response_time,
        "avg_response_time": avg_response_time,
        "strategy": self.strategy,
        "memory_size": len(self.memory.history),
        "cache_stats": self.memory.get_cache_statistics(),
    }
