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

    # Error telemetry (PR feedback fix: capture error context for debugging)
    self._error_telemetry: Optional[Dict[str, Any]] = None
    self._error_history: List[Dict[str, Any]] = []

    # Turn state tracking (per handover doc requirements)
    # These enable intelligent end_turn decisions without relying solely on LLM
    self.current_turn = 1
    self.actions_this_turn: List[FreeCivAction] = []
    self.queries_without_action = 0
    self.last_turn_units_exhausted = False

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

      # Capture structured error telemetry (PR feedback fix)
      import traceback
      self._error_telemetry = {
          "error_type": type(e).__name__,
          "error_message": str(e),
          "stack_trace": traceback.format_exc(),
          "observation_context": {
              "turn": observation.get("turn", None),
              "player_id": observation.get("playerID", observation.get("player_id", None)),
              "num_legal_actions": len(observation.get("legalActions", [])),
              "phase": observation.get("phase", None),
              "observation_keys": list(observation.keys()),
          },
          "timestamp": time.time(),
          "execution_time": execution_time,
      }

      absl_logging.error(
          "Action generation failed after %.2fs: %s [%s]",
          execution_time,
          str(e),
          type(e).__name__
      )

      if self.fallback_to_random:
        # Fall back to random legal action
        legal_actions = observation.get("legalActions", [])
        if legal_actions:
          fallback_action = self._rng.choice(legal_actions)

          # Record fallback action details in telemetry
          self._error_telemetry["fallback_action"] = {
              "action": fallback_action,
              "reason": "random_from_legal_actions",
              "legal_actions": legal_actions,
              "rethinking_attempted": self.sampler is not None,
          }

          absl_logging.warning(
              "Falling back to random action: %d (reason: %s)",
              fallback_action,
              "random_from_legal_actions"
          )

          # Add to error history for trend analysis
          self._error_history.append(self._error_telemetry.copy())

          return {"submission": fallback_action}
        else:
          # No legal actions available - record and re-raise
          self._error_telemetry["fallback_action"] = {
              "action": None,
              "reason": "no_legal_actions_available",
              "legal_actions": [],
              "rethinking_attempted": False,
          }
          self._error_history.append(self._error_telemetry.copy())

          absl_logging.error("No legal actions available for fallback")
          raise e
      else:
        # Fallback disabled - record and re-raise
        self._error_telemetry["fallback_action"] = {
            "action": None,
            "reason": "no_fallback_configured",
            "legal_actions": observation.get("legalActions", []),
            "rethinking_attempted": False,
        }
        self._error_history.append(self._error_telemetry.copy())
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
    """Extract FreeCivState from observation (stub implementation).

    This is a lightweight stub that wraps the observation dict for compatibility
    with the OpenSpiel/tournament framework where no proxy_client is available.

    For production FreeCiv3D gameplay with live WebSocket state synchronization,
    use get_action_async() instead, which calls state_synchronizer.sync_state()
    to fetch real-time game state from the FreeCiv3D server via proxy_client.

    See: get_action_async() at line ~684 for the full implementation.

    Args:
      observation: Game observation containing state data

    Returns:
      FreeCivState object for current game state
    """
    # Stub implementation: wraps observation dict directly
    # Production code uses state_synchronizer.sync_state() for live state
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

        # Validate player ID type and value
        if not isinstance(player_id, int):
          raise ValueError(
              f"Player ID must be an integer, got {type(player_id).__name__} "
              f"for key '{key}': {player_id}"
          )
        if player_id < 0:
          raise ValueError(
              f"Player ID must be non-negative, got {player_id} for key '{key}'"
          )

        return player_id

    # Try to extract from state information
    if "state" in observation:
      state_data = observation["state"]
      if isinstance(state_data, dict):
        for key in ["current_player", "active_player", "turn_player"]:
          if key in state_data:
            player_id = state_data[key]

            # Validate player ID from nested state
            if not isinstance(player_id, int):
              raise ValueError(
                  f"Player ID in state.{key} must be an integer, "
                  f"got {type(player_id).__name__}: {player_id}"
              )
            if player_id < 0:
              raise ValueError(
                  f"Player ID in state.{key} must be non-negative, got {player_id}"
              )

            return player_id

    # Check if stored from previous calls
    if hasattr(self, '_cached_player_id'):
      absl_logging.warning(
          "Using cached player ID %d (could not extract from observation)",
          self._cached_player_id
      )
      return self._cached_player_id

    # If we reach here, player ID could not be determined - RAISE ERROR
    # (fixing PR feedback bug: was silently defaulting to 1, causing multi-agent bugs)
    observation_keys = list(observation.keys())
    raise ValueError(
        f"Cannot determine player ID from observation. "
        f"Expected one of: 'playerID', 'player_id', 'current_player', 'agent_id', "
        f"or nested in 'state' object. "
        f"Available observation keys: {observation_keys}. "
        f"Please ensure the observation includes a valid player ID field."
    )

  async def _generate_action_with_llm(
      self,
      observation: Mapping[str, Any],
      state: FreeCivState,
      legal_actions: List[FreeCivAction],
      action_context: Optional[Dict[str, Any]] = None
  ) -> FreeCivAction:
    """Generate action using LLM with context and strategy.

    Args:
      observation: Game observation
      state: Current FreeCiv game state
      legal_actions: List of legal FreeCiv actions
      action_context: Optional context about turn actions. See get_action_async()
          for structure details. When provided, this context is forwarded to
          the prompt builder to generate dynamic warnings for agents.

    Returns:
      Selected FreeCivAction
    """
    # Extract player_id at function start to ensure consistency across all operations
    # This prevents player context loss during rethinking and state operations
    player_id = self._extract_player_id(observation)

    # Log available legal actions for debugging
    absl_logging.debug(
        "Generating action with %d legal options. Examples: %s",
        len(legal_actions),
        [self.action_converter.action_to_string(a) for a in legal_actions[:3]]
    )

    # Build context-aware prompt with action context
    # TODO(AGE-196): Integrate FreeCiv3D strategic guidance into action selection
    # When state._raw_state contains these fields (from llm_optimized format):
    # 1. immediate_priorities: Prioritize actions matching these suggestions
    # 2. threats: Increase defensive action scores when threats detected
    # 3. opportunities: Boost exploration/expansion actions for identified opportunities
    # See Linear issue: https://linear.app/agentclash/issue/AGE-196
    # Example:
    #   if 'immediate_priorities' in state._raw_state:
    #       priorities = state._raw_state['immediate_priorities']
    #       # Adjust action scoring based on priorities
    prompt = self._build_context_aware_prompt(
        observation, state, legal_actions, action_context=action_context
    )

    # Generate response using model (run in thread pool to avoid blocking event loop)
    self._num_model_calls += 1
    response = await asyncio.to_thread(
        self.model.generate_with_text_input, prompt
    )

    # DIAGNOSTIC: Log LLM response for debugging parser issues
    absl_logging.info(
        "🤖 LLM response (first 300 chars): %s",
        response.main_response[:300]
    )
    if len(response.main_response) > 300:
        absl_logging.debug(
            "Full LLM response (%d chars): %s",
            len(response.main_response),
            response.main_response
        )

    # Parse response to FreeCivAction
    selected_action = self._parse_llm_response(
        response.main_response, legal_actions
    )

    # Validate action is legal with detailed logging
    if selected_action not in legal_actions:
      # Log what went wrong for debugging
      absl_logging.info(
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
      # Rethinking not supported for FreeCiv dict-based observations
      # RethinkSampler is designed for OpenSpiel pyspiel.State objects
      # TODO(AGE-XXX): Implement proper rethinking by calling LLM with error feedback
      absl_logging.warning(
          f"LLM generated illegal action: {selected_action} "
          f"(type={selected_action.action_type}, actor={selected_action.actor_id}), "
          f"falling back to first legal action"
      )
      selected_action = legal_actions[0] if legal_actions else None
      if selected_action:
        absl_logging.info(f"Using fallback action: {selected_action}")

    return selected_action

  async def _generate_action_with_rethinking(
      self,
      observation: Mapping[str, Any],
      state: FreeCivState,
      legal_actions: List[FreeCivAction],
      previous_failures: List[Dict[str, Any]],
      max_attempts: int = 3
  ) -> Optional[FreeCivAction]:
    """Generate action with rethinking after previous failures.

    Args:
        observation: Current game observation
        state: FreeCivState object
        legal_actions: List of legal FreeCivAction objects
        previous_failures: List of previous failed actions with reasons
        max_attempts: Maximum rethinking attempts

    Returns:
        FreeCivAction if successful, None if all attempts fail
    """
    absl_logging.info(
        f"Rethinking after {len(previous_failures)} failure(s), "
        f"attempting {max_attempts} more tries"
    )

    for attempt in range(max_attempts):
      # Build rethinking prompt with failure context
      rethink_prompt = self._build_rethink_prompt(
          observation=observation,
          state=state,
          legal_actions=legal_actions,
          previous_failures=previous_failures,
          attempt_number=len(previous_failures) + attempt + 1
      )

      # Generate response with rethinking context
      try:
        response = await asyncio.to_thread(
            self.model.generate_with_text_input,
            rethink_prompt
        )

        # Log response
        absl_logging.info(
            f"🤖 Rethinking attempt {attempt + 1} response: "
            f"{response.main_response[:300]}"
        )

        # Parse action from response
        selected_action = self._parse_llm_response(
            response.main_response, legal_actions
        )

        # Validate against legal actions
        if selected_action in legal_actions:
          absl_logging.info(
              f"✅ Rethinking succeeded on attempt {attempt + 1}: "
              f"{selected_action.action_type}(actor={selected_action.actor_id})"
          )
          return selected_action

        # Record failure for next attempt
        previous_failures.append({
            "action": selected_action,
            "action_type": selected_action.action_type,
            "actor_id": selected_action.actor_id,
            "reason": "Not in legal actions list",
            "attempt": len(previous_failures) + attempt + 1
        })

        absl_logging.warning(
            f"❌ Rethinking attempt {attempt + 1} generated illegal action: "
            f"{selected_action.action_type}(actor={selected_action.actor_id})"
        )

      except Exception as e:
        absl_logging.error(f"Rethinking attempt {attempt + 1} failed: {e}")
        previous_failures.append({
            "error": str(e),
            "reason": "Exception during rethinking",
            "attempt": len(previous_failures) + attempt + 1
        })

    # All rethinking attempts failed - return None
    absl_logging.warning(
        f"All {max_attempts} rethinking attempts failed, "
        f"total failures: {len(previous_failures)}"
    )
    return None

  def _build_rethink_prompt(
      self,
      observation: Mapping[str, Any],
      state: FreeCivState,
      legal_actions: List[FreeCivAction],
      previous_failures: List[Dict[str, Any]],
      attempt_number: int
  ) -> str:
    """Build prompt with rethinking context.

    Includes information about previous failed actions and why they failed.
    """
    # Build base prompt
    base_prompt = self._build_context_aware_prompt(
        observation,
        state,
        legal_actions
    )

    # Add failure context
    failure_context = f"\n\n⚠️ IMPORTANT: Previous Action Failures (Attempt #{attempt_number})\n"
    failure_context += "=" * 60 + "\n"

    for i, failure in enumerate(previous_failures[-3:], 1):  # Show last 3 failures
      failure_context += f"\nFailure {i}:\n"

      if "action_type" in failure:
        failure_context += f"  Action: {failure['action_type']}"
        if "actor_id" in failure:
          failure_context += f"(actor={failure['actor_id']})"
        failure_context += "\n"

      if "reason" in failure:
        failure_context += f"  Reason: {failure['reason']}\n"

      if "server_error" in failure:
        failure_context += f"  Server Error: {failure['server_error']}\n"

      if "error_code" in failure:
        failure_context += f"  Error Code: {failure['error_code']}\n"

    failure_context += "\n" + "=" * 60 + "\n"
    failure_context += "Please carefully choose a DIFFERENT action that:\n"
    failure_context += "1. Is in the legal actions list\n"
    failure_context += "2. Won't repeat previous failures\n"
    failure_context += "3. Makes strategic sense for the current situation\n"
    failure_context += "4. Uses a different unit/city if previous attempts failed\n"

    # Insert failure context before response format section
    if "RESPONSE FORMAT" in base_prompt:
      return base_prompt.replace(
          "\nRESPONSE FORMAT",
          failure_context + "\n\nRESPONSE FORMAT"
      )
    else:
      # Fallback: append at end
      return base_prompt + "\n" + failure_context

  def _build_context_aware_prompt(
      self,
      observation: Mapping[str, Any],
      state: FreeCivState,
      legal_actions: List[FreeCivAction],
      action_context: Optional[Dict[str, Any]] = None
  ) -> tournament_util.ModelTextInput:
    """Build context-aware prompt for LLM.

    Args:
      observation: Game observation
      state: Current game state
      legal_actions: Legal actions available
      action_context: Optional context about turn actions. See get_action_async()
          for structure details. Forwarded to prompt builder for dynamic warnings.

    Returns:
      Formatted prompt for model input
    """
    # Get memory context
    memory_context = self.memory.get_context(max_tokens=1000)

    # Get strategy configuration
    strategy_config = self.strategy_manager.get_strategy_config(self.strategy)

    # Build enhanced prompt with action context
    prompt_text = self.prompt_builder.build_enhanced_prompt(
        observation=observation,
        legal_actions=legal_actions,
        model_name=self.model.model_name,
        strategy_context=strategy_config,
        memory_context=memory_context,
        action_context=action_context,
    )

    return tournament_util.ModelTextInput(prompt_text=prompt_text)

  def _parse_llm_response(
      self, response_text: str, legal_actions: List[FreeCivAction]
  ) -> FreeCivAction:
    """Parse LLM response to FreeCivAction.

    Args:
      response_text: Raw LLM response text (expected to be JSON)
      legal_actions: List of legal actions for validation

    Returns:
      Parsed FreeCivAction
    """
    # Try to parse JSON directly from the response
    try:
      action = self.action_converter.string_to_action(response_text)
      absl_logging.debug("✅ Successfully parsed JSON action: %s", response_text[:200])
      return action
    except ValueError as e:
      # JSON parsing failed - use fallback
      absl_logging.warning(
          "⚠️  JSON parsing failed: %s. Using strategy-aware fallback (strategy=%s)",
          str(e)[:100],
          self.strategy
      )
      absl_logging.warning(
          "LLM response preview: %s",
          response_text[:300]
      )
      absl_logging.debug("Full LLM response: %s", response_text[:500])

      # Choose fallback action based on agent strategy
      selected_fallback = self._choose_strategic_fallback(legal_actions)

      absl_logging.info(
          "📋 Fallback selected: %s (type=%s) based on %s strategy",
          selected_fallback.action_type,
          selected_fallback.action_type,
          self.strategy
      )

      return selected_fallback

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

  def _choose_strategic_fallback(
      self, legal_actions: List[FreeCivAction]
  ) -> FreeCivAction:
    """Choose fallback action based on agent strategy and game state.

    This method provides intelligent fallback when LLM response parsing fails.
    Instead of always defaulting to tech_research, it selects actions that align
    with the agent's strategy (balanced, aggressive_expansion, economic, etc.).

    Args:
      legal_actions: List of legal FreeCiv actions

    Returns:
      Selected fallback FreeCivAction that aligns with strategy

    Strategy priority mappings:
      - aggressive_expansion: Prioritize unit movement, attacks, military production
      - balanced: Mix of movement, production, and research
      - economic: Prioritize city production, improvements, then research
      - defensive: Prioritize fortify, city improvements, research
    """
    if not legal_actions:
      raise ValueError("Cannot choose fallback from empty legal_actions list")

    # Define strategy-based action type priorities
    strategy_priorities = {
        "aggressive_expansion": [
            'unit_move', 'unit_attack', 'unit_explore',
            'city_production', 'tech_research'
        ],
        "balanced": [
            'unit_move', 'city_production', 'tech_research',
            'unit_explore', 'unit_fortify'
        ],
        "economic": [
            'city_production', 'city_build_improvement', 'tech_research',
            'unit_move', 'unit_explore'
        ],
        "defensive": [
            'unit_fortify', 'city_build_improvement', 'city_production',
            'tech_research', 'unit_move'
        ],
    }

    # Get priority list for current strategy (default to balanced)
    priority_list = strategy_priorities.get(
        self.strategy, strategy_priorities["balanced"]
    )

    # Find first action matching priority order
    for action_type in priority_list:
      for action in legal_actions:
        if action.action_type == action_type:
          absl_logging.debug(
              "Strategy fallback: found %s action (priority %d)",
              action_type,
              priority_list.index(action_type) + 1
          )
          return action

    # Ultimate fallback: return first legal action
    absl_logging.warning(
        "No priority action found for strategy %s, using first legal action",
        self.strategy
    )
    return legal_actions[0]

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
      self,
      observation: Mapping[str, Any],
      proxy_client: FreeCivProxyClient,
      action_context: Optional[Dict[str, Any]] = None
  ) -> FreeCivAction:
    """Generate action asynchronously with WebSocket client.

    This method is used for direct integration testing and provides
    access to the full FreeCivAction object rather than just the integer.

    Args:
      observation: Game observation (can be raw dict from get_state() or old observation format)
      proxy_client: WebSocket client for FreeCiv3D communication
      action_context: Optional context about turn actions. Example structure:
          {
              'actions_taken': 5,
              'actions_remaining': 15,
              'max_actions': 20,
              'should_consider_end_turn': False
          }
          When provided, agents receive dynamic warnings about approaching
          the action limit and guidance on when to call end_turn.

    Returns:
      Selected FreeCivAction object
    """
    # OPTIMIZATION: Skip expensive sync_state() if observation is already fresh
    # When run_freeciv_game.py passes cached turn_state dict from get_state(),
    # we can use it directly instead of querying server again (saves 50% of messages)
    # IMPORTANT: Check must match FreeCivState validation requirements (game, map, players, units, cities)
    if isinstance(observation, dict) and all(k in observation for k in ['game', 'map', 'players', 'units', 'cities']):
      # observation is already a complete state dict from get_state()
      # Use it directly to avoid duplicate state query
      state = FreeCivState(observation)
      absl_logging.debug("Using cached state from observation (skipped sync_state)")
    else:
      # Legacy path or incomplete observation: sync state from server
      if isinstance(observation, dict):
        missing = [k for k in ['game', 'map', 'players', 'units', 'cities'] if k not in observation]
        absl_logging.debug(f"Synchronized state from server (observation incomplete, missing: {missing})")
      else:
        absl_logging.debug("Synchronized state from server (observation not a dict)")
      state = await self.state_synchronizer.sync_state(proxy_client, observation)

    # Extract player ID
    player_id = self._extract_player_id(observation)

    # NEW: Update turn state tracking (detects turn changes, resets counters)
    self.on_state_update(observation)
    self.queries_without_action += 1

    # NEW: Check heuristic end_turn condition BEFORE calling LLM
    # This provides fallback logic when units are exhausted, action limit approached,
    # or agent is stuck without making progress
    max_actions = action_context.get('max_actions', 20) if action_context else 20
    if self.should_end_turn(state, player_id, max_actions):
      absl_logging.info(
          "⏭️ Heuristic triggered end_turn: actions=%d/%d, "
          "queries_without_action=%d",
          len(self.actions_this_turn),
          max_actions,
          self.queries_without_action
      )
      end_turn_action = FreeCivAction(
          action_type="end_turn",
          actor_id=player_id,
          target=None,
          parameters={"turn": state.turn},  # Include turn number for PACKET_PLAYER_PHASE_DONE
          source="player",
          confidence=1.0,
          parse_method="heuristic",
          strategic_score=1.0  # High priority when heuristic decides
      )
      self.on_action_taken(end_turn_action)
      self._record_action_in_memory(end_turn_action, observation, state)
      return end_turn_action

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

    # Generate action with context (LLM-based decision)
    selected_action = await self._generate_action_with_llm(
        observation, state, legal_actions, action_context=action_context
    )

    # NEW: Record action in turn history
    self.on_action_taken(selected_action)

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

  def get_error_telemetry(self) -> Optional[Dict[str, Any]]:
    """Get most recent error telemetry for monitoring.

    Returns:
      Dictionary with structured error information, or None if no errors

    Example:
      {
        "error_type": "ValueError",
        "error_message": "Invalid game state",
        "stack_trace": "Traceback (most recent call last)...",
        "observation_context": {
          "turn": 5,
          "player_id": 1,
          "num_legal_actions": 10,
          "phase": "move"
        },
        "fallback_action": {
          "action": 42,
          "reason": "random_from_legal_actions",
          "legal_actions": [40, 41, 42, 43],
          "rethinking_attempted": False
        },
        "timestamp": 1234567890.123,
        "execution_time": 0.52
      }
    """
    return self._error_telemetry

  def get_error_history(self) -> List[Dict[str, Any]]:
    """Get full error history for trend analysis.

    Returns:
      List of error telemetry dictionaries, ordered chronologically

    This is useful for monitoring systems to detect patterns like:
    - Increasing error rates
    - Specific error types that repeat
    - Correlation between errors and game state
    """
    return self._error_history

  def clear_error_telemetry(self) -> None:
    """Clear error telemetry (useful for testing or periodic cleanup)."""
    self._error_telemetry = None
    self._error_history = []

  def on_state_update(self, game_state: Mapping[str, Any]) -> None:
    """Update internal state when receiving new game state.

    Detects turn changes and resets per-turn tracking variables.
    Should be called at the start of each decision cycle.

    Args:
      game_state: Current game state dictionary

    Example:
      >>> agent.on_state_update({"turn": 2, "phase": "movement", ...})
      # Internally resets actions_this_turn if turn changed from 1 to 2
    """
    new_turn = game_state.get('turn', self.current_turn)

    if new_turn > self.current_turn:
      absl_logging.info(
          "🔄 Turn advanced: %d → %d (reset action tracking). "
          "Previous turn: %d actions, %d queries without action",
          self.current_turn,
          new_turn,
          len(self.actions_this_turn),
          self.queries_without_action
      )
      self.current_turn = new_turn
      self.actions_this_turn = []
      self.queries_without_action = 0
      self.last_turn_units_exhausted = False

  def on_action_taken(self, action: FreeCivAction) -> None:
    """Record action in turn history.

    Resets query counter and adds action to turn history (excluding end_turn).
    Should be called immediately after generating an action.

    Args:
      action: FreeCivAction that was just generated/executed

    Example:
      >>> action = FreeCivAction(action_type="unit_move", ...)
      >>> agent.on_action_taken(action)
      # Internally: queries_without_action reset to 0, action added to history
    """
    self.queries_without_action = 0
    if action.action_type != "end_turn":
      self.actions_this_turn.append(action)
      absl_logging.debug(
          "Action recorded: %s (total this turn: %d)",
          action.action_type,
          len(self.actions_this_turn)
      )

  def should_end_turn(
      self,
      state: FreeCivState,
      player_id: int,
      max_actions_per_turn: int = 20
  ) -> bool:
    """Determine if agent should end turn using heuristic logic.

    This provides a fallback mechanism that doesn't rely solely on LLM decisions.
    Agents should end their turn when:
    1. Approaching the action limit (prevents hitting hard cap)
    2. All units have exhausted movement points (no more meaningful moves)
    3. Stuck without making progress (5+ queries without actions)

    Args:
      state: Current FreeCiv game state
      player_id: ID of the player to check
      max_actions_per_turn: Maximum allowed actions per turn (safety limit)

    Returns:
      True if agent should call end_turn, False otherwise

    Example:
      >>> if agent.should_end_turn(state, player_id=1, max_actions_per_turn=20):
      ...     return generate_end_turn_action()
    """
    # Scenario 1: Approaching action limit (leave 1 slot for end_turn)
    if len(self.actions_this_turn) >= max_actions_per_turn - 1:
      absl_logging.info(
          "Heuristic: Ending turn (approaching action limit %d/%d)",
          len(self.actions_this_turn),
          max_actions_per_turn
      )
      return True

    # Scenario 2: All units exhausted (no movement points left)
    my_units = [u for u in state.units.values() if u.owner == player_id]
    if my_units:
      all_exhausted = all(u.moves_left == 0 for u in my_units)
      if all_exhausted and len(self.actions_this_turn) >= 1:
        absl_logging.info(
            "Heuristic: Ending turn (all %d units exhausted, %d actions taken)",
            len(my_units),
            len(self.actions_this_turn)
        )
        self.last_turn_units_exhausted = True
        return True

    # Scenario 3: Stuck without progress (5+ state queries without taking action)
    if self.queries_without_action >= 5:
      absl_logging.warning(
          "Heuristic: Ending turn (stuck without progress - %d queries without action)",
          self.queries_without_action
      )
      return True

    return False
