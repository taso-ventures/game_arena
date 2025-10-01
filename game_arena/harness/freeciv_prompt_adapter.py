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

"""Prompt adapter for integrating FreeCiv prompts with rethinking sampler.

This module provides an adapter that bridges the FreeCivPromptBuilder with
the existing Game Arena prompt generation infrastructure, particularly for
integration with the rethinking sampler.

Example usage:
    >>> from game_arena.harness.freeciv_prompt_adapter import FreeCivPromptGeneratorAdapter
    >>> from game_arena.harness import rethink
    >>>
    >>> adapter = FreeCivPromptGeneratorAdapter()
    >>> sampler = rethink.RethinkSampler(
    ...     model=model,
    ...     prompt_generator=adapter,
    ...     # ... other parameters
    ... )
"""

from typing import Any, Dict, List, Mapping, Optional

from absl import logging

from game_arena.harness import prompt_generation, tournament_util
from game_arena.harness.prompt_templates import PromptTemplate
from game_arena.harness.freeciv_state import FreeCivAction
from game_arena.harness.prompts.freeciv_prompts import FreeCivPromptBuilder


class FreeCivPromptGeneratorAdapter(prompt_generation.PromptGeneratorText):
  """Adapter for integrating FreeCivPromptBuilder with rethinking sampler.

  This class adapts the FreeCiv-specific prompt builder to work with the
  existing Game Arena prompt generation interface, enabling integration
  with the rethinking sampler and other framework components.

  Attributes:
    prompt_builder: FreeCiv-specific prompt builder
    default_strategy: Default strategy configuration for prompts
  """

  def __init__(self, default_strategy: str = "balanced"):
    """Initialize FreeCiv prompt generator adapter.

    Args:
      default_strategy: Default strategy to use when none is specified
    """
    super().__init__()
    self.prompt_builder = FreeCivPromptBuilder()
    self.default_strategy = default_strategy

    logging.debug("FreeCivPromptGeneratorAdapter initialized")

  def generate_prompt_with_text_only(
      self,
      prompt_template: PromptTemplate,
      game_short_name: str,
      **prompt_substitutions: Any,
  ) -> tournament_util.ModelTextInput:
    """Generate prompt using FreeCiv-specific prompt builder.

    This method adapts the FreeCiv prompt builder to the interface expected
    by the rethinking sampler and other Game Arena components.

    Args:
      prompt_template: Template for prompt generation (may be ignored)
      game_short_name: Short name of the game (should be "freeciv")
      **prompt_substitutions: Additional substitutions for prompt generation
        Expected keys:
        - observation: Game observation data
        - legal_actions: List of legal FreeCivAction objects
        - model_name: Name of the model being used
        - strategy_context: Optional strategy configuration
        - memory_context: Optional memory context string
        - rethink_context: Optional rethinking context

    Returns:
      ModelTextInput containing the generated prompt

    Raises:
      ValueError: If required substitutions are missing
    """
    # Validate that this is for FreeCiv
    if game_short_name != "freeciv":
      raise ValueError(f"Unsupported game: {game_short_name}")

    # Extract required parameters
    observation = prompt_substitutions.get("observation")
    legal_actions = prompt_substitutions.get("legal_actions", [])
    model_name = prompt_substitutions.get("model_name", "unknown")

    if observation is None:
      raise ValueError("observation is required for FreeCiv prompt generation")

    # Extract optional parameters
    strategy_context = prompt_substitutions.get("strategy_context")
    memory_context = prompt_substitutions.get("memory_context", "")
    rethink_context = prompt_substitutions.get("rethink_context")

    # Build the prompt using FreeCiv prompt builder
    if rethink_context:
      # This is a rethinking prompt
      prompt_text = self._build_rethinking_prompt(
          observation=observation,
          legal_actions=legal_actions,
          model_name=model_name,
          strategy_context=strategy_context,
          memory_context=memory_context,
          rethink_context=rethink_context,
      )
    else:
      # This is a regular prompt
      prompt_text = self._build_regular_prompt(
          observation=observation,
          legal_actions=legal_actions,
          model_name=model_name,
          strategy_context=strategy_context,
          memory_context=memory_context,
      )

    logging.debug(
        "Generated FreeCiv prompt: %d characters, %d legal actions",
        len(prompt_text),
        len(legal_actions),
    )

    return tournament_util.ModelTextInput(text=prompt_text)

  def _build_regular_prompt(
      self,
      observation: Mapping[str, Any],
      legal_actions: List[FreeCivAction],
      model_name: str,
      strategy_context: Optional[Dict[str, Any]] = None,
      memory_context: str = "",
  ) -> str:
    """Build regular (non-rethinking) prompt.

    Args:
      observation: Game observation data
      legal_actions: List of legal FreeCivAction objects
      model_name: Name of the model
      strategy_context: Optional strategy configuration
      memory_context: Optional memory context string

    Returns:
      Generated prompt text
    """
    # Use the enhanced prompt builder
    prompt_text = self.prompt_builder.build_enhanced_prompt(
        observation=observation,
        legal_actions=legal_actions,
        model_name=model_name,
        strategy_context=strategy_context,
        memory_context=memory_context,
    )

    return prompt_text

  def _build_rethinking_prompt(
      self,
      observation: Mapping[str, Any],
      legal_actions: List[FreeCivAction],
      model_name: str,
      strategy_context: Optional[Dict[str, Any]],
      memory_context: str,
      rethink_context: Dict[str, Any],
  ) -> str:
    """Build rethinking prompt with context about previous attempt.

    Args:
      observation: Game observation data
      legal_actions: List of legal FreeCivAction objects
      model_name: Name of the model
      strategy_context: Optional strategy configuration
      memory_context: Optional memory context string
      rethink_context: Context about previous rethinking attempts

    Returns:
      Generated rethinking prompt text
    """
    # Extract rethinking information
    previous_action = rethink_context.get("previous_action")
    failure_reason = rethink_context.get("failure_reason", "Action was illegal")
    attempt_number = rethink_context.get("attempt_number", 1)

    # Build base prompt
    base_prompt = self._build_regular_prompt(
        observation=observation,
        legal_actions=legal_actions,
        model_name=model_name,
        strategy_context=strategy_context,
        memory_context=memory_context,
    )

    # Add rethinking context
    rethink_section = self._build_rethinking_section(
        previous_action=previous_action,
        failure_reason=failure_reason,
        attempt_number=attempt_number,
        legal_actions=legal_actions,
    )

    # Combine base prompt with rethinking section
    full_prompt = f"{base_prompt}\n\n{rethink_section}"

    return full_prompt

  def _build_rethinking_section(
      self,
      previous_action: Optional[str],
      failure_reason: str,
      attempt_number: int,
      legal_actions: List[FreeCivAction],
  ) -> str:
    """Build the rethinking section of the prompt.

    Args:
      previous_action: Previous action that failed
      failure_reason: Reason why the action failed
      attempt_number: Current attempt number
      legal_actions: List of legal actions

    Returns:
      Rethinking section text
    """
    rethink_lines = ["## RETHINKING REQUIRED", f"Attempt #{attempt_number}", ""]

    if previous_action:
      rethink_lines.extend([
          f"Your previous action '{previous_action}' was rejected.",
          f"Reason: {failure_reason}",
          "",
      ])

    rethink_lines.extend([
        "Please carefully reconsider your choice and select a LEGAL action.",
        (
            "Make sure your action exactly matches one of the legal actions"
            " listed above."
        ),
        "",
        (
            "Your response should contain only the action in the exact format"
            " shown."
        ),
        "Do not add any explanation or commentary.",
    ])

    return "\n".join(rethink_lines)

  def supports_rethinking(self) -> bool:
    """Check if this adapter supports rethinking prompts.

    Returns:
      True, as this adapter supports rethinking
    """
    return True

  def get_supported_games(self) -> List[str]:
    """Get list of supported game names.

    Returns:
      List containing "freeciv"
    """
    return ["freeciv"]

  def create_legal_actions_string(
      self, legal_actions: List[FreeCivAction]
  ) -> str:
    """Create formatted string of legal actions for prompt inclusion.

    Args:
      legal_actions: List of legal FreeCivAction objects

    Returns:
      Formatted string listing legal actions
    """
    if not legal_actions:
      return "No legal actions available."

    # Convert actions to canonical string format
    from game_arena.harness.freeciv_action_converter import FreeCivActionConverter

    converter = FreeCivActionConverter()

    action_strings = []
    for i, action in enumerate(legal_actions):
      action_str = converter.action_to_string(action)
      action_strings.append(f"{i+1}. {action_str}")

    return "\n".join(action_strings)

  def estimate_prompt_length(
      self,
      observation: Mapping[str, Any],
      legal_actions: List[FreeCivAction],
      model_name: str,
      **kwargs: Any,
  ) -> int:
    """Estimate the length of a prompt in characters.

    Args:
      observation: Game observation data
      legal_actions: List of legal actions
      model_name: Model name
      **kwargs: Additional prompt parameters

    Returns:
      Estimated prompt length in characters
    """
    # Quick estimation without building full prompt
    base_length = 2000  # Base prompt template length
    observation_length = len(str(observation))
    legal_actions_length = len(legal_actions) * 50  # ~50 chars per action
    memory_length = len(kwargs.get("memory_context", ""))

    estimated_length = (
        base_length + observation_length + legal_actions_length + memory_length
    )

    return estimated_length

  def validate_prompt_parameters(
      self, game_short_name: str, **prompt_substitutions: Any
  ) -> bool:
    """Validate that prompt parameters are correct.

    Args:
      game_short_name: Game name
      **prompt_substitutions: Prompt parameters

    Returns:
      True if parameters are valid

    Raises:
      ValueError: If parameters are invalid
    """
    if game_short_name != "freeciv":
      raise ValueError(f"Unsupported game: {game_short_name}")

    required_params = ["observation"]
    for param in required_params:
      if param not in prompt_substitutions:
        raise ValueError(f"Missing required parameter: {param}")

    observation = prompt_substitutions["observation"]
    if not isinstance(observation, (dict, Mapping)):
      raise ValueError("observation must be a dictionary or mapping")

    legal_actions = prompt_substitutions.get("legal_actions", [])
    if not isinstance(legal_actions, list):
      raise ValueError("legal_actions must be a list")

    return True


class FreeCivRethinkPromptTemplate:
  """Prompt template specifically for FreeCiv rethinking scenarios."""

  def __init__(self):
    """Initialize FreeCiv rethinking prompt template."""
    self.name = "freeciv_rethink"
    self.template = ""  # Template will be built dynamically
    self.substitutions = {}

  def format(self, **kwargs: Any) -> str:
    """Format the rethinking prompt template.

    Args:
      **kwargs: Template substitutions

    Returns:
      Formatted prompt text
    """
    # This template is built dynamically by the adapter
    adapter = FreeCivPromptGeneratorAdapter()
    return adapter.generate_prompt_with_text_only(
        prompt_template=self, game_short_name="freeciv", **kwargs
    ).text
