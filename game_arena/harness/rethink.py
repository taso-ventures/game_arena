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

"""Rethinking sampler that prompts again if an illegal action is generated."""

from typing import Any, Callable

import pyspiel
from absl import logging

from game_arena.harness import (
    model_generation,
    parsers,
    prompt_generation,
    prompts,
    rethink_fn,
    samplers,
    tournament_util,
)

_RETHINK_WITH_ENV_UNPARSABLE_TEMPLATE = """Your previously suggested move was not parsable.
Please think carefully and generate a new and legal move. Your previous response was:
{generation}
"""

_RETHINK_WITH_ENV_ILLEGAL_TEMPLATE = """Your previously suggested move was: {last_move}, which is an illegal move.
Please think carefully and generate a new and legal move.
"""

_RETHINK_WITH_ENV_ILLEGAL_HISTORY_TEMPLATE = """You previously suggested the moves {illegal_history} which are all illegal.
Please think carefully and generate a new and legal move.
"""

_RETHINK_WITH_ENV_RULE_TEMPLATE = """Your previously suggested move was: {last_move}, which is an illegal move.
Potential reason: {reason}
Please think carefully and generate a new and legal move.
"""


def _sample_parse_return(
    model_input: Any,
    generate_fn: Callable[[Any], tournament_util.GenerateReturn],
    parser: parsers.TextParser,
) -> tuple[str | None, tournament_util.GenerateReturn]:
  generate_return = generate_fn(model_input)
  parsed_action = parser.parse(
      parsers.TextParserInput(text=generate_return.main_response)
  )
  return parsed_action, generate_return


class RethinkSampler(samplers.Sampler):
  """Samples action by prompting again if an illegal action is generated."""

  def __init__(
      self,
      model: model_generation.Model,
      # TODO(google-deepmind): put these into a config class:
      strategy: tournament_util.SamplerChoice,
      num_max_rethinks: int,
      move_parser: parsers.TextParser,
      legality_parser: parsers.TextParser,
      game_short_name: str,
      prompt_generator: prompt_generation.PromptGeneratorSupportsText,
      rethink_template: str | None,
  ):
    if strategy not in (
        tournament_util.SamplerChoice.RETHINK,
        tournament_util.SamplerChoice.RETHINK_WITH_ENV,
        tournament_util.SamplerChoice.RETHINK_WITH_ENV_ILLEGAL_HISTORY,
        tournament_util.SamplerChoice.RETHINK_WITH_ENV_RULE,
    ):
      raise ValueError(f"Unsupported strategy: {strategy}")
    super().__init__(model)
    self._strategy = strategy
    self._num_max_rethinks = num_max_rethinks
    self._game_short_name = game_short_name
    self._move_parser = move_parser
    self._legality_parser = legality_parser
    self._prompt_generator = prompt_generator
    self._rethink_template = rethink_template

  def sample_action_with_text_and_state_input(
      self,
      state: pyspiel.State,
      prompt_template: prompts.PromptTemplate | None = None,
      **prompt_substitutions,
  ) -> samplers.SamplerOutput:
    return self._sample_action(
        self._model.generate_with_text_input,
        state,
        prompt_template,
        **prompt_substitutions,
    )

  def sample_action_with_image_text_and_state_input(
      self,
      state: pyspiel.State,
      prompt_template: prompts.PromptTemplate | None = None,
      **prompt_substitutions,
  ) -> samplers.SamplerOutput:
    # TODO(google-deepmind): flatten MultimodalModel to just Model so we don't have
    # to do this separate dispatch and isinstance check.
    if isinstance(self._model, model_generation.MultimodalModel):
      return self._sample_action(
          self._model.generate_with_image_text_input,
          state,
          prompt_template,
          **prompt_substitutions,
      )
    else:
      raise ValueError("Multimodal model expected but not provided.")

  def _sample_action(
      self,
      model_generate_fn: Callable[[Any], tournament_util.GenerateReturn],
      state: pyspiel.State,
      prompt_template: prompts.PromptTemplate | None = None,
      **prompt_substitutions,
  ) -> samplers.SamplerOutput:
    if prompt_template not in (
        prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
        prompts.PromptTemplate.NO_LEGAL_ACTIONS_WITH_PIECE_DICT_RETHINK_APPENDED,
        prompts.PromptTemplate.NO_LEGAL_ACTIONS_WITH_ASCII_BOARD_RETHINK_APPENDED,
    ):
      raise ValueError(
          f"Unsupported prompt template for rethinking: {prompt_template}"
      )

    # The zeroth prompt does not have any rethinking text substituted into it:
    prompt_substitutions["rethink_prompt"] = ""

    if prompts.is_image_text(prompt_template):
      if isinstance(
          self._prompt_generator,
          prompt_generation.PromptGeneratorSupportsImageText,
      ):
        prompt_fn = self._prompt_generator.generate_prompt_with_image_text
        prompt_args = {
            "prompt_template": prompt_template,
            "game_short_name": self._game_short_name,
            "state": state,
            **prompt_substitutions,
        }
      else:
        raise ValueError(
            "A prompt that expects image text input was provided, but the"
            " prompt generator does not support it."
        )
    else:
      if isinstance(
          self._prompt_generator,
          prompt_generation.PromptGeneratorSupportsText,
      ):
        prompt_fn = self._prompt_generator.generate_prompt_with_text_only
        prompt_args = {
            "prompt_template": prompt_template,
            "game_short_name": self._game_short_name,
            **prompt_substitutions,
        }
      else:
        raise ValueError(
            "A prompt that expects text input was provided, but the"
            " prompt generator does not support it."
        )

    parsed_action = None
    parsed_action_history = []
    maybe_legal_action = None
    num_attempts = 0
    generate_returns = []
    auxiliary_outputs = {}

    # Zeroth attempt is not counted against the max number of rethinks:
    while (
        parsed_action is None or maybe_legal_action is None
    ) and num_attempts < self._num_max_rethinks + 1:
      if num_attempts > 0:
        logging.info(
            "Rethinking attempt %d with rethink prompt substitution: %s",
            num_attempts,
            prompt_args["rethink_prompt"],
        )
      complete_prompt = prompt_fn(**prompt_args)
      parsed_action, generate_return = _sample_parse_return(
          complete_prompt,
          model_generate_fn,
          self._move_parser,
      )
      generate_returns.append(generate_return)
      auxiliary_outputs[f"parsed_action_attempt_{num_attempts}"] = parsed_action
      auxiliary_outputs[f"rethink_prompt_attempt_{num_attempts}"] = prompt_args[
          "rethink_prompt"
      ]
      if parsed_action is not None:
        parsed_action_history.append(parsed_action)
        maybe_legal_action = self._legality_parser.parse(
            parsers.TextParserInput(
                text=parsed_action,
                state_str=state.to_string(),
                legal_moves=parsers.get_legal_action_strings(state),
                player_number=state.current_player(),
            )
        )
        auxiliary_outputs[f"maybe_legal_action_attempt_{num_attempts}"] = (
            maybe_legal_action
        )
        if maybe_legal_action is not None:
          # We have a legal action and can return now:
          break
      else:
        auxiliary_outputs[f"maybe_legal_action_attempt_{num_attempts}"] = None

      # Prepare for the next attempt by generating the rethinking part of the
      # prompt:
      num_attempts += 1
      match self._strategy:
        case tournament_util.SamplerChoice.RETHINK:
          if self._rethink_template is None:
            rethink_prompt = ""
          else:
            raise ValueError("Rethink template should be initialized as None.")
        case tournament_util.SamplerChoice.RETHINK_WITH_ENV:
          if parsed_action is None:
            rethink_prompt = _RETHINK_WITH_ENV_UNPARSABLE_TEMPLATE.format(
                generation=generate_returns[-1].main_response
            )
          else:
            rethink_prompt = _RETHINK_WITH_ENV_ILLEGAL_TEMPLATE.format(
                last_move=parsed_action
            )
        case tournament_util.SamplerChoice.RETHINK_WITH_ENV_ILLEGAL_HISTORY:
          if parsed_action is None:
            rethink_prompt = _RETHINK_WITH_ENV_UNPARSABLE_TEMPLATE.format(
                generation=generate_returns[-1].main_response
            )
          else:
            if num_attempts > 1:
              rethink_prompt = (
                  _RETHINK_WITH_ENV_ILLEGAL_HISTORY_TEMPLATE.format(
                      illegal_history=", ".join(parsed_action_history)
                  )
              )
            else:
              rethink_prompt = _RETHINK_WITH_ENV_ILLEGAL_TEMPLATE.format(
                  last_move=parsed_action
              )
        case tournament_util.SamplerChoice.RETHINK_WITH_ENV_RULE:
          # TODO(google-deepmind): Add support for other games.
          if self._game_short_name != "chess":
            raise ValueError(
                "Only chess is supported for rule-based rethinking. Got game"
                f" name: {self._game_short_name}"
            )
          if parsed_action is None:
            rethink_prompt = _RETHINK_WITH_ENV_UNPARSABLE_TEMPLATE.format(
                generation=generate_returns[-1].main_response
            )
          else:
            # TODO(google-deepmind): the rule-parsed string may have extraneous
            # characters that do not fit the regex used in formulating the
            # illegality explanation (derived from python-chess). So it may
            # benefit to clean up the string before trying to get an
            # explanation.
            reason = rethink_fn.rule_explain_illegal_move(
                fen=state.to_string(), move_str=parsed_action
            )
            rethink_prompt = _RETHINK_WITH_ENV_RULE_TEMPLATE.format(
                last_move=parsed_action, reason=reason
            )
        case _:
          raise ValueError(f"Unsupported strategy: {self._strategy}")

      prompt_args["rethink_prompt"] = rethink_prompt

    return samplers.SamplerOutput(
        # The `action` returned here may be illegal!
        action=(
            maybe_legal_action
            if maybe_legal_action is not None
            else parsed_action
        ),
        extracted_action=parsed_action,
        matched_action=maybe_legal_action,
        generate_returns=generate_returns,
        auxiliary_outputs=auxiliary_outputs,
        move_type=(
            tournament_util.MoveType.LEGAL
            if maybe_legal_action is not None
            else tournament_util.MoveType.ILLEGAL
        ),
    )
