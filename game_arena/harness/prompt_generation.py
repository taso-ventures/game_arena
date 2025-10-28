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

"""Library for generating prompts using templates."""

from typing import Generic, Protocol, runtime_checkable

import pyspiel

from game_arena.harness import tournament_util
from game_arena.harness.prompt_templates import PromptTemplate


@runtime_checkable
class PromptGeneratorSupportsImageText(
    Generic[tournament_util.ModelImageTextInputT], Protocol
):
    """Generator of prompts containing text and e.g. a board image."""

    # Keep dependency on PySpiel state as we may use prompt generators with
    # renderers that work with PySpiel or call PySpiel for rendering:
    def generate_prompt_with_image_text(
        self,
        prompt_template: PromptTemplate,
        game_short_name: str,
        state: pyspiel.State | None = None,
        **prompt_substitutions,
    ) -> tournament_util.ModelImageTextInputT: ...


@runtime_checkable
class PromptGeneratorSupportsText(Generic[tournament_util.ModelTextInputT], Protocol):
    """Generator of prompts containing text only."""

    def generate_prompt_with_text_only(
        self,
        prompt_template: PromptTemplate,
        game_short_name: str,
        **prompt_substitutions,
    ) -> tournament_util.ModelTextInputT: ...


class PromptGeneratorText(PromptGeneratorSupportsText):
    """Generator of prompts containing text only."""

    def generate_prompt_with_text_only(
        self,
        prompt_template: PromptTemplate,
        game_short_name: str,
        **prompt_substitutions,
    ) -> tournament_util.ModelTextInput:
        prompt_substitutions["game_short_name"] = game_short_name
        match prompt_template:
            case prompts.PromptTemplate.NO_LEGAL_ACTIONS:
                actual_template = prompts.PROMPT_TEMPLATE_NO_LEGAL_ACTIONS
            case prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED:
                actual_template = (
                    prompts.PROMPT_TEMPLATE_NO_LEGAL_ACTIONS_RETHINK_APPENDED
                )
            case prompts.PromptTemplate.NO_LEGAL_ACTIONS_WITH_PIECE_DICT:
                actual_template = (
                    prompts.PROMPT_TEMPLATE_NO_LEGAL_ACTIONS_WITH_PIECE_DICT
                )
            case (
                prompts.PromptTemplate.NO_LEGAL_ACTIONS_WITH_PIECE_DICT_RETHINK_APPENDED
            ):
                actual_template = (
                    prompts.PROMPT_TEMPLATE_NO_LEGAL_ACTIONS_WITH_PIECE_DICT_RETHINK_APPENDED
                )
            case prompts.PromptTemplate.NO_LEGAL_ACTIONS_WITH_ASCII_BOARD:
                actual_template = (
                    prompts.PROMPT_TEMPLATE_NO_LEGAL_ACTIONS_WITH_ASCII_BOARD
                )
            case (
                prompts.PromptTemplate.NO_LEGAL_ACTIONS_WITH_ASCII_BOARD_RETHINK_APPENDED
            ):
                actual_template = (
                    prompts.PROMPT_TEMPLATE_NO_LEGAL_ACTIONS_WITH_ASCII_BOARD_RETHINK_APPENDED
                )
            case prompts.PromptTemplate.WITH_LEGAL_ACTIONS:
                actual_template = prompts.PROMPT_TEMPLATE_WITH_LEGAL_ACTIONS
            case prompts.PromptTemplate.FREECIV_ENHANCED:
                # Import FreeCivPromptBuilder for enhanced FreeCiv prompts
                from game_arena.harness.prompts.freeciv_prompts import \
                    FreeCivPromptBuilder

                # Extract required data from prompt_substitutions
                observation = prompt_substitutions.get("observation")
                legal_actions = prompt_substitutions.get("legal_actions", [])
                model_name = prompt_substitutions.get("model_name", "gpt-5")

                if observation is None:
                    raise ValueError(
                        "FREECIV_ENHANCED requires 'observation' in prompt_substitutions"
                    )

                # Use FreeCivPromptBuilder to generate enhanced prompt
                builder = FreeCivPromptBuilder()
                prompt_text = builder.build_enhanced_prompt(
                    observation, legal_actions, model_name
                )

                return tournament_util.ModelTextInput(prompt_text=prompt_text)
            case _:
                raise ValueError(f"Unsupported prompt template: {prompt_template}")
        return tournament_util.ModelTextInput(
            prompt_text=actual_template.format(**prompt_substitutions)
        )


# TODO(google-deepmind): implement multimodal prompt generator.
