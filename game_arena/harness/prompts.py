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

"""Prompts for the tournament."""

import enum


@enum.unique
class PromptTemplate(enum.Enum):
    """Prompt template."""

    WITH_LEGAL_ACTIONS = "WITH_LEGAL_ACTIONS"
    NO_LEGAL_ACTIONS = "NO_LEGAL_ACTIONS"
    NO_LEGAL_ACTIONS_NO_HISTORY = "NO_LEGAL_ACTIONS_NO_HISTORY"
    NO_LEGAL_ACTIONS_RETHINK_APPENDED = "NO_LEGAL_ACTIONS_RETHINK_APPENDED"
    NO_LEGAL_ACTIONS_WITH_PIECE_DICT = "NO_LEGAL_ACTIONS_WITH_PIECE_DICT"
    NO_LEGAL_ACTIONS_WITH_PIECE_DICT_RETHINK_APPENDED = (
        "NO_LEGAL_ACTIONS_WITH_PIECE_DICT_RETHINK_APPENDED"
    )
    NO_LEGAL_ACTIONS_WITH_ASCII_BOARD = "NO_LEGAL_ACTIONS_WITH_ASCII_BOARD"
    NO_LEGAL_ACTIONS_WITH_ASCII_BOARD_RETHINK_APPENDED = (
        "NO_LEGAL_ACTIONS_WITH_ASCII_BOARD_RETHINK_APPENDED"
    )
    WITH_BOARD_IMAGE = "WITH_BOARD_IMAGE"
    WITH_SVG_RENDERED_IMAGE = "WITH_SVG_RENDERED_IMAGE"

    @classmethod
    def from_string(cls, value: str) -> "PromptTemplate":
        return PromptTemplate[value]


def is_image_text(prompt_template: PromptTemplate) -> bool:
    """Returns whether the prompt template is image and text."""
    return prompt_template in (
        PromptTemplate.WITH_BOARD_IMAGE,
        PromptTemplate.WITH_SVG_RENDERED_IMAGE,
    )


PROMPT_TEMPLATE_NO_LEGAL_ACTIONS = """Let's play {game_short_name}. The current game state in {notation} is:
{readable_state_str}
The moves played so far are:
{move_history}
You are playing as player {player_name}.
It is now your turn. Play your strongest move. The move MUST be legal. Reason step by step to come up with your move, then output your final answer in the format "Final Answer: X" where X is your chosen move in {move_notation}."""

PROMPT_TEMPLATE_NO_LEGAL_ACTIONS_RETHINK_APPENDED = (
    PROMPT_TEMPLATE_NO_LEGAL_ACTIONS
    + """
{rethink_prompt}"""
)

PROMPT_TEMPLATE_NO_LEGAL_ACTIONS_WITH_PIECE_DICT = """Let's play {game_short_name}. The current game state in {notation} is:
{readable_state_str}
The current piece positions are:
{piece_dict}
The moves played so far are:
{move_history}
You are playing as player {player_name}.
It is now your turn. Play your strongest move. The move MUST be legal. Reason step by step to come up with your move, then output your final answer in the format "Final Answer: X" where X is your chosen move in {move_notation}."""

PROMPT_TEMPLATE_NO_LEGAL_ACTIONS_WITH_PIECE_DICT_RETHINK_APPENDED = (
    PROMPT_TEMPLATE_NO_LEGAL_ACTIONS_WITH_PIECE_DICT
    + """
{rethink_prompt}"""
)

PROMPT_TEMPLATE_NO_LEGAL_ACTIONS_WITH_ASCII_BOARD = """Let's play {game_short_name}. The current game state in {notation} is:
{readable_state_str}
The current board is visualized below:
{ascii_board}
The moves played so far are:
{move_history}
You are playing as player {player_name}.
It is now your turn. Play your strongest move. The move MUST be legal. Reason step by step to come up with your move, then output your final answer in the format "Final Answer: X" where X is your chosen move in {move_notation}."""

PROMPT_TEMPLATE_NO_LEGAL_ACTIONS_WITH_ASCII_BOARD_RETHINK_APPENDED = (
    PROMPT_TEMPLATE_NO_LEGAL_ACTIONS_WITH_ASCII_BOARD
    + """
{rethink_prompt}"""
)

PROMPT_TEMPLATE_NO_LEGAL_ACTIONS_NO_HISTORY = """Let's play {game_short_name}. The current game state in {notation} is:
{readable_state_str}
You are playing as player {player_name}.
It is now your turn. Play your strongest move. The move MUST be legal. Reason step by step to come up with your move, then output your final answer in the format "Final Answer: X" where X is your chosen move in {move_notation}."""


PROMPT_TEMPLATE_WITH_LEGAL_ACTIONS = """Let's play {game_short_name}. The current game state in {notation} is:
{readable_state_str}
The moves played so far are:
{move_history}
The list of legal moves is:
{legal_actions}
You are playing as player {player_name}.
It is now your turn. Choose the strongest move from the list of legal moves. Reason step by step to come up with your move, then output your final answer in the format "Final Answer: X" where X is your chosen move."""


PROMPT_TEMPLATE_NO_LEGAL_ACTIONS_PREFIX = """Let's play {game_short_name}. The current game board is shown as:
"""

PROMPT_TEMPLATE_NO_LEGAL_ACTIONS_SUFFIX = """
You are playing as player {player_name}.
It is now your turn. Play your strongest move. The move MUST be legal. Reason step by step to come up with your move, then output your final answer in the format "Final Answer: X" where X is your chosen move in {move_notation}."""
