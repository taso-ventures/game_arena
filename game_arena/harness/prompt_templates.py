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

"""Prompt template definitions for the tournament."""

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
  FREECIV_ENHANCED = "FREECIV_ENHANCED"

  @classmethod
  def from_string(cls, value: str) -> "PromptTemplate":
    return PromptTemplate[value]


def is_image_text(prompt_template: PromptTemplate) -> bool:
  """Returns whether the prompt template is image and text."""
  return prompt_template in (
      PromptTemplate.WITH_BOARD_IMAGE,
      PromptTemplate.WITH_SVG_RENDERED_IMAGE,
  )