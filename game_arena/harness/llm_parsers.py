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

"""LLM-based parsers."""

import dataclasses

from absl import logging
from game_arena.harness import model_generation
from game_arena.harness import parsers
from game_arena.harness import tournament_util

import re


@dataclasses.dataclass(frozen=True, kw_only=True)
class InstructionConfig:
  name: str
  instruction: str
  final_answer_prefix: str
  no_action_answer: str


# Game-agnostic version. Excess formatting examples may be too specific or may
# result in unintended removal of characters for certain game moves. Requires
# that the input has the answer prefixed with "Final Answer". Forces
# cleaned answer to be alphanumeric.
InstructionConfig_V0 = InstructionConfig(
    name="InstructionConfig_V0",
    instruction="""## Instructions for Extracting Final Answers

**Objective:** Given a response containing context and the final answer, extract the final answer without excess formatting.

**Process:**

1. **Analyze the response:** From the response, identify the context preceding the raw final answer and the raw final answer itself. The raw final answer might be preceded with the two words Final Answer.
   If no raw final answer is present, skip the steps below and then present "Clean Answer: LLMEXTRACT_NO_FINAL_ANSWER" on a new line (no additional markings or explanations are needed).
2. **Extract the raw final answer:** Extract the raw final answer. Ignore any explanations that follow the raw final answer.
3. **Remove excess formatting:** Remove any excess formatting surrounding the extracted raw final answer to produce a clean final answer.
   For example: remove LaTeX formatting, HTML tags, surrounding brackets, excess newlines, leading whitespace, trailing whitespace, and terminating periods.
4. **Present the clean final answer**  Present the clean final answer on a new line, preceded by "Clean Answer: ".

**Note:** No additional markings or explanations are needed beyond "Clean Answer: " and the extracted answer.""",
    final_answer_prefix="Clean Answer: ",
    no_action_answer="LLMEXTRACT_NO_FINAL_ANSWER",
)

ChessInstructionConfig_V0 = InstructionConfig(
    name="ChessInstructionConfig_V0",
    instruction="""## Instructions for Extracting Final Proposed Chess Move

**Objective:** Given a response containing context and the final proposed chess move, extract the final proposed chess move without excess formatting.

**Process:**

1. **Analyze the response:** From the response, identify the context preceding the final proposed chess move and the final proposed chess move itself.
   If no final proposed chess move is present, skip the steps below and then present "Clean Move: LLMEXTRACT_NO_PROPOSED_MOVE" on a new line (no additional markings or explanations are needed).
2. **Extract the final proposed chess move:** Extract the final proposed chess move.
3. **Remove excess formatting:** From the extracted final proposed chess move, identify characters that define the move in chess algebraic notation and characters that are not part of the move.
   Refer to your knowledge of chess algebraic notation.
   Examples:
     - Capture may be indicated with x immediately before the destination square e.g. Bxe5
     - Capture may also be indicated with a multiplication sign (×) or a colon (:) before the destination square e.g. B×f5, B:f5 or after the destination square e.g. Bf5×, Bf5:
     - En passant may be indicated with the suffix "e.p."
     - Pawn promotion may be indicated in different ways. For example, a pawn on e7 promoting to a queen on e8 may be indicated as e8Q, e8=Q, e8(Q), or e8/Q
     - Castling may be indicated with dashes. For example, 0-0 or O-O for king-side castling and 0-0-0 or O-O-O for queen-side castling
     - Check may be indicated with one plus sign (+) or a dagger (†) or the suffix "ch"
     - Checkmate may be indicated with # or double dagger (‡) or two plus signs (++)
     - Draw offer may be indicated with an equals sign in parentheses (=)
   Include the chess algebraic notation characters that define the move.
   Discard characters at the beginning that indicate the move number e.g. remove 9. from 9.Rd1 Nb6.
   Discard excess characters such as LaTeX formatting, excess newlines, excess periods, excess spaces, excess commas, excess quotations, excess ticks, excess asterisks, excess brackets.
4. **Present the clean final proposed chess move**  Present the clean final proposed chess move on a new line, preceded by "Clean Move: ".

**Note:** No additional markings or explanations are needed beyond "Clean Move: " and the clean final proposed chess move.""",
    final_answer_prefix="Clean Move: ",
    no_action_answer="LLMEXTRACT_NO_PROPOSED_MOVE",
)

OpenSpielChessInstructionConfig_V0 = InstructionConfig(
    name="OpenSpielChessInstructionConfig_V0",
    instruction="""## Instructions for Extracting and Standardizing Final Proposed Chess Move

**Process:**

1. **Analyze the response:** From the response, identify the final proposed chess move.
   If no final proposed chess move is present, skip the steps below and then present "Standardized Move: LLMEXTRACT_NO_PROPOSED_MOVE" on a new line (no additional markings or explanations are needed).
2. **Extract the final proposed chess move:** Extract the final proposed chess move. It may be in standard algebraic notation (SAN), long algebraic notation (LAN), or portable game notation (PGN).
3. **Interpret the final proposed chess move:** Using your knowledge of standardized algebraic notation, long algebraic notation, and portable game notation, carefully interpret the final proposed chess move. The move may have the following indications:
     - Capture may be indicated with x immediately before the destination square e.g. Bxe5
     - Capture may also be indicated with a multiplication sign (×) or a colon (:) before the destination square e.g. B×f5, B:f5 or after the destination square e.g. Bf5×, Bf5:
     - En passant may be indicated with the suffix "e.p."
     - Pawn promotion may be indicated in different ways. For example, a pawn on e7 promoting to a queen on e8 may be indicated as e8Q, e8=Q, e8(Q), or e8/Q
     - Castling may be indicated with dashes. For example, 0-0 or O-O for king-side castling and 0-0-0 or O-O-O for queen-side castling
     - Check may be indicated with one plus sign (+) or a dagger (†) or the suffix "ch"
     - Checkmate may be indicated with # or double dagger (‡) or two plus signs (++)
     - Draw offer may be indicated with an equals sign in parentheses (=)
4. **Standardize the final proposed chess move:** Carefully standardize the final proposed chess move according to the formatting rules below:

  There are three types of standardized moves:
  i). O-O (short castle)
  ii). O-O-O (long castle)
  iii). [piece type][from file][from rank][x][to square][=Promo][annotations]

  [piece type] is omitted for pawns
  [from file] is only included if 1) move is a pawn capture, or 2) it's required for disambiguation (see below).
  [from rank] is only included if it's required for disambiguation.
  [x] is only included for captures
  [to square] is always included
  [=Promo] is only included for promotions ("=N", "=B", "=R", "=Q" depending on type promoting to).
  [annotations] are a list of characters with different meanings. The only ones we care about are plus sign (+) for check, and hash symbol (#) for checkmate. All others are optional.

  Disambiguation:
  If a move is not uniquely-identified otherwise, the file and/or rank of the from square should be inserted to disambiguate. When either one will disambiguate, the file should be used. If the file is unique, the file should be used. Otherwise if the rank is unique, the rank should be used. If neither is unique, both should be used.

  Examples:
  * e4 (pawn to e4)
  * exd5 (pawn on file e capture the piece on d5)
  * Nf3 (knight to f3)
  * Nxd5 (knight captures piece on d5)
  * Bed5 (bishop on file e to d5)
  * B5xc3 (bishop on rank 5 capture piece on c3)
  * Ne5f7 (knight on e5 to f7, when there are 3 knights on the board, one on e file, and one on 5th rank)
  * exd8=N#!! (pawn on e file capture piece on d8 and promote to knight resulting in checkmate in a surprisingly good move)
  * O-O-O!!N+/- (a surprisingly good long castle that is a theoretical novelty that gives white a clear but not winning advantage)

  Discard excess characters such as LaTeX formatting, newlines, periods, spaces, commas, quotations, ticks, asterisks, and brackets.
  Discard characters at the beginning that are move number indicators. This can be a number followed by a period or a number followed by three periods.

5. **Present the standardized final proposed chess move**  Present the standardized final proposed chess move on a new line, preceded by "Standardized Move: ".

**Note:** No additional markings or explanations are needed beyond "Standardized Move: " and the standardized final proposed chess move.""",
    final_answer_prefix="Standardized Move: ",
    no_action_answer="LLMEXTRACT_NO_PROPOSED_MOVE",
)


def _parse_extractor_response(
    *, response: str, final_answer_prefix: str
) -> str:
  """Parses the response from the extractor LLM."""
  # Regex captures all text following prefix on the same line:
  final_answer_match = re.search(
      rf"{final_answer_prefix}\s*(.*)",
      response,
  )
  final_answer = final_answer_match.group(1) if final_answer_match else ""
  if final_answer:
    # Removes leading and trailing whitespace!
    final_answer = final_answer.splitlines()[0].strip()
  return final_answer


class LLMParser(parsers.TextParser):
  """Parses move from a LLM response with another (separate) LLM."""

  def __init__(
      self, model: model_generation.Model, instruction_config: InstructionConfig
  ):
    self._model = model
    self._instruction_config = instruction_config

  def parse(self, parser_input: parsers.TextParserInput) -> str | None:
    if not parser_input.text:
      logging.warning("Empty input text for LLMParser.")
      return None
    extractor_response = self._model.generate_with_text_input(
        model_input=tournament_util.ModelTextInput(
            prompt_text=parser_input.text,
            system_instruction=self._instruction_config.instruction,
        )
    )
    logging.info(
        "Extractor input last line: %s", parser_input.text.splitlines()[-1]
    )
    logging.info(
        "Extractor response pre-parse: %s", extractor_response.main_response
    )
    parsed_extractor_response = _parse_extractor_response(
        response=extractor_response.main_response,
        final_answer_prefix=self._instruction_config.final_answer_prefix,
    )
    logging.info("Extractor response post-parse: %s", parsed_extractor_response)
    if parsed_extractor_response == self._instruction_config.no_action_answer:
      return None
    return parsed_extractor_response
