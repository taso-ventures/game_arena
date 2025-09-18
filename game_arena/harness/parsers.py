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

"""Move parsers."""

import dataclasses
import random
import re
from typing import Protocol, Sequence

import chess
import pyspiel
from absl import logging

from game_arena.harness import game_notation_examples


def get_legal_action_strings(state: pyspiel.State) -> Sequence[str]:
    action_ints = state.legal_actions()
    return [
        state.action_to_string(state.current_player(), action_int)
        for action_int in action_ints
    ]


def parse_move_from_response(
    response: str, action_tag: str = "Final Answer: "
) -> dict[str, str] | None:
    """Extracts move, and the text before the move from response."""
    if response is None:
        return None
    try:
        last_index = -1
        final_split_token = ""
        for split_token in [action_tag, ":", "is"]:
            tmp_index = response.rfind(split_token)
            if tmp_index > last_index:
                last_index = tmp_index
                final_split_token = split_token
        if last_index == -1:
            return None
        # Split the string
        suffix = response[last_index + len(final_split_token) :]
        if suffix is None:
            return None
        move_str = (
            suffix.strip(" .")
            .replace("$", "")
            .replace("\\boxed{", "")
            .replace("\\text{", "")
            .replace("\boxed{", "")
            .replace("\text{", "")
            .replace("}", "")
            .replace("*", "")
            .replace(" ", "")
            .replace("`", "")
            .replace("\n", "")
        )
        clean = re.compile("<.*?>")
        move_str = re.sub(clean, "", move_str)
        if not move_str:
            return None
        prefix = response[:last_index]
        return {
            "move": move_str,
            "prefix": "" if prefix is None else prefix.strip(),
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.error(
            "parse_move_from_response failed with input response %s", response
        )
        raise e


# TODO(google-deepmind): divide this into TextParserInput and
# TextParserWithStateInput:
@dataclasses.dataclass(frozen=True, kw_only=True)
class TextParserInput:
    text: str
    state_str: str | None = None
    legal_moves: Sequence[str] | None = None
    player_number: int | None = None


class TextParser(Protocol):
    """Generic text parser."""

    # TODO(google-deepmind): should we define a parser output type?
    def parse(self, parser_input: TextParserInput) -> str | None: ...


class ChainedMoveParser(TextParser):
    """Parses text into a move by successively applying parsers.

    For example, parsing with a LLM followed by parsing of the structured output
    with a rule-based parser or with a soft-parser against legal moves.
    """

    def __init__(self, parsers: Sequence[TextParser]):
        self._parsers = parsers

    def parse(self, parser_input: TextParserInput) -> str | None:
        maybe_text_output = None
        for parser in self._parsers:
            maybe_text_output = parser.parse(parser_input)
            if maybe_text_output is None:
                return None
            parser_input = dataclasses.replace(parser_input, text=maybe_text_output)
        return maybe_text_output


class RuleBasedMoveParser(TextParser):
    """Parses text into a move with string processing built into Python."""

    def __init__(self, action_tag: str = "Final Answer: "):
        self._action_tag = action_tag

    def parse(self, parser_input: TextParserInput) -> str | None:
        maybe_move_and_prefix = parse_move_from_response(
            parser_input.text, action_tag=self._action_tag
        )
        if maybe_move_and_prefix is None:
            return None
        return maybe_move_and_prefix["move"]


def _chess_soft_parser_v1(
    state_str: str, selected_action: str, spiel_legal_moves: Sequence[str]
) -> str | None:
    """Chess parser that matches against legal moves."""
    if selected_action is None:
        return None

    selected_action = selected_action.strip()

    if not selected_action:
        return None

    if selected_action[0].isdigit():
        # \d+ is the first capturing group, matching one or more digits.
        # \.|\.\.\. is the second capturing group, matching one or three dots.
        # e.g. 1. for first white move, 2... for second black move.
        # .* captures remaining characters.
        match = re.search(r"(\d+)(\.|\.\.\.)(.*)", selected_action)
        if match is not None:
            _, _, selected_action = match.groups()
        else:
            return None
    # There might be whitespace between the dot(s) and the move:
    selected_action = selected_action.lstrip()
    # python-chess uses a regex that expects the move to end with the destination
    # square, followed optionally by pawn promotion and/or + or # for checkmate.
    # The following characters should definitely be removed if they are at the end
    # and they also do not appear or are not differentiating according to chess
    # notation:
    for char_to_remove in [
        ":",  # Indicates capture but not expected by python-chess.
        ".",  # Only used in the move number.
        "*",
        ",",
        "&",
        "^",
        "\\",
        "<",
        ">",
        "{",
        "}",
        "[",
        "]",
        "?",  # Move quality comment.
        "!",  # Move quality comment.
    ]:
        selected_action = selected_action.replace(char_to_remove, "")

    # En passant annotation (which is e.p. or e.p), but we removed the dots.
    selected_action = selected_action.removesuffix("ep")
    # N.B. python-chess also considers castling indicated with zeros instead of
    # capital letter Os. Zeroes are non-standard notation.

    board = chess.Board(state_str)
    maybe_legal_move_san = selected_action
    # Match between python-chess and OpenSpiel with UCI standard, which uses
    # ambiguity-free pure algebraic coordinate notation. It is slightly different
    # from LAN: https://www.chessprogramming.org/Algebraic_Chess_Notation#UCI
    selected_uci = None
    try:
        selected_uci = board.parse_san(maybe_legal_move_san).uci()
    except ValueError as e:
        error_str = str(e)
        # TODO(google-deepmind): handle other ambiguous cases.
        if "ambiguous" in error_str:
            possible_moves_uci = []
            for legal_move in board.legal_moves:
                legal_move_san = board.san(legal_move)
                legal_move_san_short = legal_move_san.rstrip("+#")
                maybe_legal_move_san_short = maybe_legal_move_san.rstrip("+#")
                # Examples: Rad1 and Rhd1, R2d5 and R8d5, Ngf3 and Nef3
                if legal_move_san_short.startswith(
                    maybe_legal_move_san_short[0]
                ) and legal_move_san_short.endswith(maybe_legal_move_san_short[-2:]):
                    possible_moves_uci.append(legal_move.uci())
            if not possible_moves_uci:
                return None
            rng = random.Random(42)
            selected_uci = rng.choice(possible_moves_uci)
    if selected_uci is None:
        return None
    else:
        legal_move = board.parse_uci(selected_uci)
        # Translate the move to a SAN string.
        selected_action = board.san(legal_move)
    # Match exactly with OpenSpiel legal (SAN) moves:
    if selected_action not in spiel_legal_moves:
        return None
    return selected_action


class SoftMoveParser(TextParser):
    """Parses text into a move by soft matching against legal moves."""

    def __init__(self, game_short_name: str):
        self._game_short_name = game_short_name

    def parse(self, parser_input: TextParserInput) -> str | None:
        if (
            parser_input.state_str is None
            or parser_input.legal_moves is None
            or parser_input.player_number is None
        ):
            raise ValueError(
                "State as a string, legal moves, and player number are required for"
                " soft move parsing."
            )

        selected_action = parser_input.text

        if not selected_action:
            return None

        if self._game_short_name.startswith("chess"):
            try:
                return _chess_soft_parser_v1(
                    parser_input.state_str, selected_action, parser_input.legal_moves
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error(
                    "Soft parser failed for chess with inputs state %s selected_action"
                    " %s legal_moves %s"
                )
                raise e
        elif (
            self._game_short_name.startswith("tic_tac_toe")
            or self._game_short_name.startswith("connect_four")
            or self._game_short_name.startswith("go")
        ):
            player_name = game_notation_examples.GAME_SPECIFIC_NOTATIONS[
                self._game_short_name
            ]["player_map"][parser_input.player_number]
            if self._game_short_name.startswith("go"):
                # OpenSpiel Go action string is lowercased.
                selected_action = (
                    player_name
                    + " "
                    + selected_action.strip("(),").replace(",", "").lower()
                )
            elif self._game_short_name.startswith("tic_tac_toe"):
                selected_action = (
                    player_name
                    + ("" if selected_action.startswith("(") else "(")
                    + selected_action.replace(" ", "")
                    + ("" if selected_action.endswith(")") else ")")
                )
            else:
                selected_action = player_name + selected_action
        elif self._game_short_name.startswith("universal_poker"):
            selected_action = "player={} move={}".format(
                parser_input.player_number, selected_action
            )
        if selected_action not in parser_input.legal_moves:
            return None
        return selected_action
