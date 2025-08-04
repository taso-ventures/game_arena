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

"""Library for data classes and functions for running tournaments."""

import dataclasses
import enum
from typing import Any, Mapping, Sequence, TypeVar

import chess
import chess.pgn
import dataclasses_json
from game_arena.harness.formatters import go as go_formatter
import immutabledict

import pyspiel


ModelTextInputT = TypeVar("ModelTextInputT")
ModelImageTextInputT = TypeVar("ModelImageTextInputT")


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModelTextInput:
  prompt_text: str
  system_instruction: str | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class ModelImageTextInput(ModelTextInput):
  prompt_image_bytes: bytes
  prompt_image_mime_type: str


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True, kw_only=True)
class GenerateReturn:
  # It's not possible to always correctly split the main response and thoughts:
  main_response: str
  main_response_and_thoughts: str
  request_for_logging: dict[str, Any] | None = None
  response_for_logging: dict[str, Any] | None = None
  generation_tokens: int | None = None
  prompt_tokens: int | None = None
  reasoning_tokens: int | None = None


class ParserChoice(enum.Enum):
  RULE_THEN_SOFT = "rule_then_soft"
  LLM_ONLY = "llm_only"
  LLM_THEN_SOFT = "llm_then_soft"


class SamplerChoice(enum.Enum):
  STANDARD = "standard"
  MAJORITY_VOTE = "majority_vote"
  RETHINK = "rethink"
  RETHINK_WITH_ENV = "rethink_with_env"
  RETHINK_WITH_ENV_ILLEGAL_HISTORY = "rethink_with_env_illegal_history"
  RETHINK_WITH_ENV_RULE = "rethink_with_env_rule"


class MoveType(enum.Enum):
  LEGAL = "legal"
  ILLEGAL = "illegal"
  NO_ACTION_TAG = "no_action_tag"
  # TODO(google-deepmind): introduce UNPARSABLE type?


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class SingleMove:
  """Represents a single move in a game."""

  action: str
  extracted_action: str | None
  matched_action: str | None
  legal_actions: Sequence[str]
  player: int
  player_model: str
  prefix_to_action: str
  generate_returns: Sequence[GenerateReturn]
  debug_string: str
  raw_prompt: str
  is_illegal: bool = dataclasses.field(default=False)
  is_no_action_tag: bool = dataclasses.field(default=False)
  raw_prompt_image: bytes | None = dataclasses.field(default=None)
  auxiliary_outputs: dict[str, Any] | None = dataclasses.field(default=None)


@dataclasses_json.dataclass_json
@dataclasses.dataclass(frozen=True)
class SingleGame:
  game_number: int
  state_str: str
  model_player_one: str
  model_player_two: str
  prompt_template_player_one: str
  prompt_template_player_two: str
  moves: immutabledict.immutabledict[str, SingleMove] = dataclasses.field(
      default_factory=immutabledict.immutabledict
  )
  n_illegal_moves: int = 0
  n_no_action_tags: int = 0


# TODO(google-deepmind): should we implement a safe formatter that raises if any
# format arg is left unused in the template?

# TODO(google-deepmind): following helper functions require cleanup and tests and
# may be moved into a separate file.


def convert_to_readable_state(
    *, game_short_name: str, state_str: str, current_player: int
) -> str:
  """Converts a state to a readable string."""
  if game_short_name.startswith("universal_poker"):
    lines = state_str.split("\n")
    new_lines = []
    for line in lines:
      if line.startswith("ACPC State:"):
        continue
      elif line.startswith("P0 Cards:") and current_player == 1:
        continue
      elif line.startswith("P1 Cards:") and current_player == 0:
        continue
      else:
        new_lines.append(line)
    return "\n".join(new_lines)
  elif game_short_name.startswith("tic_tac_toe") or game_short_name.startswith(
      "connect_four"
  ):
    return "\n".join(
        [" ".join(list(line.strip())) for line in state_str.split("\n")]
    )
  elif game_short_name.startswith("go"):
    return go_formatter.convert_state(state_str)
  else:
    return state_str


def format_chess_movetext(
    game: chess.pgn.Game,
    numbering_scheme: str,
    use_lan: bool,
    add_current_fen: bool,
) -> str:
  """Formats the movetext for a given chess game.

  See (https://en.wikipedia.org/wiki/Portable_Game_Notation#Movetext).

  Args:
    game: Chess game.
    numbering_scheme: Move number formatting.
    use_lan: Whether to use long algebraic notation (defaults to SAN).
    add_current_fen: Whether to add the current FEN at the end of the movetext.

  Returns:
    The formatted movetext for the given game.
  """
  rv = []
  preceding_comment = False
  preceding_board = game.board()
  nodes = list(game.mainline())
  for i in range(len(nodes) + 1):
    if add_current_fen and i == len(nodes):
      rv.append(f"{{ [%FEN {preceding_board.fen()}] }}")
    fullmove_number = f"{(i + 2) // 2}"
    if i % 2 == 0:
      fullmove_number += "."
    else:
      fullmove_number += "..."
    if numbering_scheme == "all":
      rv.append(fullmove_number)
    elif numbering_scheme == "default":
      if i % 2 == 0 or preceding_comment:
        rv.append(fullmove_number)
    elif numbering_scheme == "none":
      pass
    else:
      raise ValueError(f"Unknown numbering_scheme: {numbering_scheme}")
    if i == len(nodes):
      break
    else:
      n = nodes[i]
    if use_lan:
      rv.append(str(n.move))
    else:
      rv.append(str(preceding_board.san(n.move)))
    if n.comment:
      preceding_comment = True
      rv.append(f"{{ {n.comment} }}")
    else:
      preceding_comment = False
    preceding_board = n.board()
  return " ".join(rv)


def get_pgn(target_state, player_names=None) -> chess.pgn.Game:
  """Creates a PGN game from a target state."""
  if player_names is None:
    player_names = ["Black", "White"]
  game = pyspiel.load_game("chess")
  state = game.new_initial_state()
  moves = []
  for action in target_state.history():
    # Append move in UCI notation.
    moves.append(pyspiel.chess.action_to_move(action, state.board()).to_lan())
    state.apply_action(action)

  # Create a new game.
  pgn_game = chess.pgn.Game()
  pgn_game.headers["Event"] = "Chess Game"
  pgn_game.headers["White"] = player_names[1]
  pgn_game.headers["Black"] = player_names[0]
  # Add results header.
  if target_state.is_terminal():
    score = {
        -1: "0",
        0: "1/2",
        1: "1",
    }
    int_returns = [score[x] for x in target_state.returns()]
    # Note: Results are 'white-black', while returns are 'black, white'.
    result = "-".join(reversed(int_returns))
  else:
    result = "*"
  pgn_game.headers["Result"] = result

  # Add moves to the game.
  pgn_game.add_line([chess.Move.from_uci(move) for move in moves])
  return pgn_game


def get_action_string_history(state: pyspiel.State) -> str:
  """Returns a string history of actions."""
  game_name = state.get_game().get_type().short_name
  if game_name.startswith("chess"):
    game = get_pgn(state)
    move_history = format_chess_movetext(
        game,
        numbering_scheme="default",
        use_lan=False,
        add_current_fen=False,
    )
    return move_history

  tmp_state = state.get_game().new_initial_state()
  move_list = []
  for action in state.history():
    action_str = tmp_state.action_to_string(tmp_state.current_player(), action)
    move_list.append(action_str)
    tmp_state.apply_action(action)

  # Use 'None' for empty history, matching C++ handler's initial state output.
  if game_name.startswith("universal_poker") or game_name.startswith("go"):
    move_history = ", ".join(move_list) if move_list else "None"
  else:
    move_history = " ".join(move_list) if move_list else "None"
  return move_history


def get_piece_positions(
    game_short_name: str, state: pyspiel.State
) -> Mapping[str, str]:
  """Returns a dictionary of piece positions and types for a given state."""
  if game_short_name.startswith("chess"):
    return get_chess_piece_positions(state.board().to_fen())
  else:
    raise ValueError(
        f"get_piece_dict is not implemented for game: {game_short_name}"
    )


def get_chess_piece_positions(
    fen: str, include_empty_squares: bool = False
) -> Mapping[str, str]:
  """Returns a dictionary of piece positions and types for a given FEN string.

  Args:
    fen: The FEN (Forsyth-Edwards Notation) string representing the board state.
    include_empty_squares: Include empty squares in the dict with value "empty".

  Returns:
    A dictionary where the keys are the square notation (e.g., "e4") and
    the values are the piece names (e.g., "white pawn").
  """
  board = chess.Board(fen)

  piece_map = {}
  for square in chess.SQUARES:
    piece = board.piece_at(square)
    square_name = chess.square_name(square)
    if piece:
      piece_color = "white" if piece.color == chess.WHITE else "black"
      piece_type_name = chess.piece_name(piece.piece_type)
      piece_map[square_name] = f"{piece_color} {piece_type_name}"
    elif include_empty_squares:
      piece_map[square_name] = "empty"
  return piece_map


def get_ascii_board(game_short_name: str, state: pyspiel.State) -> str:
  """Returns an ASCII board representation of the game state."""
  if game_short_name.startswith("chess"):
    return get_chess_ascii_board(state.board().to_fen(), True)
  else:
    raise ValueError(
        f"get_ascii_board is not implemented for game: {game_short_name}"
    )


def get_chess_ascii_board(fen: str, caption: bool = False) -> str:
  """Returns an ASCII chess board representation given the FEN string."""
  ascii_lines = str(chess.Board(fen)).split("\n")
  ascii_lines_with_numbers = []
  for i, line in enumerate(ascii_lines):
    ascii_lines_with_numbers.append(line + " " + str(8 - i))
  ascii_lines_with_numbers_and_letters = ascii_lines_with_numbers + [
      "a b c d e f g h  "
  ]
  ascii_board = "\n".join(ascii_lines_with_numbers_and_letters)
  if caption:
    return (
        ascii_board
        + "\n"
        + "Uppercase letters represent white pieces and lowercase letters"
        " represent black pieces. The numbers in the right-most column denote"
        " the ranks and the letters in the bottom row denote the files."
    )
  return ascii_board
