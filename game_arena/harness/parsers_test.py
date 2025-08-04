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

"""Tests for move parsers."""

from game_arena.harness import parsers
from absl.testing import absltest
from absl.testing import parameterized


class MoveParsingTest(parameterized.TestCase):

  def test_trailing_newline(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: e5
"""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )

  @parameterized.named_parameters(
      ('one_whitespace', 1),
      ('two_whitespace', 2),
  )
  def test_trailing_whitespace(self, num_trailing_whitespace):
    raw_response_with_no_trailing_whitespace = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: e5"""
    with_trailing_whitespace = (
        raw_response_with_no_trailing_whitespace + ' ' * num_trailing_whitespace
    )
    self.assertEqual(
        parsers.parse_move_from_response(with_trailing_whitespace)['move'],
        'e5',
    )

  def test_single_backslash_boxed_answer(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: \boxed{e5}"""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )

  def test_double_backslash_boxed_answer(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: \\boxed{e5}"""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )

  def test_single_backslash_text_answer(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: \text{e5}"""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )

  def test_double_backslash_text_answer(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: \\text{e5}"""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )

  def test_latex_dollar_sign_answer(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: $e5$"""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )

  def test_space_and_period_at_end(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: e5 ."""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )

  def test_final_answer_tag(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

The final answer is e5
"""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )

  def test_html_tags(self):
    raw_response = """
Okay, I'm playing Black. The position is:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1

White has played 1. e4. This is the King's Pawn Opening. My most common and solid response is to play e5, meeting White's central pawn challenge. This controls the center, develops a pawn, and prepares to develop pieces.

Other reasonable options exist, like c5 (the Sicilian Defense), e6 (the French Defense), and Nc6. However, e5 is a very solid and principled reply.

Final Answer: <answer>e5</answer>"""
    self.assertEqual(
        parsers.parse_move_from_response(raw_response)['move'],
        'e5',
    )


class ChessSoftParserV1Test(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'simple_move',
          'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
          'e4',
          ['e4'],
          'e4',
      ),
      (
          'with_move_number',
          'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
          '1. e4',
          ['e4'],
          'e4',
      ),
      (
          'with_dots',
          'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2',
          '2...Nf3',
          ['Nf3'],
          'Nf3',
      ),
      (
          'with_extra_chars',
          'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
          'e4.',
          ['e4'],
          'e4',
      ),
      (
          'ambiguous_move_resolvable',
          (
              'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq'
              ' - 0 2'
          ),
          'Nxf7',
          ['Nxf7'],
          'Nxf7',
      ),
      (
          'invalid_move',
          'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
          'e5',
          ['e4'],
          None,
      ),
      (
          'not_in_legal_moves',
          'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
          'd4',
          ['e4'],
          None,
      ),
      (
          'en_passant',
          'r1b2rk1/1p4pp/p1n1p3/4Pp2/4R3/2NB4/PPP3PP/2K4R w - f6 0 17',
          'exf6e.p',
          ['exf6'],
          'exf6',
      ),
  )
  def test_chess_soft_parser(
      self, state_str, selected_action, spiel_legal_moves, expected_move
  ):
    self.assertEqual(
        parsers._chess_soft_parser_v1(
            state_str, selected_action, spiel_legal_moves
        ),
        expected_move,
    )


if __name__ == '__main__':
  absltest.main()
