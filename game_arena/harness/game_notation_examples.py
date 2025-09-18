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

"""File stores some state, action notations for each game."""

################################################################################
##### CHESS #####
################################################################################
chess_state = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
chess_action = "a3; a4; Na3; Nc3"
chess_player_map = {0: "Black", 1: "White"}
chess_notation = "standard algebraic notation (SAN)"
chess_state_notation = "Forsyth-Edwards Notation (FEN) notation"

################################################################################
##### CONNECT FOUR #####
################################################################################
connect_four_state = """.......
.......
.......
.......
.......
......x

"""
connect_four_action = "o2; o6; x0. The game begins with player x."
connect_four_player_map = {0: "x", 1: "o"}
connect_four_notation = "zero-based column number"
connect_four_state_notation = "ASCII notation"

################################################################################
##### TIC TAC TOE #####
################################################################################
tic_tac_toe_state = """x..
...
...
"""
tic_tac_toe_action = "o(0,1); o(2,0); x(2,1). The game begins with player x."
tic_tac_toe_player_map = {0: "x", 1: "o"}
tic_tac_toe_notation = "zero-based (row,col)"
tic_tac_toe_state_notation = "ASCII notation"


################################################################################
##### GO #####
################################################################################
go_state = """#  7 +++++++
#  6 +++++++
#  5 +++++++
#  4 +++++++
#  3 +++++++
#  2 +++++++
#  1 +++++++
#    ABCDEFG
"""
go_action = "B a1; B b1; B c1; B d1; B e1"
go_player_map = {0: "B", 1: "W"}
go_notation = "col,row using coordinates in the grid"
go_state_notation = "ASCII notation"


################################################################################
##### UNIVERSAL POKER #####
################################################################################
universal_poker_state = """# BettingAbstraction: FULLGAME
# P0 Cards: 5s
# BoardCards
# Node type?: Player node for player 0
# ]
# Round: 0
# Spent: [P0: 100  P1: 100  ]
#
# Action Sequence: dd
"""
universal_poker_action = (
    "player=0 move=Call; player=0 move=Bet200; player=0 move=Bet201;"
)
universal_poker_player_map = {0: "0", 1: "1"}
universal_poker_notation = (
    "Call, Fold or BetX where X is the number of chips you are betting"
)
universal_poker_state_notation = "text"


GAME_SPECIFIC_NOTATIONS = {
    "chess": {
        "state": chess_state,
        "action": chess_action,
        "player_map": chess_player_map,
        "move_notation": chess_notation,
        "state_notation": chess_state_notation,
    },
    "connect_four": {
        "state": connect_four_state,
        "action": connect_four_action,
        "player_map": connect_four_player_map,
        "move_notation": connect_four_notation,
        "state_notation": connect_four_state_notation,
    },
    "tic_tac_toe": {
        "state": tic_tac_toe_state,
        "action": tic_tac_toe_action,
        "player_map": tic_tac_toe_player_map,
        "move_notation": tic_tac_toe_notation,
        "state_notation": tic_tac_toe_state_notation,
    },
    "go": {
        "state": go_state,
        "action": go_action,
        "player_map": go_player_map,
        "move_notation": go_notation,
        "state_notation": go_state_notation,
    },
    "universal_poker": {
        "state": universal_poker_state,
        "action": universal_poker_action,
        "player_map": universal_poker_player_map,
        "move_notation": universal_poker_notation,
        "state_notation": universal_poker_state_notation,
    },
}
