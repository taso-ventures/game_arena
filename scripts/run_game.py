#!/usr/bin/env python3
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

"""Unified script to run games for Chess, Go, or FreeCiv."""

import argparse
import os
import sys
from typing import Optional

import termcolor

# Add the parent directory to the path so we can import game_arena
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

colored = termcolor.colored


def run_chess_game(
    num_moves: int, parser_choice: str, gemini_model: str, openai_model: str
) -> None:
    """Run a chess game using the existing harness demo."""
    from game_arena.harness import harness_demo

    # Set up command line arguments for the chess demo
    sys.argv = [
        "harness_demo.py",
        f"--num_moves={num_moves}",
        f"--parser_choice={parser_choice}",
        f"--gemini_model={gemini_model}",
        f"--openai_model={openai_model}",
    ]

    print(colored(f"Running Chess game with {num_moves} moves...", "green"))
    harness_demo.main(None)


def run_freeciv_game(
    num_moves: int,
    parser_choice: str,
    gemini_model: str,
    openai_model: str,
    freeciv_server_url: Optional[str] = None,
    freeciv_ws_url: Optional[str] = None,
) -> None:
    """Run a FreeCiv game using the FreeCiv harness demo."""
    from game_arena.harness import freeciv_harness_demo

    # Set up command line arguments for the FreeCiv demo
    args = [
        "freeciv_harness_demo.py",
        f"--num_moves={num_moves}",
        f"--parser_choice={parser_choice}",
        f"--gemini_model={gemini_model}",
        f"--openai_model={openai_model}",
    ]

    if freeciv_server_url:
        args.append(f"--freeciv_server_url={freeciv_server_url}")
    if freeciv_ws_url:
        args.append(f"--freeciv_ws_url={freeciv_ws_url}")

    sys.argv = args

    print(colored(f"Running FreeCiv game with {num_moves} moves...", "green"))
    print(
        colored(
            f"FreeCiv server: {freeciv_server_url or os.getenv('FREECIV_SERVER_URL', 'http://localhost:8080')}",
            "blue",
        )
    )
    freeciv_harness_demo.main(None)


def run_go_game(
    num_moves: int, parser_choice: str, gemini_model: str, openai_model: str
) -> None:
    """Run a Go game (placeholder - would need to be implemented)."""
    print(colored("Go game support not yet implemented", "yellow"))
    print(
        "You can use the existing harness_demo.py with pyspiel.load_game('go') to test Go games"
    )


def check_freeciv_connection() -> bool:
    """Check if FreeCiv3D server is accessible."""
    import requests

    server_url = os.getenv("FREECIV_SERVER_URL", "http://localhost:8080")

    try:
        response = requests.get(f"{server_url}/status", timeout=10)
        if response.status_code == 200:
            print(colored(f"✓ FreeCiv3D server accessible at {server_url}", "green"))
            return True
        else:
            print(
                colored(
                    f"✗ FreeCiv3D server returned status {response.status_code}", "red"
                )
            )
            return False
    except requests.exceptions.RequestException as e:
        print(
            colored(f"✗ Cannot connect to FreeCiv3D server at {server_url}: {e}", "red")
        )
        return False


def check_api_keys() -> bool:
    """Check if required API keys are configured."""
    missing_keys = []

    if not os.getenv("GEMINI_API_KEY"):
        missing_keys.append("GEMINI_API_KEY")

    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")

    if missing_keys:
        print(colored(f"✗ Missing API keys: {', '.join(missing_keys)}", "red"))
        print(
            colored(
                "Please set these environment variables or copy .env.example to .env and fill in your keys",
                "yellow",
            )
        )
        return False
    else:
        print(colored("✓ API keys configured", "green"))
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Run Game Arena games (Chess, Go, or FreeCiv)"
    )

    parser.add_argument(
        "game_type", choices=["chess", "go", "freeciv"], help="Type of game to run"
    )

    parser.add_argument(
        "--num_moves",
        type=int,
        default=10,
        help="Number of moves to play (default: 10)",
    )

    parser.add_argument(
        "--parser_choice",
        choices=["rule_then_soft", "llm_only"],
        default="rule_then_soft",
        help="Parser strategy to use (default: rule_then_soft)",
    )

    parser.add_argument(
        "--gemini_model",
        default="gemini-2.5-flash",
        help="Gemini model for player 1 (default: gemini-2.5-flash)",
    )

    parser.add_argument(
        "--openai_model",
        default="gpt-4.1",
        help="OpenAI model for player 2 (default: gpt-4.1)",
    )

    parser.add_argument(
        "--freeciv_server_url",
        help="FreeCiv3D server URL (default: from env FREECIV_SERVER_URL or http://localhost:8080)",
    )

    parser.add_argument(
        "--freeciv_ws_url",
        help="FreeCiv3D WebSocket URL (default: from env FREECIV_WS_URL or ws://localhost:4002)",
    )

    parser.add_argument(
        "--check_setup",
        action="store_true",
        help="Check setup (API keys, FreeCiv server) and exit",
    )

    args = parser.parse_args()

    print(colored("Game Arena - Unified Game Runner", "cyan"))
    print("=" * 50)

    # Check setup if requested
    if args.check_setup:
        print("Checking setup...")
        api_keys_ok = check_api_keys()

        if args.game_type == "freeciv":
            freeciv_ok = check_freeciv_connection()
            if api_keys_ok and freeciv_ok:
                print(
                    colored("✓ Setup looks good! Ready to run FreeCiv games.", "green")
                )
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            if api_keys_ok:
                print(
                    colored(
                        f"✓ Setup looks good! Ready to run {args.game_type} games.",
                        "green",
                    )
                )
                sys.exit(0)
            else:
                sys.exit(1)

    # Check API keys before running
    if not check_api_keys():
        sys.exit(1)

    # Run the specified game
    try:
        if args.game_type == "chess":
            run_chess_game(
                args.num_moves, args.parser_choice, args.gemini_model, args.openai_model
            )

        elif args.game_type == "go":
            run_go_game(
                args.num_moves, args.parser_choice, args.gemini_model, args.openai_model
            )

        elif args.game_type == "freeciv":
            # Check FreeCiv server connection before running
            if not check_freeciv_connection():
                print(
                    colored(
                        "Please ensure FreeCiv3D server is running on port 8080",
                        "yellow",
                    )
                )
                print(
                    colored(
                        "You can start it by running: cd ../freeciv3d && docker-compose up",
                        "yellow",
                    )
                )
                sys.exit(1)

            run_freeciv_game(
                args.num_moves,
                args.parser_choice,
                args.gemini_model,
                args.openai_model,
                args.freeciv_server_url,
                args.freeciv_ws_url,
            )

        print(
            colored(f"\n{args.game_type.title()} game completed successfully!", "green")
        )

    except KeyboardInterrupt:
        print(colored("\nGame interrupted by user", "yellow"))
        sys.exit(0)
    except Exception as e:
        print(colored(f"\nError running {args.game_type} game: {e}", "red"))
        sys.exit(1)


if __name__ == "__main__":
    main()
