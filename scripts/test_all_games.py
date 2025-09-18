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

"""Test all Game Arena games (Chess, Go, FreeCiv) with comprehensive validation."""

import argparse
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import termcolor

colored = termcolor.colored


class GameTestRunner:
    """Runner for testing all Game Arena games."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.test_results: Dict[str, Dict[str, Any]] = {}

    def run_test(
        self, game_type: str, num_moves: int = 3, parser_choice: str = "rule_then_soft"
    ) -> bool:
        """Run a test for a specific game type.

        Args:
            game_type: Type of game (chess, go, freeciv)
            num_moves: Number of moves to test
            parser_choice: Parser strategy to use

        Returns:
            True if test passed, False otherwise
        """
        print(colored(f"\nTesting {game_type.upper()} game...", "blue"))
        print("-" * 40)

        start_time = time.time()
        try:
            # Build command
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "run_game.py"),
                game_type,
                f"--num_moves={num_moves}",
                f"--parser_choice={parser_choice}",
            ]

            # Run the test
            if self.verbose:
                print(f"Running command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
            )

            end_time = time.time()
            duration = end_time - start_time

            # Store results
            self.test_results[game_type] = {
                "success": result.returncode == 0,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

            if result.returncode == 0:
                print(
                    colored(
                        f"✓ {game_type.title()} test PASSED ({duration:.1f}s)", "green"
                    )
                )
                if self.verbose and result.stdout:
                    print("Output:")
                    print(result.stdout[-500:])  # Last 500 chars
                return True
            else:
                print(
                    colored(
                        f"✗ {game_type.title()} test FAILED ({duration:.1f}s)", "red"
                    )
                )
                print(f"Return code: {result.returncode}")
                if result.stderr:
                    print("Error output:")
                    print(result.stderr[-500:])  # Last 500 chars
                return False

        except subprocess.TimeoutExpired:
            print(colored(f"✗ {game_type.title()} test TIMEOUT (>300s)", "red"))
            self.test_results[game_type] = {
                "success": False,
                "duration": 300,
                "stdout": "",
                "stderr": "Test timed out",
                "returncode": -1,
            }
            return False
        except Exception as e:
            print(colored(f"✗ {game_type.title()} test ERROR: {e}", "red"))
            self.test_results[game_type] = {
                "success": False,
                "duration": time.time() - start_time,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
            }
            return False

    def run_setup_check(self, game_type: str) -> bool:
        """Run setup check for a specific game type.

        Args:
            game_type: Type of game to check

        Returns:
            True if setup is OK, False otherwise
        """
        try:
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "run_game.py"),
                game_type,
                "--check_setup",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            return result.returncode == 0

        except Exception:
            return False

    def check_dependencies(self) -> bool:
        """Check that all required dependencies are available."""
        print(colored("Checking dependencies...", "blue"))

        missing_deps = []

        # Check Python packages
        required_packages = ["pyspiel", "termcolor", "requests"]
        for package in required_packages:
            try:
                __import__(package)
                print(f"✓ {package}")
            except ImportError:
                print(colored(f"✗ {package}", "red"))
                missing_deps.append(package)

        # Check optional packages
        optional_packages = ["websockets", "aiohttp"]
        for package in optional_packages:
            try:
                __import__(package)
                print(f"✓ {package} (optional)")
            except ImportError:
                print(
                    colored(
                        f"⚠ {package} (optional - needed for FreeCiv WebSocket)",
                        "yellow",
                    )
                )

        if missing_deps:
            print(
                colored(
                    f"\nMissing required dependencies: {', '.join(missing_deps)}", "red"
                )
            )
            print("Install with: pip install " + " ".join(missing_deps))
            return False

        return True

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print(colored("TEST SUMMARY", "cyan"))
        print("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r["success"])

        for game_type, result in self.test_results.items():
            status = (
                colored("PASSED", "green")
                if result["success"]
                else colored("FAILED", "red")
            )
            duration = result["duration"]
            print(f"{game_type.ljust(15)} {status.ljust(15)} ({duration:.1f}s)")

            if not result["success"] and self.verbose:
                print(f"  Error: {result['stderr'][:100]}...")

        print("-" * 60)
        print(f"Total: {passed_tests}/{total_tests} tests passed")

        if passed_tests == total_tests:
            print(colored("🎉 All tests passed!", "green"))
            return True
        else:
            print(colored("❌ Some tests failed", "red"))
            return False


def main():
    parser = argparse.ArgumentParser(description="Test all Game Arena games")

    parser.add_argument(
        "--games",
        nargs="+",
        choices=["chess", "go", "freeciv"],
        default=[
            "chess",
            "freeciv",
        ],  # Skip go by default as it's not fully implemented
        help="Games to test (default: chess freeciv)",
    )

    parser.add_argument(
        "--num_moves",
        type=int,
        default=3,
        help="Number of moves to test per game (default: 3)",
    )

    parser.add_argument(
        "--parser_choice",
        choices=["rule_then_soft", "llm_only"],
        default="rule_then_soft",
        help="Parser strategy to use (default: rule_then_soft)",
    )

    parser.add_argument(
        "--skip_setup_check",
        action="store_true",
        help="Skip setup checks before running tests",
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run tests sequentially (default: sequential)",
    )

    args = parser.parse_args()

    print(colored("Game Arena - Comprehensive Test Suite", "cyan"))
    print("=" * 60)

    runner = GameTestRunner(verbose=args.verbose)

    # Check dependencies
    if not runner.check_dependencies():
        sys.exit(1)

    print()

    # Run setup checks
    if not args.skip_setup_check:
        print(colored("Running setup checks...", "blue"))
        all_setup_ok = True

        for game_type in args.games:
            print(f"Checking {game_type}...")
            if runner.run_setup_check(game_type):
                print(colored(f"✓ {game_type} setup OK", "green"))
            else:
                print(colored(f"✗ {game_type} setup failed", "red"))
                all_setup_ok = False

        if not all_setup_ok:
            print(
                colored(
                    "\nSetup checks failed. Please fix issues before running tests.",
                    "red",
                )
            )
            sys.exit(1)

        print()

    # Run tests
    print(colored("Running game tests...", "blue"))
    all_tests_passed = True

    for game_type in args.games:
        success = runner.run_test(game_type, args.num_moves, args.parser_choice)
        if not success:
            all_tests_passed = False

        # Add small delay between tests
        if len(args.games) > 1:
            time.sleep(2)

    # Print summary
    runner.print_summary()

    # Exit with appropriate code
    sys.exit(0 if all_tests_passed else 1)


if __name__ == "__main__":
    main()
