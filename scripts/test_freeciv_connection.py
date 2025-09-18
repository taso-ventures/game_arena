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

"""Test FreeCiv3D server connection and basic functionality."""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import requests
import termcolor

# Add the parent directory to the path so we can import game_arena
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game_arena.harness.freeciv_client import FreeCivClient
from game_arena.harness.freeciv_state import FreeCivState

colored = termcolor.colored


def test_http_connection(server_url: str) -> bool:
    """Test basic HTTP connection to FreeCiv3D server."""
    print(colored("Testing HTTP connection...", "blue"))

    try:
        # Test basic server status
        response = requests.get(f"{server_url}/status", timeout=10)
        if response.status_code == 200:
            print(
                colored(
                    f"✓ Server status check passed ({response.status_code})", "green"
                )
            )
        else:
            print(
                colored(f"✗ Server status check failed ({response.status_code})", "red")
            )
            return False

        # Test game launcher endpoint
        response = requests.post(
            f"{server_url}/civclientlauncher",
            data={"action": "new", "type": "multiplayer"},
            timeout=10,
        )

        if response.status_code in [200, 201]:
            print(
                colored(
                    f"✓ Game launcher endpoint accessible ({response.status_code})",
                    "green",
                )
            )
            try:
                result = response.json()
                print(f"  Response: {json.dumps(result, indent=2)}")
            except:
                print(f"  Response (non-JSON): {response.text[:200]}...")
        else:
            print(
                colored(
                    f"✗ Game launcher endpoint failed ({response.status_code})", "red"
                )
            )
            print(f"  Response: {response.text[:200]}...")

        return True

    except requests.exceptions.ConnectError:
        print(colored("✗ Cannot connect to server - is FreeCiv3D running?", "red"))
        return False
    except requests.exceptions.Timeout:
        print(colored("✗ Connection timeout - server may be overloaded", "red"))
        return False
    except Exception as e:
        print(colored(f"✗ HTTP connection error: {e}", "red"))
        return False


def test_websocket_connection(ws_url: str) -> bool:
    """Test WebSocket connection to FreeCiv3D server."""
    print(colored("Testing WebSocket connection...", "blue"))

    try:
        import asyncio

        import websockets

        async def test_ws():
            try:
                websocket = await websockets.connect(ws_url, timeout=10)
                print(colored("✓ WebSocket connection established", "green"))

                # Send a test message
                test_message = {"type": "ping", "timestamp": time.time()}
                await websocket.send(json.dumps(test_message))
                print(colored("✓ Test message sent", "green"))

                # Try to receive a response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    print(colored(f"✓ Received response: {response[:100]}...", "green"))
                except asyncio.TimeoutError:
                    print(
                        colored(
                            "⚠ No response received (timeout) - this may be normal",
                            "yellow",
                        )
                    )

                await websocket.close()
                return True

            except Exception as e:
                print(colored(f"✗ WebSocket test failed: {e}", "red"))
                return False

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(test_ws())

    except ImportError:
        print(
            colored(
                "⚠ websockets library not available - skipping WebSocket test", "yellow"
            )
        )
        return True
    except Exception as e:
        print(colored(f"✗ WebSocket connection error: {e}", "red"))
        return False


def test_freeciv_client(server_url: str, ws_url: str) -> bool:
    """Test FreeCiv client functionality."""
    print(colored("Testing FreeCiv client...", "blue"))

    try:
        client = FreeCivClient(server_url, ws_url)

        # Test connection
        client.connect()
        print(colored("✓ FreeCiv client connected", "green"))

        # Test getting game state
        state_data = client.get_game_state()
        print(colored("✓ Game state retrieved", "green"))
        print(f"  State keys: {list(state_data.keys())}")

        # Test creating FreeCiv state adapter
        freeciv_state = FreeCivState(state_data)
        print(colored("✓ FreeCiv state adapter created", "green"))
        print(f"  Turn: {freeciv_state.turn}")
        print(f"  Phase: {freeciv_state.phase}")
        print(f"  Players: {len(freeciv_state.players)}")
        print(f"  Units: {len(freeciv_state.units)}")
        print(f"  Cities: {len(freeciv_state.cities)}")

        # Test getting legal actions
        legal_actions = freeciv_state.get_legal_actions(1)
        print(
            colored(f"✓ Legal actions retrieved: {len(legal_actions)} actions", "green")
        )

        if legal_actions:
            print("  Sample actions:")
            for i, action in enumerate(legal_actions[:3]):
                print(f"    {i+1}. {action.action_type} (actor: {action.actor_id})")

        # Test observation generation
        observation = freeciv_state.to_observation(1, format="enhanced")
        print(colored("✓ Enhanced observation generated", "green"))
        if "metrics" in observation:
            metrics = observation["metrics"]
            print(f"  Estimated tokens: {metrics.get('estimated_tokens', 'unknown')}")
            print(f"  Detail level: {metrics.get('detail_level', 'unknown')}")

        # Clean up
        client.disconnect()
        print(colored("✓ FreeCiv client disconnected", "green"))

        return True

    except Exception as e:
        print(colored(f"✗ FreeCiv client test failed: {e}", "red"))
        import traceback

        print(f"  Full traceback:\n{traceback.format_exc()}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test FreeCiv3D server connection")

    parser.add_argument(
        "--server_url",
        default=os.getenv("FREECIV_SERVER_URL", "http://localhost:8080"),
        help="FreeCiv3D server URL",
    )

    parser.add_argument(
        "--ws_url",
        default=os.getenv("FREECIV_WS_URL", "ws://localhost:4002"),
        help="FreeCiv3D WebSocket URL",
    )

    parser.add_argument(
        "--skip_websocket", action="store_true", help="Skip WebSocket connection test"
    )

    parser.add_argument(
        "--skip_client", action="store_true", help="Skip FreeCiv client test"
    )

    args = parser.parse_args()

    print(colored("FreeCiv3D Connection Test", "cyan"))
    print("=" * 50)
    print(f"Server URL: {args.server_url}")
    print(f"WebSocket URL: {args.ws_url}")
    print()

    all_tests_passed = True

    # Test HTTP connection
    if not test_http_connection(args.server_url):
        all_tests_passed = False

    print()

    # Test WebSocket connection
    if not args.skip_websocket:
        if not test_websocket_connection(args.ws_url):
            all_tests_passed = False
        print()

    # Test FreeCiv client
    if not args.skip_client:
        if not test_freeciv_client(args.server_url, args.ws_url):
            all_tests_passed = False
        print()

    # Summary
    print("=" * 50)
    if all_tests_passed:
        print(colored("✓ All tests passed! FreeCiv3D connection is working.", "green"))
        print()
        print("You can now run FreeCiv games with:")
        print(colored("  python scripts/run_game.py freeciv", "cyan"))
        sys.exit(0)
    else:
        print(colored("✗ Some tests failed. Please check your FreeCiv3D setup.", "red"))
        print()
        print("Common solutions:")
        print("1. Start FreeCiv3D server: cd ../freeciv3d && docker-compose up")
        print("2. Check server URL and port configuration")
        print("3. Ensure ports 8080 and 4002 are accessible")
        sys.exit(1)


if __name__ == "__main__":
    main()
