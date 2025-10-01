#!/usr/bin/env python3
"""Test script for LLM Gateway integration with Game Arena

This script tests:
1. WebSocket connection to LLM handler endpoint
2. LLM authentication protocol
3. State query handling
4. Action submission
5. Spectator data flow
"""

import asyncio
import json
import os
import sys
import time
import websockets
from pathlib import Path

# Add game_arena to path
sys.path.insert(0, str(Path(__file__).parent))

from game_arena.harness.freeciv_proxy_client import FreeCivProxyClient
from game_arena.harness.freeciv_state_sync import FreeCivStateSynchronizer
from game_arena.harness.freeciv_state import FreeCivState

# Test configuration
TEST_GAME_ID = f"test-{int(time.time())}"
TEST_AGENT_ID = f"test-agent-{TEST_GAME_ID[:8]}"
LLM_GATEWAY_URL = "ws://localhost:8003/llmsocket/8002"
API_TOKEN = os.getenv("LLM_API_TOKENS", "test-token-fc3d-001").split(",")[0]

async def test_llm_handler_connection():
    """Test direct connection to LLM handler"""
    print("\n=== Testing LLM Handler Connection ===")

    try:
        # Connect to the LLM handler endpoint
        ws = await websockets.connect("ws://localhost:8002/llmsocket/8002")
        print("✓ Connected to LLM handler WebSocket")

        # Should receive welcome message
        welcome = await asyncio.wait_for(ws.recv(), timeout=2.0)
        welcome_data = json.loads(welcome)
        print(f"✓ Received welcome: {welcome_data.get('type')} - {welcome_data.get('message', '')[:50]}...")

        # Send LLM authentication
        auth_msg = {
            "type": "llm_connect",
            "agent_id": TEST_AGENT_ID,
            "api_token": API_TOKEN,
            "port": 6001,
            "capabilities": ["unit_move", "city_production", "tech_research"]
        }

        await ws.send(json.dumps(auth_msg))
        print(f"✓ Sent LLM authentication for agent: {TEST_AGENT_ID}")

        # Wait for auth response
        auth_response = await asyncio.wait_for(ws.recv(), timeout=5.0)
        auth_data = json.loads(auth_response)

        if auth_data.get("type") == "auth_success":
            player_id = auth_data.get("player_id")
            session_id = auth_data.get("session_id")
            print(f"✓ Authentication successful!")
            print(f"  - Player ID: {player_id}")
            print(f"  - Session ID: {session_id}")
            print(f"  - Capabilities: {auth_data.get('capabilities', [])}")

            # Test state query
            state_query = {
                "type": "state_query",
                "format": "llm_optimized",
                "include_actions": True
            }

            await ws.send(json.dumps(state_query))
            print("✓ Sent state query")

            # Wait for state response
            state_response = await asyncio.wait_for(ws.recv(), timeout=10.0)
            state_data = json.loads(state_response)

            if state_data.get("type") == "state_response":
                print(f"✓ Received state response: format={state_data.get('format')}")
                game_state = state_data.get("data", {})
                print(f"  - Turn: {game_state.get('turn', 'N/A')}")
                print(f"  - Phase: {game_state.get('phase', 'N/A')}")

                if 'strategic_summary' in game_state:
                    summary = game_state['strategic_summary']
                    print(f"  - Cities: {summary.get('cities_count', 0)}")
                    print(f"  - Units: {summary.get('units_count', 0)}")
                    print(f"  - Tech Level: {summary.get('tech_progress', 'N/A')}")

                if 'legal_actions' in game_state:
                    actions = game_state['legal_actions']
                    print(f"  - Available actions: {len(actions)}")
                    if actions:
                        print(f"    • First action: {actions[0].get('type')} - {actions[0].get('description', '')}")
            else:
                print(f"✗ Unexpected response type: {state_data.get('type')}")
                print(f"  Response: {json.dumps(state_data, indent=2)}")

            await ws.close()
            return True

        else:
            print(f"✗ Authentication failed: {auth_data.get('message', 'Unknown error')}")
            await ws.close()
            return False

    except asyncio.TimeoutError:
        print("✗ Connection timed out")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

async def test_gateway_integration():
    """Test Game Arena integration through LLM Gateway"""
    print("\n=== Testing Game Arena Integration ===")

    try:
        # Create proxy client for Game Arena
        proxy = FreeCivProxyClient(
            host="localhost",
            port=8003,  # LLM Gateway port
            api_token=API_TOKEN
        )

        # Connect through gateway
        await proxy.connect()
        print("✓ Connected to LLM Gateway")

        # Get initial state
        state_data = await proxy.get_state()
        print(f"✓ Retrieved game state")

        if state_data:
            # Create FreeCivState object
            freeciv_state = FreeCivState(state_data)
            print(f"  - Turn: {freeciv_state.turn}")
            print(f"  - Players: {len(freeciv_state.players)}")
            print(f"  - Units: {len(freeciv_state.units)}")
            print(f"  - Cities: {len(freeciv_state.cities)}")

            # Get legal actions
            legal_actions = freeciv_state.get_legal_actions(player_id=1)
            print(f"  - Legal actions: {len(legal_actions)}")

            if legal_actions:
                # Try to send an action
                action = legal_actions[0]
                print(f"✓ Sending action: {action}")

                result = await proxy.send_action(action)
                if result.get("success"):
                    print(f"✓ Action executed successfully")
                else:
                    print(f"⚠ Action failed: {result.get('error', 'Unknown error')}")

        await proxy.disconnect()
        print("✓ Disconnected from gateway")
        return True

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_spectator_connection():
    """Test spectator WebSocket connection"""
    print("\n=== Testing Spectator Connection ===")

    try:
        # Connect as spectator
        ws = await websockets.connect(f"ws://localhost:8003/ws/spectator/{TEST_GAME_ID}")
        print(f"✓ Connected as spectator to game: {TEST_GAME_ID}")

        # Send initial message
        await ws.send(json.dumps({
            "type": "spectator_join",
            "game_id": TEST_GAME_ID
        }))

        print("✓ Waiting for game updates...")

        # Listen for a few messages
        for i in range(3):
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                msg_data = json.loads(message)
                print(f"  Message {i+1}: type={msg_data.get('type')}")

                if msg_data.get("type") == "state_update":
                    print(f"    - Turn: {msg_data.get('turn')}")
                    print(f"    - Players: {len(msg_data.get('players', {}))}")
                    print(f"    - Units: {len(msg_data.get('units', []))}")
                    print(f"    - Cities: {len(msg_data.get('cities', []))}")
                    print(f"    - Visible tiles: {len(msg_data.get('visible_tiles', []))}")

            except asyncio.TimeoutError:
                print(f"  No message {i+1} (timeout)")

        await ws.close()
        print("✓ Spectator connection test complete")
        return True

    except Exception as e:
        print(f"✗ Spectator test failed: {e}")
        return False

async def main():
    """Run all integration tests"""
    print("=" * 60)
    print("LLM Gateway Integration Test Suite")
    print("=" * 60)

    results = []

    # Test 1: Direct LLM handler connection
    result1 = await test_llm_handler_connection()
    results.append(("LLM Handler Connection", result1))

    # Small delay between tests
    await asyncio.sleep(1)

    # Test 2: Game Arena integration
    result2 = await test_gateway_integration()
    results.append(("Game Arena Integration", result2))

    # Test 3: Spectator connection
    result3 = await test_spectator_connection()
    results.append(("Spectator Connection", result3))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30} {status}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print("-" * 60)
    print(f"Total: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed! Integration is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)