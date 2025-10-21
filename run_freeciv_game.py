#!/usr/bin/env python3
"""Run LLMs playing FreeCiv using the Game Arena and FreeCiv3D LLM Gateway.

This script orchestrates a game between two LLM agents using the full Game Arena
framework with FreeCiv3D integration.

Prerequisites:
1. FreeCiv3D server running: cd ../freeciv3d && docker-compose up
2. API keys configured in .env file
3. Game Arena container running: docker-compose up game-arena

Usage:
  # Run from Docker container
  docker exec game-arena python run_freeciv_game.py

  # Or run with custom options
  docker exec game-arena python run_freeciv_game.py --turns=50 --player1=gemini --player2=openai
"""

import asyncio
import os
import sys
import time
import uuid
from typing import Optional

import requests
from absl import app, flags
import termcolor

# dotenv is not available in the container, so we'll read .env manually if it exists
def load_dotenv():
    """Simple .env loader."""
    try:
        with open('/app/.env', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    except FileNotFoundError:
        pass  # .env file doesn't exist, use existing environment

from game_arena.harness.freeciv_proxy_client import FreeCivProxyClient
from game_arena.harness.freeciv_llm_agent import FreeCivLLMAgent
from game_arena.harness.model_generation_sdk import (
    AIStudioModel,
    OpenAIChatCompletionsModel,
    AnthropicModel
)

colored = termcolor.colored

# Command line flags
_MAX_TURNS = flags.DEFINE_integer(
    "turns", 50, "Maximum number of turns to play"
)
_PLAYER1_MODEL = flags.DEFINE_enum(
    "player1", "gemini", ["gemini", "openai", "anthropic"],
    "Model for Player 1"
)
_PLAYER2_MODEL = flags.DEFINE_enum(
    "player2", "openai", ["gemini", "openai", "anthropic"],
    "Model for Player 2"
)
_FREECIV_HOST = flags.DEFINE_string(
    "host", "fciv-net", "FreeCiv3D server host"
)
_FREECIV_WS_PORT = flags.DEFINE_integer(
    "ws_port", 8003, "FreeCiv3D WebSocket port (4002 for standard, 8003 for LLM Gateway)"
)
_FREECIV_HTTP_PORT = flags.DEFINE_integer(
    "http_port", 8080, "FreeCiv3D HTTP server port"
)
_API_TOKEN = flags.DEFINE_string(
    "api_token", "test-token-fc3d-001", "API token for FreeCiv3D LLM Gateway"
)
_STRATEGY1 = flags.DEFINE_string(
    "strategy1", "balanced", "Strategy for Player 1"
)
_STRATEGY2 = flags.DEFINE_string(
    "strategy2", "aggressive_expansion", "Strategy for Player 2"
)
_VERBOSE = flags.DEFINE_boolean(
    "verbose", True, "Show detailed game progress"
)
_AUTO_SELECT_NATIONS = flags.DEFINE_boolean(
    "auto_select_nations", True, "Automatically select nations for players"
)
_PLAYER1_NATION = flags.DEFINE_string(
    "player1_nation", "Americans", "Nation for Player 1 (Americans, Romans, Chinese, French, Germans, British, Japanese, Indians, Russians, Spanish)"
)
_PLAYER2_NATION = flags.DEFINE_string(
    "player2_nation", "Romans", "Nation for Player 2"
)
_PLAYER1_LEADER = flags.DEFINE_string(
    "player1_leader", "AI Player 1", "Leader name for Player 1"
)
_PLAYER2_LEADER = flags.DEFINE_string(
    "player2_leader", "AI Player 2", "Leader name for Player 2"
)


def check_freeciv_server(host: str, port: int) -> bool:
    """Check if FreeCiv3D server is running."""
    try:
        # Just check if the main FreeCiv3D page loads
        try:
            response = requests.get(f"http://{host}:{port}/", timeout=5)
            if response.status_code == 200 and "freeciv" in response.text.lower():
                return True
            return False
        except:
            return False
    except Exception:
        return False


def create_model(model_type: str, api_keys: dict):
    """Create LLM model based on type."""
    if model_type == "gemini":
        return AIStudioModel(
            model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            api_key=api_keys.get("gemini")
        )
    elif model_type == "openai":
        return OpenAIChatCompletionsModel(
            model_name=os.getenv("OPENAI_MODEL", "gpt-4.1"),
            api_key=api_keys.get("openai")
        )
    elif model_type == "anthropic":
        return AnthropicModel(
            model_name=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
            api_key=api_keys.get("anthropic")
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


async def run_freeciv_game():
    """Run the FreeCiv LLM vs LLM game."""

    # Load environment variables
    load_dotenv()

    # Get API keys
    api_keys = {
        "gemini": os.getenv("GEMINI_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY")
    }

    # Get LLM Gateway API token
    llm_api_token = os.getenv("LLM_API_TOKEN", _API_TOKEN.value)

    # Validate required API keys
    required_keys = [_PLAYER1_MODEL.value, _PLAYER2_MODEL.value]
    for model_type in required_keys:
        if not api_keys.get(model_type):
            print(colored(f"✗ Missing API key for {model_type.upper()}", "red"))
            print(f"Please set {model_type.upper()}_API_KEY in your .env file")
            return False

    # Check FreeCiv3D server (skipping HTTP check, testing WebSocket directly)
    print(colored("Skipping HTTP check, will test LLM Gateway WebSocket directly...", "blue"))

    try:
        # Generate unique game_id for this match
        game_id = f"game_{uuid.uuid4().hex[:8]}"

        # Create SEPARATE proxy clients for each agent with nation preferences
        print(colored("Creating proxy connections for both players...", "blue"))
        proxy1 = FreeCivProxyClient(
            host=_FREECIV_HOST.value,
            port=_FREECIV_WS_PORT.value,
            agent_id=f"agent_player1_{uuid.uuid4().hex[:8]}",
            game_id=game_id,  # Same game_id for both players
            api_token=llm_api_token,
            nation=_PLAYER1_NATION.value if _AUTO_SELECT_NATIONS.value else None,
            leader_name=_PLAYER1_LEADER.value
        )

        proxy2 = FreeCivProxyClient(
            host=_FREECIV_HOST.value,
            port=_FREECIV_WS_PORT.value,
            agent_id=f"agent_player2_{uuid.uuid4().hex[:8]}",
            game_id=game_id,  # Same game_id for both players
            api_token=llm_api_token,
            nation=_PLAYER2_NATION.value if _AUTO_SELECT_NATIONS.value else None,
            leader_name=_PLAYER2_LEADER.value
        )

        # Track timing for diagnostics
        auth_start_time = time.time()

        # Connect Player 1
        print(colored(f"[{time.time():.1f}] Connecting Player 1 to LLM Gateway...", "blue"))
        await proxy1.connect()
        player1_auth_time = time.time() - auth_start_time
        print(colored(f"[{time.time():.1f}] ✓ Player 1 connected (took {player1_auth_time:.1f}s)", "green"))

        # Connect Player 2
        player2_start = time.time()
        print(colored(f"[{time.time():.1f}] Connecting Player 2 to LLM Gateway...", "blue"))
        await proxy2.connect()
        player2_auth_time = time.time() - player2_start
        print(colored(f"[{time.time():.1f}] ✓ Player 2 connected (took {player2_auth_time:.1f}s)", "green"))

        # Wait for game_ready signal from server (event-driven with timeout)
        # FreeCiv3D will send game_ready when:
        # - All players have connected and authenticated
        # - Nations have been assigned
        # - Units and cities have been created
        # - Game is ready for actions
        print(colored(f"[{time.time():.1f}] ⏳ Waiting for game_ready signal from server (max 45s)...", "yellow"))

        # Increased timeout to 45s for slow Docker environments
        max_wait = 45.0
        try:
            await asyncio.wait_for(proxy1.game_ready_event.wait(), timeout=max_wait)
            game_ready_time = time.time() - auth_start_time
            print(colored(f"[{time.time():.1f}] ✅ Game ready signal received! (total time: {game_ready_time:.1f}s)", "green"))
        except asyncio.TimeoutError:
            timeout_time = time.time() - auth_start_time
            print(colored(f"[{time.time():.1f}] ⚠️ Timeout after {timeout_time:.1f}s waiting for game_ready", "yellow"))

            # CRITICAL: Check if event was set during race condition
            if proxy1.game_ready_event.is_set():
                print(colored("   ✅ Game ready event WAS set (caught race condition!)", "green"))
            else:
                print(colored("   ❌ No game_ready signal - game may not be initialized", "yellow"))
                print(colored("   Proceeding with state validation...", "yellow"))

        # Verify game state is properly initialized before starting game loop
        print(colored("🔍 Verifying game initialization...", "cyan"))

        # Initialize state variables (defined outside try block so they're always available)
        units1, cities1 = [], []

        try:
            # Check both proxies for game state
            state1 = await proxy1.get_state()
            state2 = await proxy2.get_state()

            # Log player info
            players1 = state1.get("players", [])
            units1 = state1.get("units", [])
            cities1 = state1.get("cities", [])

            print(colored(f"Player 1 state: {len(players1)} players, {len(units1)} units, {len(cities1)} cities", "blue"))
            print(colored(f"Turn: {state1.get('turn', 'unknown')}, Phase: {state1.get('game', {}).get('phase', 'unknown')}", "blue"))

            # Verify nation assignment
            if players1:
                for i, player in enumerate(players1[:2]):  # Check first 2 players
                    player_nation = player.get("nation", "unassigned")
                    player_name = player.get("name", f"Player {i+1}")
                    print(colored(f"  {player_name}: nation={player_nation}", "blue"))

                    if player_nation == "unassigned" or not player_nation:
                        print(colored(f"    ⚠️ Nation not assigned to {player_name}!", "yellow"))

            # Check game_ready flag
            if proxy1.game_ready:
                print(colored("✅ Game ready signal confirmed", "green"))
            else:
                print(colored("⚠️ No game_ready signal received yet", "yellow"))

            # If game looks uninitialized, wait longer
            if not units1 and not cities1:
                print(colored("⚠️ Game not fully initialized (no units/cities), waiting additional 10s...", "yellow"))
                await asyncio.sleep(10)

                # Re-check after additional wait
                state1 = await proxy1.get_state()
                units1 = state1.get("units", [])
                cities1 = state1.get("cities", [])
                print(colored(f"After additional wait: {len(units1)} units, {len(cities1)} cities", "blue"))

                if not units1 and not cities1:
                    print(colored("❌ WARNING: Game still appears uninitialized! Proceeding anyway...", "yellow"))
                    print(colored("   This may result in only tech_research actions being available.", "yellow"))

        except Exception as e:
            print(colored(f"⚠️ Error verifying game state: {e}", "yellow"))
            print(colored("Proceeding with game anyway...", "yellow"))

        # Create LLM models
        print(colored("Creating LLM agents...", "blue"))
        model1 = create_model(_PLAYER1_MODEL.value, api_keys)
        model2 = create_model(_PLAYER2_MODEL.value, api_keys)

        # Create agents using the FreeCiv LLM Agent framework
        # Note: FreeCivLLMAgent constructor takes specific parameters
        agent1 = FreeCivLLMAgent(
            model=model1,
            strategy=_STRATEGY1.value,
            use_rethinking=True,
            max_rethinks=2,
            memory_size=10,
            fallback_to_random=True
        )

        agent2 = FreeCivLLMAgent(
            model=model2,
            strategy=_STRATEGY2.value,
            use_rethinking=True,
            max_rethinks=2,
            memory_size=10,
            fallback_to_random=True
        )

        print(colored(f"✓ Created agents:", "green"))
        print(f"  Player 1: {_PLAYER1_MODEL.value.upper()} ({_STRATEGY1.value})")
        print(f"  Player 2: {_PLAYER2_MODEL.value.upper()} ({_STRATEGY2.value})")

        # Display initialization timing summary
        total_init_time = time.time() - auth_start_time
        print(colored(f"\n📊 Initialization Timing Summary:", "cyan"))
        print(colored(f"  Player 1 auth: {player1_auth_time:.1f}s", "blue"))
        print(colored(f"  Player 2 auth: {player2_auth_time:.1f}s", "blue"))
        print(colored(f"  Total initialization: {total_init_time:.1f}s", "blue"))
        if proxy1.game_ready:
            print(colored(f"  ✅ Game ready signal: RECEIVED", "green"))
        else:
            print(colored(f"  ⚠️ Game ready signal: NOT RECEIVED", "yellow"))

        # Display spectator URLs (with game readiness status)
        print(colored("\n" + "=" * 60, "cyan"))
        print(colored("📺 SPECTATOR MODE", "cyan"))
        print(colored("=" * 60, "cyan"))
        print(f"Game ID: {game_id}")

        # Display game readiness status
        if units1 and cities1 and proxy1.game_ready:
            print(colored("✅ Game is fully initialized and ready for viewing", "green"))
        elif units1 or cities1:
            print(colored("⚠️ Game partially initialized (spectator may show incomplete state)", "yellow"))
        else:
            print(colored("❌ WARNING: Game not initialized yet (spectator may show pre-game lobby)", "yellow"))
            print(colored("   Wait a few moments and refresh the spectator page", "yellow"))

        # LLM Gateway Spectator URL (uses broadcast system, not direct civserver connection)
        print(f"\nSpectator URL (LLM Gateway Broadcast - Player 1 View):")
        spectator_url = f"http://localhost:8080/webclient/spectator.jsp?game_id={game_id}"
        print(colored(f"  {spectator_url}", "green"))
        print(f"\n📝 Note: Spectator connects to LLM Gateway (ws://localhost:8003/ws/spectator/{game_id})")
        print(f"   Receives Player 1's perspective via packet broadcast")
        civserver_port = proxy1.civserver_port or 6000
        print(f"   Game running on civserver port {civserver_port}")
        print(f"\n🔧 Debug Commands:")
        print(f"  Check LLM Gateway logs: docker exec fciv-net grep '{game_id}' /docker/llm-gateway/logs/*.log 2>/dev/null")
        print(f"  Check proxy logs: docker exec fciv-net cat /docker/logs/freeciv-proxy-8002.log | grep '{game_id}'")
        print(f"  WebSocket test (spectator): wscat -c ws://localhost:8003/ws/spectator/{game_id}")
        print(colored("=" * 60 + "\n", "cyan"))

        # Game loop
        print(colored(f"\\nStarting FreeCiv LLM vs LLM game (max {_MAX_TURNS.value} turns)...", "green"))
        print("=" * 60)

        turn = 0
        game_over = False

        while turn < _MAX_TURNS.value and not game_over:
            try:
                if _VERBOSE.value:
                    print(colored(f"\\nTurn {turn + 1}", "cyan"))

                # Determine current player and use THEIR proxy
                current_player = (turn % 2) + 1
                current_agent = agent1 if current_player == 1 else agent2
                current_proxy = proxy1 if current_player == 1 else proxy2
                model_name = _PLAYER1_MODEL.value if current_player == 1 else _PLAYER2_MODEL.value

                # Get current game state from this player's proxy
                state = await current_proxy.get_state()

                # Check if game is over
                if isinstance(state, dict) and state.get("game_over", False):
                    winner = state.get("winner", "Unknown")
                    print(colored(f"\\n🎉 Game Over! Winner: {winner}", "green"))
                    game_over = True
                    break

                if _VERBOSE.value:
                    print(f"Player {current_player} ({model_name.upper()}) thinking...")

                try:
                    # Get action from the agent based on current state
                    action = await current_agent.get_action_async(state, current_proxy)

                    if action:
                        # Send the action to the FreeCiv server using this player's proxy
                        result = await current_proxy.send_action(action)

                        if _VERBOSE.value:
                            print(f"Turn {turn + 1}: Player {current_player} ({model_name}) - Action: {action.action_type if hasattr(action, 'action_type') else 'unknown'}")
                            if result.get("success"):
                                print(f"  ✓ Action executed successfully")
                            else:
                                print(f"  ✗ Action failed: {result.get('error', 'unknown error')}")
                    else:
                        print(f"Turn {turn + 1}: Player {current_player} ({model_name}) - No action available")

                except Exception as e:
                    print(colored(f"Error getting/sending action: {e}", "yellow"))
                    if _VERBOSE.value:
                        import traceback
                        traceback.print_exc()

                turn += 1

            except Exception as e:
                print(colored(f"Error in turn {turn + 1}: {e}", "red"))
                if _VERBOSE.value:
                    import traceback
                    traceback.print_exc()
                break

        if turn >= _MAX_TURNS.value:
            print(colored(f"\\n⏰ Game ended after {_MAX_TURNS.value} turns (time limit)", "yellow"))

        print(colored(f"\\nGame completed. Total turns: {turn}", "blue"))

        # Disconnect both proxies
        await proxy1.disconnect()
        await proxy2.disconnect()
        print(colored("✓ Both players disconnected from FreeCiv3D", "green"))

        return True

    except Exception as e:
        print(colored(f"✗ Game failed: {e}", "red"))
        if _VERBOSE.value:
            import traceback
            traceback.print_exc()
        return False


def main(argv):
    """Main entry point."""
    if len(argv) > 1:
        print("FreeCiv LLM vs LLM Game Runner")
        print("Usage: python run_freeciv_game.py [flags]")
        return

    print(colored("FreeCiv LLM vs LLM Game Runner", "green"))
    print("=" * 40)

    # Run the async game
    success = asyncio.run(run_freeciv_game())

    if success:
        print(colored("\\n✅ Game runner completed successfully!", "green"))
    else:
        print(colored("\\n❌ Game runner failed!", "red"))
        sys.exit(1)


if __name__ == "__main__":
    app.run(main)