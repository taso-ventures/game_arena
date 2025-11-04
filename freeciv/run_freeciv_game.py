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
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, Optional

import webbrowser
import requests
from absl import app, flags
import termcolor

# Configure module-level logger
logger = logging.getLogger(__name__)

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
_OPEN_BROWSER = flags.DEFINE_boolean(
    "open_browser", False, "Automatically open spectator URL in browser"
)

# Game configuration constants for FreeCiv simultaneous turn model
MAX_ACTIONS_PER_TURN = 20  # Maximum actions per player per turn (safety limit)
TURN_TIMEOUT_SECONDS = 240  # Maximum time per game turn (both players) - increased for slow LLM responses
ACTION_TIMEOUT_SECONDS = 120  # Maximum time per individual action (increased from 60s for LLM latency + E101 retries)


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


async def execute_player_turn(
    player_num: int,
    agent,  # FreeCivLLMAgent
    proxy,  # FreeCivProxyClient
    game_turn: int,
    model_name: str,
    max_actions: int = MAX_ACTIONS_PER_TURN,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute one player's complete turn with multiple actions.

    FreeCiv uses simultaneous turns where both players act during the same game turn.
    Each player submits multiple actions (move units, research tech, build cities, etc.)
    and must call end_turn when done. The civserver only advances the turn when ALL
    players have ended their turn.

    Args:
        player_num: Player number (1 or 2)
        agent: FreeCivLLMAgent instance
        proxy: FreeCivProxyClient instance
        game_turn: Current game turn number
        model_name: Model name for logging
        max_actions: Maximum actions per turn (safety limit)
        verbose: Whether to print detailed logs

    Returns:
        Dictionary with turn results:
            - player: Player number
            - actions: List of actions taken
            - ended_turn: Whether end_turn was called
            - action_count: Number of actions taken
            - error: Error message if any
    """
    actions_taken = []
    error = None
    message_count = 0  # Track messages sent for diagnostics
    # Thread-safe: Each concurrent player turn gets isolated local variable
    last_state_refresh_time = 0  # Track when we last refreshed state

    try:
        if verbose:
            print(colored(f"  Player {player_num} ({model_name.upper()}) starting turn...", "blue"))

        # OPTIMIZATION: Query state ONCE at turn start instead of before every action
        # This reduces message frequency by 60-80%, preventing E101 rate limit errors
        turn_state = await asyncio.wait_for(
            proxy.get_state(),
            timeout=ACTION_TIMEOUT_SECONDS
        )
        message_count += 1  # Count initial state query
        last_state_refresh_time = time.time()  # Track refresh time
        logger.debug(f"Player {player_num} ({model_name}): Initial state query completed, message_count={message_count}")

        for action_count in range(max_actions):
            try:
                # Check if turn advanced externally (game moved on without us)
                current_turn = turn_state.get('turn', game_turn)
                if current_turn > game_turn:
                    logger.warning(f"Player {player_num}: Turn advanced to {current_turn} (server moved on)")
                    if verbose:
                        print(colored(
                            f"  Player {player_num}: Turn advanced to {current_turn} (server moved on)",
                            "yellow"
                        ))
                    break

                # Calculate action context for agent decision-making
                actions_taken_count = len(actions_taken)
                actions_remaining = max_actions - actions_taken_count - 1  # -1 for current action

                action_context = {
                    'actions_taken': max(0, actions_taken_count),  # Prevent negative
                    'actions_remaining': max(0, actions_remaining),
                    'max_actions': max_actions,
                    'should_consider_end_turn': actions_remaining <= 3 and actions_remaining > 0  # Warn when <=3 actions left
                }

                # Get action from agent using CACHED turn_state
                # No need to query state again - it hasn't changed yet
                action = await asyncio.wait_for(
                    agent.get_action_async(turn_state, proxy, action_context=action_context),
                    timeout=ACTION_TIMEOUT_SECONDS
                )

                if not action:
                    logger.debug(f"Player {player_num}: No action available")
                    if verbose:
                        print(colored(f"  Player {player_num}: No action available", "yellow"))
                    break

                # Send action to server
                result = await asyncio.wait_for(
                    proxy.send_action(action),
                    timeout=ACTION_TIMEOUT_SECONDS
                )
                message_count += 1  # Count action message
                logger.debug(f"Player {player_num}: Action sent, message_count={message_count}")

                # Brief delay after sending action to allow server processing
                await asyncio.sleep(0.5)

                # OPTIMIZATION: Only refresh state if action succeeded AND likely changed state
                # Actions like tech_research don't change state immediately (research happens over turns)
                # This reduces messages by ~60% compared to refreshing after every successful action
                if result.get('success'):
                    action_type = action.action_type if hasattr(action, 'action_type') else None

                    # List of actions that immediately change game state
                    # These actions modify units, cities, or player state and require fresh state query
                    state_changing_actions = [
                        'unit_move', 'unit_attack', 'unit_build_city', 'unit_change_homecity',
                        'unit_fortify', 'unit_sentry', 'unit_unload', 'unit_load',
                        'city_buy_production', 'city_change_production', 'city_sell_improvement',
                        'city_change_specialist', 'player_government', 'player_set_tech_goal',
                        'tech_research',  # Tech progress changes over turns, needs state refresh
                        'end_turn'  # Always refresh after end_turn
                    ]

                    # Determine if we should refresh state
                    time_since_refresh = time.time() - last_state_refresh_time
                    should_refresh = (
                        action_type in state_changing_actions or  # Action changes state
                        time_since_refresh > 10.0  # Or it's been >10s since last refresh
                    )

                    if should_refresh:
                        turn_state = await asyncio.wait_for(
                            proxy.get_state(),
                            timeout=ACTION_TIMEOUT_SECONDS
                        )
                        message_count += 1  # Count state refresh
                        last_state_refresh_time = time.time()
                        logger.debug(f"Player {player_num}: State refreshed after {action_type}, message_count={message_count}")
                else:
                    # Action failed - always refresh state as it might be stale
                    # Failed actions often indicate state mismatch (unit moved, city destroyed, etc.)
                    turn_state = await asyncio.wait_for(
                        proxy.get_state(),
                        timeout=ACTION_TIMEOUT_SECONDS
                    )
                    message_count += 1  # Count state refresh
                    last_state_refresh_time = time.time()
                    logger.debug(f"Player {player_num}: State refreshed after failed action, message_count={message_count}")
                    if verbose:
                        print(colored(
                            f"    ↻ State refreshed after failed action (preventing stale cache)",
                            "yellow"
                        ))

                # Throttle delay to avoid E101 rate limiting from FreeCiv3D proxy
                # Increased from 1.5s to 3.0s to provide more breathing room between messages
                # With 3s delay, max rate is ~20 actions/min = ~40 messages/min (well under 200 msg/min limit)
                await asyncio.sleep(3.0)

                # Record action
                action_type = action.action_type if hasattr(action, 'action_type') else 'unknown'
                actions_taken.append({
                    'action': action,
                    'action_type': action_type,
                    'result': result,
                    'action_number': action_count + 1,
                })

                # Log action
                if verbose:
                    success_mark = "✓" if result.get("success") else "✗"
                    print(colored(
                        f"    {success_mark} Action {action_count + 1}: {action_type}",
                        "green" if result.get("success") else "red"
                    ))

                # Check if this was end_turn
                if action_type == 'end_turn':
                    if verbose:
                        print(colored(
                            f"  Player {player_num} ended turn after {len(actions_taken)} actions",
                            "green"
                        ))
                    break

            except asyncio.TimeoutError:
                error = f"Action {action_count + 1} timed out"
                logger.error(f"Player {player_num}: {error}")
                if verbose:
                    print(colored(f"  Player {player_num}: {error}", "red"))
                break
            except Exception as e:
                error = f"Action {action_count + 1} failed: {e}"
                error_str = str(e)

                # Check for rate limit errors (E101) - graceful handling
                if "E101" in error_str:
                    logger.warning(f"Player {player_num}: Rate limit exceeded (E101), slowing down")
                    if verbose:
                        print(colored(
                            f"⚠️ Player {player_num}: Rate limit exceeded (E101), slowing down...",
                            "yellow"
                        ))
                    # Wait longer before retrying
                    await asyncio.sleep(5.0)
                    # Don't break - continue with next action attempt
                    continue

                # Check for connection lost (E123) - attempt automatic reconnection
                if "E123" in error_str:
                    logger.warning(f"Player {player_num}: Connection lost (E123), attempting session resumption")
                    if verbose:
                        print(colored(
                            f"🔌 Player {player_num}: Connection lost (E123), attempting session resumption...",
                            "yellow"
                        ))

                    # CRITICAL: Set disconnect time for session resumption tracking
                    # E123 is detected via exception (not clean disconnect), so we must set this manually
                    # This allows reconnect_with_session() to verify we're within the 60s window
                    proxy.connection_manager.last_disconnect_time = time.time()

                    # Attempt session resumption within 60s window
                    try:
                        reconnected = await asyncio.wait_for(
                            proxy.connection_manager.reconnect_with_session(),
                            timeout=10.0
                        )

                        if reconnected:
                            logger.info(f"Player {player_num}: Session resumed successfully")
                            if verbose:
                                print(colored(
                                    f"✅ Player {player_num}: Session resumed! Retrying action...",
                                    "green"
                                ))
                            # Brief delay before retrying action
                            await asyncio.sleep(2.0)
                            # Don't break - continue with retry
                            continue
                        else:
                            logger.error(f"Player {player_num}: Session resumption failed")
                            if verbose:
                                print(colored(
                                    f"❌ Player {player_num}: Session resumption failed",
                                    "red"
                                ))
                            # Mark as server terminated since reconnection failed
                            error = f"SERVER_TERMINATED: {error}"
                            break
                    except asyncio.TimeoutError:
                        logger.error(f"Player {player_num}: Reconnection timed out")
                        if verbose:
                            print(colored(
                                f"⏱️ Player {player_num}: Reconnection timed out",
                                "red"
                            ))
                        error = f"SERVER_TERMINATED: {error}"
                        break
                    except Exception as reconnect_error:
                        logger.error(f"Player {player_num}: Reconnection error: {reconnect_error}")
                        if verbose:
                            print(colored(
                                f"❌ Player {player_num}: Reconnection error: {reconnect_error}",
                                "red"
                            ))
                        error = f"SERVER_TERMINATED: {error}"
                        break

                # Check for other game server termination errors
                is_server_terminated = (
                    "UNKNOWN" in error_str or  # Unknown server error
                    "E140" in error_str  # Failed to connect to game server
                )

                if is_server_terminated:
                    logger.error(f"Player {player_num}: Game server terminated - {error}")
                    if verbose:
                        print(colored(
                            f"🛑 Player {player_num}: Game server terminated - {error}",
                            "red"
                        ))
                    # Signal server termination by setting special error marker
                    error = f"SERVER_TERMINATED: {error}"
                else:
                    logger.error(f"Player {player_num}: {error}")
                    if verbose:
                        print(colored(f"  Player {player_num}: {error}", "red"))

                if verbose:
                    import traceback
                    traceback.print_exc()
                break

        # Check if we hit max actions without ending turn
        if len(actions_taken) >= max_actions:
            last_action_type = actions_taken[-1]['action_type'] if actions_taken else None
            if last_action_type != 'end_turn':
                logger.warning(f"Player {player_num} hit max actions ({max_actions}) without calling end_turn")
                if verbose:
                    print(colored(
                        f"  ⚠️ Player {player_num} hit max actions ({max_actions}) without calling end_turn",
                        "yellow"
                    ))

    except Exception as e:
        error = f"Turn execution failed: {e}"
        logger.error(f"Player {player_num}: {error}")
        if verbose:
            print(colored(f"  Player {player_num}: {error}", "red"))
            import traceback
            traceback.print_exc()

    return {
        'player': player_num,
        'actions': actions_taken,
        'ended_turn': (
            actions_taken[-1]['action_type'] == 'end_turn'
            if actions_taken else False
        ),
        'action_count': len(actions_taken),
        'message_count': message_count,  # Include message count for diagnostics
        'error': error,
    }


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
        logger.info(f"Connecting Player 1 to LLM Gateway at {_FREECIV_HOST.value}:{_FREECIV_WS_PORT.value}")
        print(colored(f"[{time.time():.1f}] Connecting Player 1 to LLM Gateway...", "blue"))
        player1_success = await proxy1.connect()
        player1_auth_time = time.time() - auth_start_time
        logger.debug(f"Player 1 connection attempt completed in {player1_auth_time:.1f}s, success={player1_success}")

        if not player1_success:
            logger.error("Player 1 authentication failed")
            print(colored(f"[{time.time():.1f}] ✗ Player 1 authentication failed", "red"))
            print(colored("   → Check if FreeCiv civserver is running: docker ps | grep fciv", "yellow"))
            print(colored("   → Check logs: docker logs fciv-net | grep -E '(civserver|E140)'", "yellow"))
            await proxy1.disconnect()
            await proxy2.disconnect()
            return False

        logger.info(f"Player 1 connected successfully in {player1_auth_time:.1f}s")
        print(colored(f"[{time.time():.1f}] ✓ Player 1 connected (took {player1_auth_time:.1f}s)", "green"))

        # CRITICAL: Add delay after Player 1 connects to allow civserver game initialization
        # FreeCiv3D proxy needs time to:
        # 1. Establish CivCom connection to civserver
        # 2. Execute /take command to control AI player
        # 3. Process nation selection
        # 4. Initialize game session
        # Without this delay, Player 2 connection may fail with E140 (civserver busy)
        initialization_delay = 3.0  # 3 seconds to allow game session setup
        logger.debug(f"Waiting {initialization_delay}s for game session initialization")
        print(colored(f"[{time.time():.1f}] ⏳ Waiting {initialization_delay}s for game session initialization...", "yellow"))
        await asyncio.sleep(initialization_delay)

        # Connect Player 2
        player2_start = time.time()
        logger.info("Connecting Player 2 to LLM Gateway")
        print(colored(f"[{time.time():.1f}] Connecting Player 2 to LLM Gateway...", "blue"))
        player2_success = await proxy2.connect()
        player2_auth_time = time.time() - player2_start
        logger.debug(f"Player 2 connection attempt completed in {player2_auth_time:.1f}s, success={player2_success}")

        if not player2_success:
            logger.error("Player 2 authentication failed")
            print(colored(f"[{time.time():.1f}] ✗ Player 2 authentication failed", "red"))
            print(colored("   → Check if FreeCiv civserver slots are available", "yellow"))
            print(colored("   → May need to restart FreeCiv3D: docker-compose restart", "yellow"))
            await proxy1.disconnect()
            await proxy2.disconnect()
            return False

        logger.info(f"Player 2 connected successfully in {player2_auth_time:.1f}s")
        print(colored(f"[{time.time():.1f}] ✓ Player 2 connected (took {player2_auth_time:.1f}s)", "green"))

        # Display observer URL early (before waiting for game_ready)
        civserver_port = proxy1.civserver_port or 6000
        observer_url = (
            f"http://localhost:8080/webclient/"
            f"?action=observe"
            f"&renderer=webgl"
            f"&civserverport={civserver_port}"
            f"&civserverhost=localhost"
            f"&multi=true"
            f"&type=multiplayer"
        )

        print("\n" + "=" * 80, flush=True)
        print(colored("🎮 FREECIV3D GAME INITIALIZED", "green", attrs=['bold']), flush=True)
        print("=" * 80, flush=True)
        print(colored(f"\n📺 OBSERVER MODE:", "cyan", attrs=['bold']), flush=True)
        print(colored(f"   Game ID: {game_id}", "yellow"), flush=True)
        print(colored(f"   Observer URL: {observer_url}", "green", attrs=['bold']), flush=True)
        print(colored(f"\n   👉 Open this URL in your browser to watch the game!", "white", attrs=['bold']), flush=True)
        print("=" * 80 + "\n", flush=True)

        # Wait for game_ready signal from server (event-driven with timeout)
        # FreeCiv3D will send game_ready when:
        # - All players have connected and authenticated
        # - Nations have been assigned
        # - Units and cities have been created
        # - Game is ready for actions
        logger.info("Waiting for game_ready signal from server (max 45s)")
        print(colored(f"[{time.time():.1f}] ⏳ Waiting for game_ready signal from server (max 45s)...", "yellow"))

        # Increased timeout to 45s for slow Docker environments
        max_wait = 45.0
        try:
            await asyncio.wait_for(proxy1.game_ready_event.wait(), timeout=max_wait)
            game_ready_time = time.time() - auth_start_time
            logger.info(f"Game ready signal received after {game_ready_time:.1f}s")
            print(colored(f"[{time.time():.1f}] ✅ Game ready signal received! (total time: {game_ready_time:.1f}s)", "green"))
        except asyncio.TimeoutError:
            timeout_time = time.time() - auth_start_time
            logger.warning(f"Timeout after {timeout_time:.1f}s waiting for game_ready signal")
            print(colored(f"[{time.time():.1f}] ⚠️ Timeout after {timeout_time:.1f}s waiting for game_ready", "yellow"))

            # CRITICAL: Check if event was set during race condition
            if proxy1.game_ready_event.is_set():
                logger.info("Game ready event was set (caught race condition)")
                print(colored("   ✅ Game ready event WAS set (caught race condition!)", "green"))
            else:
                logger.warning("No game_ready signal received, proceeding with state validation")
                print(colored("   ❌ No game_ready signal - game may not be initialized", "yellow"))
                print(colored("   Proceeding with state validation...", "yellow"))

        # Verify game state is properly initialized before starting game loop
        logger.info("Verifying game initialization")
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

            logger.debug(f"Player 1 state: {len(players1)} players, {len(units1)} units, {len(cities1)} cities")
            logger.debug(f"Turn: {state1.get('turn', 'unknown')}, Phase: {state1.get('game', {}).get('phase', 'unknown')}")
            print(colored(f"Player 1 state: {len(players1)} players, {len(units1)} units, {len(cities1)} cities", "blue"))
            print(colored(f"Turn: {state1.get('turn', 'unknown')}, Phase: {state1.get('game', {}).get('phase', 'unknown')}", "blue"))

            # DIAGNOSTIC: Check legal_actions availability
            legal_actions = state1.get("legal_actions", [])
            if legal_actions:
                action_types = set(a.get("type") for a in legal_actions if isinstance(a, dict))
                logger.debug(f"Legal actions: {len(legal_actions)} actions, types: {action_types}")
                print(colored(f"Legal actions: {len(legal_actions)} actions", "blue"))
                print(colored(f"Action types available: {action_types}", "blue"))

                # Warn if only tech_research available
                if len(action_types) == 1 and "tech_research" in action_types and units1:
                    logger.warning("Only tech_research actions available despite having units")
                    print(colored("⚠️ WARNING: Only tech_research actions available despite having units!", "yellow"))
                    print(colored("   This may indicate FreeCiv3D gateway is not generating unit actions.", "yellow"))
                    print(colored("   Expected: unit_move, unit_build_city, etc.", "yellow"))
            else:
                logger.warning("No legal_actions in state at initialization")
                print(colored("⚠️ No legal_actions in state at initialization!", "yellow"))

            # Verify nation assignment
            if players1:
                for i, player in enumerate(players1[:2]):  # Check first 2 players
                    player_nation = player.get("nation", "unassigned")
                    player_name = player.get("name", f"Player {i+1}")
                    logger.debug(f"{player_name}: nation={player_nation}")
                    print(colored(f"  {player_name}: nation={player_nation}", "blue"))

                    if player_nation == "unassigned" or not player_nation:
                        logger.warning(f"Nation not assigned to {player_name}")
                        print(colored(f"    ⚠️ Nation not assigned to {player_name}!", "yellow"))

            # Check game_ready flag
            if proxy1.game_ready:
                logger.info("Game ready signal confirmed")
                print(colored("✅ Game ready signal confirmed", "green"))
            else:
                logger.warning("No game_ready signal received yet")
                print(colored("⚠️ No game_ready signal received yet", "yellow"))

            # If game looks uninitialized, wait longer
            if not units1 and not cities1:
                logger.warning("Game not fully initialized (no units/cities), waiting additional 10s")
                print(colored("⚠️ Game not fully initialized (no units/cities), waiting additional 10s...", "yellow"))
                await asyncio.sleep(10)

                # Re-check after additional wait
                state1 = await proxy1.get_state()
                units1 = state1.get("units", [])
                cities1 = state1.get("cities", [])
                logger.debug(f"After additional wait: {len(units1)} units, {len(cities1)} cities")
                print(colored(f"After additional wait: {len(units1)} units, {len(cities1)} cities", "blue"))

                if not units1 and not cities1:
                    logger.error("Game still appears uninitialized after additional wait")
                    print(colored("❌ WARNING: Game still appears uninitialized! Proceeding anyway...", "yellow"))
                    print(colored("   This may result in only tech_research actions being available.", "yellow"))

        except Exception as e:
            logger.error(f"Error verifying game state: {e}")
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
        logger.info(f"Initialization complete: Player 1={player1_auth_time:.1f}s, Player 2={player2_auth_time:.1f}s, Total={total_init_time:.1f}s")
        print(colored(f"\n📊 Initialization Timing Summary:", "cyan"))
        print(colored(f"  Player 1 auth: {player1_auth_time:.1f}s", "blue"))
        print(colored(f"  Player 2 auth: {player2_auth_time:.1f}s", "blue"))
        print(colored(f"  Total initialization: {total_init_time:.1f}s", "blue"))
        if proxy1.game_ready:
            logger.info("Game ready signal: RECEIVED")
            print(colored(f"  ✅ Game ready signal: RECEIVED", "green"))
        else:
            logger.warning("Game ready signal: NOT RECEIVED")
            print(colored(f"  ⚠️ Game ready signal: NOT RECEIVED", "yellow"))

        # Display observer URLs (with game readiness status)
        print(colored("\n" + "=" * 60, "cyan"))
        print(colored("📺 OBSERVER MODE", "cyan"))
        print(colored("=" * 60, "cyan"))
        print(f"Game ID: {game_id}")

        # Display game readiness status
        if units1 and cities1 and proxy1.game_ready:
            print(colored("✅ Game is fully initialized and ready for viewing", "green"))
        elif units1 or cities1:
            print(colored("⚠️ Game partially initialized (observer may show incomplete state)", "yellow"))
        else:
            print(colored("❌ WARNING: Game not initialized yet (observer may show pre-game lobby)", "yellow"))
            print(colored("   Wait a few moments and refresh the observer page", "yellow"))

        # FreeCiv WebGL Observer URL (direct civserver connection)
        civserver_port = proxy1.civserver_port or 6000
        observer_url = (
            f"http://localhost:8080/webclient/"
            f"?action=observe"
            f"&renderer=webgl"
            f"&civserverport={civserver_port}"
            f"&civserverhost=localhost"
            f"&multi=true"
            f"&type=multiplayer"
        )

        print(f"\nObserver URL (WebGL Direct Connection):")
        print(colored(f"  {observer_url}", "green"))
        print(f"\n📝 Note: Observer connects directly to civserver (port {civserver_port}) via WebGL")
        print(f"   Renders full game state from civserver, not gateway broadcast")
        print(f"   Supports multiplayer observation with real-time updates")
        print(f"\n🔧 Debug Commands:")
        print(f"  Check civserver logs: docker exec fciv-net tail -f /docker/civserver-logs/civserver-{civserver_port}.log")
        print(f"  Check LLM Gateway logs: docker exec fciv-net grep '{game_id}' /docker/llm-gateway/logs/*.log 2>/dev/null")
        print(f"  Check proxy logs: docker exec fciv-net cat /docker/logs/freeciv-proxy-8002.log | grep '{game_id}'")
        print(colored("=" * 60 + "\n", "cyan"))

        # Auto-open browser if requested
        if _OPEN_BROWSER.value:
            print(colored("🌐 Opening observer view in browser...", "cyan"))
            try:
                # Wait a moment for services to be ready
                await asyncio.sleep(2)
                webbrowser.open(observer_url)
                print(colored(f"✅ Browser opened: {observer_url}", "green"))
            except Exception as e:
                print(colored(f"❌ Failed to open browser: {e}", "red"))
                print(colored(f"   Please manually open: {observer_url}", "yellow"))

        # Game loop using FreeCiv's simultaneous turn model
        # In FreeCiv, both players act during the same game turn and must call
        # end_turn when finished. The civserver only advances the turn when ALL
        # players have ended their turn.
        print(colored(f"\nStarting FreeCiv LLM vs LLM game (max {_MAX_TURNS.value} turns)...", "green"))
        print(colored("Using simultaneous turn model: both players act concurrently per turn", "blue"))
        print("=" * 60)

        game_turn = 1
        game_over = False
        turns_completed = 0

        while game_turn <= _MAX_TURNS.value and not game_over:
            try:
                # Display turn header
                print(colored(f"\n{'=' * 60}", "cyan"))
                print(colored(f"GAME TURN {game_turn}", "cyan", attrs=['bold']))
                print(colored(f"{'=' * 60}", "cyan"))

                # Check for game over before executing turn
                try:
                    state = await proxy1.get_state()
                    if isinstance(state, dict) and state.get("game_over", False):
                        winner = state.get("winner", "Unknown")
                        logger.info(f"Game over detected, winner: {winner}")
                        print(colored(f"\n🎉 Game Over! Winner: {winner}", "green"))
                        game_over = True
                        break
                except Exception as e:
                    logger.warning(f"Could not check game state: {e}")
                    print(colored(f"⚠️ Warning: Could not check game state: {e}", "yellow"))

                # Execute both players' turns concurrently
                # Each player will submit multiple actions until they call end_turn
                turn_start_time = time.time()

                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(
                            execute_player_turn(
                                player_num=1,
                                agent=agent1,
                                proxy=proxy1,
                                game_turn=game_turn,
                                model_name=_PLAYER1_MODEL.value,
                                max_actions=MAX_ACTIONS_PER_TURN,
                                verbose=_VERBOSE.value,
                            ),
                            execute_player_turn(
                                player_num=2,
                                agent=agent2,
                                proxy=proxy2,
                                game_turn=game_turn,
                                model_name=_PLAYER2_MODEL.value,
                                max_actions=MAX_ACTIONS_PER_TURN,
                                verbose=_VERBOSE.value,
                            ),
                            return_exceptions=True
                        ),
                        timeout=TURN_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Turn {game_turn} timed out after {TURN_TIMEOUT_SECONDS}s")
                    print(colored(
                        f"⚠️ Turn {game_turn} timed out after {TURN_TIMEOUT_SECONDS}s",
                        "red"
                    ))
                    break

                turn_duration = time.time() - turn_start_time

                # Process results
                player1_result, player2_result = results

                # Check for exceptions
                if isinstance(player1_result, Exception):
                    logger.error(f"Player 1 error ({type(player1_result).__name__}): {player1_result}")
                    print(colored(
                        f"❌ Player 1 error ({type(player1_result).__name__}): {player1_result}",
                        "red"
                    ))
                if isinstance(player2_result, Exception):
                    logger.error(f"Player 2 error ({type(player2_result).__name__}): {player2_result}")
                    print(colored(
                        f"❌ Player 2 error ({type(player2_result).__name__}): {player2_result}",
                        "red"
                    ))

                # Check for game server termination in either player's error
                server_terminated = False
                for result in [player1_result, player2_result]:
                    if isinstance(result, dict) and result.get('error'):
                        if 'SERVER_TERMINATED' in result['error']:
                            server_terminated = True
                            break

                if server_terminated:
                    logger.error(f"Game server terminated during Turn {game_turn}")
                    print(colored(
                        f"\n🛑 Game server terminated during Turn {game_turn} - ending game early",
                        "red",
                        attrs=['bold']
                    ))
                    print(colored(
                        "   The FreeCiv civserver stopped responding. This usually happens after ~2 minutes of gameplay.",
                        "yellow"
                    ))
                    game_over = True
                    break

                # Log turn summary
                if not isinstance(player1_result, Exception) and not isinstance(player2_result, Exception):
                    logger.info(f"Turn {game_turn} completed: P1={player1_result['action_count']} actions, "
                               f"P2={player2_result['action_count']} actions, duration={turn_duration:.1f}s")
                    print(colored(f"\n📊 Turn {game_turn} Summary:", "cyan"))
                    print(f"  Player 1: {player1_result['action_count']} actions, "
                          f"ended_turn: {player1_result['ended_turn']}, "
                          f"messages: {player1_result.get('message_count', 'N/A')}")
                    print(f"  Player 2: {player2_result['action_count']} actions, "
                          f"ended_turn: {player2_result['ended_turn']}, "
                          f"messages: {player2_result.get('message_count', 'N/A')}")
                    print(f"  Duration: {turn_duration:.1f}s")

                    # Calculate total messages for the turn
                    total_messages = player1_result.get('message_count', 0) + player2_result.get('message_count', 0)
                    if total_messages > 0:
                        # FreeCiv3D configuration: MAX_MESSAGES_PER_TURN=24 (recommended for 2-player games)
                        # Gateway burst limit is 40 msg/s, but staying under 24 is safer
                        message_status = "🟢 OK" if total_messages <= 24 else "🔴 HIGH"
                        logger.debug(f"Turn {game_turn} message count: {total_messages}/24 per turn")
                        print(colored(
                            f"  📨 Total messages: {total_messages}/24 per turn - {message_status}",
                            "green" if total_messages <= 24 else "yellow"
                        ))
                        if total_messages > 24:
                            logger.warning(f"Turn {game_turn} exceeded recommended message limit: {total_messages}/24")
                            print(colored(
                                f"      ⚠️ Exceeded recommended limit! May trigger E429 rate warnings",
                                "yellow"
                            ))

                    # Check if both players ended their turn
                    if not player1_result['ended_turn']:
                        logger.warning(f"Turn {game_turn}: Player 1 did not call end_turn")
                        print(colored("  ⚠️ Player 1 did not call end_turn", "yellow"))
                    if not player2_result['ended_turn']:
                        logger.warning(f"Turn {game_turn}: Player 2 did not call end_turn")
                        print(colored("  ⚠️ Player 2 did not call end_turn", "yellow"))

                # Verify turn advanced in the game server
                try:
                    state = await proxy1.get_state()
                    new_turn = state.get('turn', game_turn)

                    if new_turn > game_turn:
                        logger.info(f"Turn advanced: {game_turn} → {new_turn}")
                        print(colored(
                            f"✓ Turn advanced: {game_turn} → {new_turn}",
                            "green",
                            attrs=['bold']
                        ))
                        game_turn = new_turn
                        turns_completed += 1
                    elif new_turn == game_turn:
                        logger.warning(f"Turn did not advance (still at turn {game_turn})")
                        print(colored(
                            f"⚠️ Turn did not advance (still at turn {game_turn})",
                            "yellow"
                        ))
                        print(colored(
                            "   This may indicate players did not call end_turn, or server issue",
                            "yellow"
                        ))
                        # Still increment to avoid infinite loop
                        game_turn += 1
                    else:
                        logger.warning(f"Unexpected turn value: {new_turn} (expected >= {game_turn})")
                        print(colored(
                            f"⚠️ Unexpected turn value: {new_turn} (expected >= {game_turn})",
                            "yellow"
                        ))
                        game_turn = new_turn

                except Exception as e:
                    logger.error(f"Could not verify turn advancement: {e}")
                    print(colored(f"⚠️ Could not verify turn advancement: {e}", "yellow"))
                    # Assume turn advanced to avoid infinite loop
                    game_turn += 1

            except Exception as e:
                logger.error(f"Error in game turn {game_turn}: {e}")
                print(colored(f"❌ Error in game turn {game_turn}: {e}", "red"))
                if _VERBOSE.value:
                    import traceback
                    traceback.print_exc()
                break

        # Game end summary
        logger.info(f"Game ended: turns_completed={turns_completed}, final_turn={game_turn}")
        print(colored(f"\n{'=' * 60}", "cyan"))
        if game_turn > _MAX_TURNS.value:
            logger.info(f"Game ended after {_MAX_TURNS.value} turns (max limit)")
            print(colored(f"⏰ Game ended after {_MAX_TURNS.value} turns (max limit)", "yellow"))

        print(colored(f"Game completed. Turns played: {turns_completed}", "blue"))
        print(colored(f"Final turn number: {game_turn}", "blue"))

        # Disconnect both proxies
        await proxy1.disconnect()
        await proxy2.disconnect()
        logger.info("Both players disconnected from FreeCiv3D")
        print(colored("✓ Both players disconnected from FreeCiv3D", "green"))

        return True

    except Exception as e:
        logger.error(f"Game failed: {e}")
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