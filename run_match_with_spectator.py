#!/usr/bin/env python3
"""
Enhanced FreeCiv LLM match runner that displays spectator URL
"""
import re
import sys
import subprocess
import webbrowser
import time

def run_match_with_spectator(turns=10, host="fciv-net", ws_port=8003, player1="gemini", player2="openai", open_browser=True):
    """Run LLM match and extract spectator information"""

    cmd = [
        "docker", "exec", "game-arena", "python", "run_freeciv_game.py",
        f"--host={host}",
        f"--ws_port={ws_port}",
        f"--turns={turns}",
        f"--player1={player1}",
        f"--player2={player2}",
        "--verbose=true"
    ]

    print("🚀 Starting LLM vs LLM FreeCiv match...")
    print(f"⚙️  Command: {' '.join(cmd)}")

    # Run the command and capture output with timeout
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)  # 3 minute timeout

    # Print the normal output
    print(result.stdout)
    if result.stderr:
        print("Debug info:", result.stderr)

    # Enhanced game_id extraction - try multiple patterns
    game_id = None

    # Try to extract from authentication response
    game_id_match = re.search(r"'game_id': '([^']*)'", result.stderr)
    if game_id_match:
        game_id = game_id_match.group(1)

    # Try to extract from connection logs
    if not game_id:
        game_id_match = re.search(r'game_id=([a-zA-Z0-9_-]+)', result.stderr)
        if game_id_match:
            game_id = game_id_match.group(1)

    # Try to extract from LLM game session logs
    if not game_id:
        session_match = re.search(r'session[_-]([a-zA-Z0-9_-]+)', result.stderr)
        if session_match:
            game_id = f"llm_{session_match.group(1)}"

    # Fallback to default
    if not game_id:
        game_id = "default"

    # Extract agent_id with enhanced patterns
    agent_id_match = re.search(r"'agent_id': '([^']*)'", result.stderr)
    if not agent_id_match:
        agent_id_match = re.search(r'agent_([a-f0-9]+)', result.stderr)

    if agent_id_match:
        # Handle both quoted and unquoted agent IDs
        if "'" in agent_id_match.group(0):
            agent_id = agent_id_match.group(1)
        else:
            agent_id = agent_id_match.group(0)
    else:
        agent_id = "unknown_agent"

    # Extract session_id
    session_id_match = re.search(r"'session_id': '([^']*)'", result.stderr)
    session_id = session_id_match.group(1) if session_id_match else "check_gateway_logs"

    print("\n" + "="*60)
    print("🎮 GAME SESSION INFORMATION")
    print("="*60)
    print(f"Game ID: {game_id}")
    print(f"Agent ID: {agent_id}")
    print(f"Session ID: {session_id}")

    # Determine the correct spectator URL based on game type
    if game_id.startswith('llm_') or game_id == 'default':
        # LLM Gateway game - use port 8003
        spectator_url = f"http://localhost:8080/webclient/spectator.jsp?game_id={game_id}&port=8003"
        print(f"\n📺 LLM GATEWAY SPECTATOR URL:")
        print(f"Primary: {spectator_url}")
    else:
        # Traditional FreeCiv game - try multiple ports
        spectator_url = f"http://localhost:8080/webclient/spectator.jsp?game_id={game_id}&port=6000"
        print(f"\n📺 FREECIV SPECTATOR URLS:")
        print(f"Primary: {spectator_url}")
        print(f"Port 6001: http://localhost:8080/webclient/spectator.jsp?game_id={game_id}&port=6001")
        print(f"Port 6002: http://localhost:8080/webclient/spectator.jsp?game_id={game_id}&port=6002")

    print(f"\n🔧 DEBUG COMMANDS:")
    print(f"Check gateway logs: docker exec fciv-net sh -c \"grep '{game_id}' /docker/project-root/llm-gateway/logs/*.log 2>/dev/null || echo 'No matches found in log files'\"")
    print(f"Container logs: docker logs fciv-net | grep '{game_id}'")
    print(f"WebSocket test: wscat -c ws://localhost:8003/ws/spectator/{game_id}")
    print("="*60)

    # Auto-open browser if requested
    if open_browser:
        print(f"\n🌐 Opening spectator view in browser...")
        try:
            # Wait a moment for services to be ready
            time.sleep(2)
            webbrowser.open(spectator_url)
            print(f"✅ Browser opened: {spectator_url}")
        except Exception as e:
            print(f"❌ Failed to open browser: {e}")
            print(f"   Please manually open: {spectator_url}")

    return game_id

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FreeCiv LLM match with spectator info")
    parser.add_argument("--turns", type=int, default=10, help="Number of turns")
    parser.add_argument("--host", default="fciv-net", help="FreeCiv host")
    parser.add_argument("--ws_port", type=int, default=8003, help="WebSocket port")
    parser.add_argument("--player1", default="gemini", choices=["gemini", "openai", "anthropic"])
    parser.add_argument("--player2", default="openai", choices=["gemini", "openai", "anthropic"])
    parser.add_argument("--no-browser", action="store_true", help="Don't automatically open browser")

    args = parser.parse_args()

    game_id = run_match_with_spectator(
        turns=args.turns,
        host=args.host,
        ws_port=args.ws_port,
        player1=args.player1,
        player2=args.player2,
        open_browser=not args.no_browser
    )

    if game_id:
        sys.exit(0)
    else:
        sys.exit(1)