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

"""Diagnostic tests for FreeCiv3D civserver health and initialization.

These tests are designed to diagnose the root cause of E140/E120 authentication
failures by checking:
1. Docker container health
2. Civserver process status and timing
3. LLM Gateway connectivity
4. Network connectivity between containers
5. Port availability

Run with: python3 -m unittest tests.diagnostic.test_civserver_health
"""

import asyncio
import json
import subprocess
import time
import unittest
from typing import Optional

import websockets


class TestCivserverHealth(unittest.TestCase):
  """Diagnostic tests for civserver health and initialization timing."""

  def test_docker_container_running(self):
    """Test that FreeCiv3D Docker container is running."""
    try:
      result = subprocess.run(
          ['docker', 'ps', '--filter', 'name=fciv-net', '--format', '{{.Names}}'],
          capture_output=True,
          text=True,
          check=True,
          timeout=10
      )
      containers = result.stdout.strip().split('\n')
      self.assertIn('fciv-net', containers,
                    "FreeCiv3D container 'fciv-net' is not running")
      print(f"✓ FreeCiv3D container found: {containers}")
    except subprocess.CalledProcessError as e:
      self.fail(f"Failed to check Docker containers: {e}")
    except subprocess.TimeoutExpired:
      self.fail("Docker command timed out - Docker daemon may not be running")

  def test_civserver_process_running(self):
    """Test that civserver process is running inside container."""
    try:
      result = subprocess.run(
          ['docker', 'exec', 'fciv-net', 'pgrep', '-f', 'civserver'],
          capture_output=True,
          text=True,
          timeout=10
      )
      # pgrep returns 0 if process found, 1 if not found
      if result.returncode == 0:
        pids = result.stdout.strip().split('\n')
        print(f"✓ civserver process(es) found: PIDs {pids}")
      else:
        print("⚠️ No civserver process found - may be normal if not started")
    except subprocess.CalledProcessError as e:
      print(f"⚠️ Error checking civserver process: {e}")
    except subprocess.TimeoutExpired:
      self.fail("Docker exec command timed out")

  def test_llm_gateway_port_accessible(self):
    """Test that LLM Gateway WebSocket port is accessible."""
    try:
      # Check if port 8003 is listening
      result = subprocess.run(
          ['docker', 'exec', 'fciv-net', 'netstat', '-tlnp'],
          capture_output=True,
          text=True,
          timeout=10
      )
      output = result.stdout

      # Look for port 8003 (LLM Gateway WebSocket)
      if ':8003' in output or '0.0.0.0:8003' in output:
        print("✓ LLM Gateway port 8003 is listening")
      else:
        print("⚠️ LLM Gateway port 8003 not found in netstat output")
        print(f"Available ports:\n{output}")
    except subprocess.CalledProcessError as e:
      print(f"⚠️ Error checking ports: {e}")
    except subprocess.TimeoutExpired:
      self.fail("Docker exec command timed out")

  def test_civserver_port_accessible(self):
    """Test that civserver port 6000 is accessible."""
    try:
      result = subprocess.run(
          ['docker', 'exec', 'fciv-net', 'netstat', '-tlnp'],
          capture_output=True,
          text=True,
          timeout=10
      )
      output = result.stdout

      # Look for port 6000 (civserver default)
      if ':6000' in output or '0.0.0.0:6000' in output:
        print("✓ Civserver port 6000 is listening")
      else:
        print("⚠️ Civserver port 6000 not found - server may not be started")
        print(f"Available ports:\n{output}")
    except subprocess.CalledProcessError as e:
      print(f"⚠️ Error checking ports: {e}")
    except subprocess.TimeoutExpired:
      self.fail("Docker exec command timed out")


class TestLLMGatewayConnectivity(unittest.IsolatedAsyncioTestCase):
  """Diagnostic tests for LLM Gateway WebSocket connectivity."""

  async def test_websocket_connection_raw(self):
    """Test raw WebSocket connection to LLM Gateway."""
    # Use fciv-net hostname when running inside game-arena container
    ws_url = "ws://fciv-net:8003/ws/agent/test_diagnostic_agent"

    try:
      async with websockets.connect(ws_url, close_timeout=5) as websocket:
        print(f"✓ Successfully connected to {ws_url}")

        # Try to receive welcome message
        try:
          welcome_msg = await asyncio.wait_for(
              websocket.recv(), timeout=5.0
          )
          print(f"✓ Received welcome message: {welcome_msg[:100]}...")

          # Try to parse as JSON
          try:
            welcome_data = json.loads(welcome_msg)
            print(f"✓ Welcome message is valid JSON: type={welcome_data.get('type')}")
          except json.JSONDecodeError as e:
            print(f"⚠️ Welcome message is not JSON: {e}")

        except asyncio.TimeoutError:
          print("⚠️ No welcome message received within 5s")

    except Exception as e:
      self.fail(f"Failed to connect to LLM Gateway WebSocket: {e}")

  async def test_authentication_handshake(self):
    """Test authentication handshake with LLM Gateway."""
    ws_url = "ws://fciv-net:8003/ws/agent/test_diagnostic_agent_auth"

    try:
      async with websockets.connect(ws_url, close_timeout=5) as websocket:
        print(f"✓ Connected to {ws_url}")

        # Receive welcome
        welcome_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        print(f"✓ Received welcome: {welcome_msg[:100]}...")

        # Send authentication message
        auth_message = {
            "type": "llm_connect",
            "agent_id": "test_diagnostic_agent_auth",
            "timestamp": time.time(),
            "data": {
                "api_token": "test-token-fc3d-001",
                "model": "test-model",
                "game_id": "test_game_diagnostic",
                "capabilities": ["move", "build", "research"]
            }
        }

        await websocket.send(json.dumps(auth_message))
        print("✓ Sent authentication message")

        # Wait for auth response
        try:
          auth_response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
          print(f"✓ Received auth response: {auth_response[:200]}...")

          # Parse response
          try:
            auth_data = json.loads(auth_response)
            response_type = auth_data.get("type")

            if response_type == "error":
              error_info = auth_data.get("data", {})
              error_code = error_info.get("code")
              error_message = error_info.get("message")
              print(f"❌ Authentication failed: [{error_code}] {error_message}")
              print(f"Full error response: {json.dumps(auth_data, indent=2)}")
            elif response_type == "llm_connect":
              data_section = auth_data.get("data", {})
              if data_section.get("type") == "auth_success":
                player_id = data_section.get("player_id")
                civserver_port = data_section.get("civserver_port")
                print(f"✓ Authentication successful! player_id={player_id}, civserver_port={civserver_port}")
              else:
                print(f"⚠️ Unexpected auth response data: {data_section}")
            else:
              print(f"⚠️ Unexpected response type: {response_type}")

          except json.JSONDecodeError as e:
            print(f"❌ Auth response is not valid JSON: {e}")

        except asyncio.TimeoutError:
          print("❌ No auth response received within 10s")

    except Exception as e:
      print(f"❌ Authentication test failed: {e}")
      raise


class TestGameInitializationTiming(unittest.IsolatedAsyncioTestCase):
  """Diagnostic tests for game initialization timing and sequence."""

  async def test_single_player_connection_timing(self):
    """Measure timing for single player connection and game_ready."""
    ws_url = "ws://fciv-net:8003/ws/agent/test_timing_player1"
    game_id = f"test_timing_game_{int(time.time())}"

    connection_start = time.time()

    try:
      async with websockets.connect(ws_url, close_timeout=5) as websocket:
        connection_time = time.time() - connection_start
        print(f"✓ WebSocket connected in {connection_time:.2f}s")

        # Receive welcome
        welcome_start = time.time()
        welcome_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        welcome_time = time.time() - welcome_start
        print(f"✓ Welcome received in {welcome_time:.2f}s")

        # Send auth
        auth_start = time.time()
        auth_message = {
            "type": "llm_connect",
            "agent_id": "test_timing_player1",
            "timestamp": time.time(),
            "data": {
                "api_token": "test-token-fc3d-001",
                "model": "test-model",
                "game_id": game_id,
                "nation": "Americans",
                "leader_name": "George Washington",
                "capabilities": ["move", "build", "research"]
            }
        }
        await websocket.send(json.dumps(auth_message))

        # Wait for auth response
        auth_response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
        auth_time = time.time() - auth_start
        print(f"✓ Auth response in {auth_time:.2f}s")

        auth_data = json.loads(auth_response)
        if auth_data.get("type") == "error":
          error_info = auth_data.get("data", {})
          print(f"❌ Auth failed: [{error_info.get('code')}] {error_info.get('message')}")
          return

        # Wait for game_ready signal
        print("⏳ Waiting for game_ready signal (max 60s)...")
        game_ready_start = time.time()
        game_ready_received = False

        while time.time() - game_ready_start < 60.0:
          try:
            msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            msg_data = json.loads(msg)

            if msg_data.get("type") == "game_ready":
              game_ready_time = time.time() - game_ready_start
              print(f"✅ game_ready received in {game_ready_time:.2f}s")
              game_ready_received = True
              break
            else:
              print(f"📨 Received message type: {msg_data.get('type')}")
          except asyncio.TimeoutError:
            continue

        if not game_ready_received:
          total_wait = time.time() - game_ready_start
          print(f"❌ No game_ready signal received after {total_wait:.2f}s")
          print("⚠️ This suggests civserver is not starting or game initialization is failing")

    except Exception as e:
      print(f"❌ Timing test failed: {e}")
      raise

  async def test_concurrent_connection_timing(self):
    """Measure timing for concurrent two-player connection."""
    game_id = f"test_concurrent_{int(time.time())}"

    async def connect_player(player_num: int, nation: str, leader: str):
      """Connect a single player and measure timing."""
      ws_url = f"ws://fciv-net:8003/ws/agent/test_concurrent_p{player_num}"

      try:
        start_time = time.time()
        async with websockets.connect(ws_url, close_timeout=5) as websocket:
          # Welcome
          await asyncio.wait_for(websocket.recv(), timeout=5.0)

          # Auth
          auth_message = {
              "type": "llm_connect",
              "agent_id": f"test_concurrent_p{player_num}",
              "timestamp": time.time(),
              "data": {
                  "api_token": "test-token-fc3d-001",
                  "model": "test-model",
                  "game_id": game_id,
                  "nation": nation,
                  "leader_name": leader,
                  "capabilities": ["move", "build", "research"]
              }
          }
          await websocket.send(json.dumps(auth_message))

          # Wait for auth response
          auth_response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
          auth_time = time.time() - start_time

          auth_data = json.loads(auth_response)
          if auth_data.get("type") == "error":
            error_info = auth_data.get("data", {})
            print(f"❌ Player {player_num} auth failed: [{error_info.get('code')}] {error_info.get('message')}")
            return None, None

          print(f"✓ Player {player_num} authenticated in {auth_time:.2f}s")

          # Wait for game_ready (max 60s)
          game_ready_start = time.time()
          while time.time() - game_ready_start < 60.0:
            try:
              msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
              msg_data = json.loads(msg)

              if msg_data.get("type") == "game_ready":
                game_ready_time = time.time() - game_ready_start
                print(f"✅ Player {player_num} received game_ready in {game_ready_time:.2f}s")
                return auth_time, game_ready_time
            except asyncio.TimeoutError:
              continue

          print(f"❌ Player {player_num} no game_ready after 60s")
          return auth_time, None

      except Exception as e:
        print(f"❌ Player {player_num} connection failed: {e}")
        return None, None

    # Connect both players concurrently
    print("🔄 Testing concurrent two-player connection...")
    results = await asyncio.gather(
        connect_player(1, "Americans", "George Washington"),
        connect_player(2, "Romans", "Julius Caesar"),
        return_exceptions=True
    )

    print("\n📊 Concurrent Connection Results:")
    for i, result in enumerate(results, 1):
      if isinstance(result, Exception):
        print(f"  Player {i}: Exception - {result}")
      elif result[0] is not None:
        auth_time, ready_time = result
        print(f"  Player {i}: Auth={auth_time:.2f}s, GameReady={ready_time or 'TIMEOUT'}s")
      else:
        print(f"  Player {i}: Failed to connect")


if __name__ == '__main__':
  unittest.main()
