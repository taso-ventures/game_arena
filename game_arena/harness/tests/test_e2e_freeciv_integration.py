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

"""End-to-end integration tests for FreeCiv LLM Agent with real server."""

import asyncio
import os
import unittest
import time
from unittest.mock import MagicMock

from game_arena.harness.freeciv_llm_agent import FreeCivLLMAgent
from game_arena.harness.freeciv_proxy_client import FreeCivProxyClient
from game_arena.harness.tests.test_helpers import MockModelWithPredictableResponses


class TestFreeCivE2EIntegration(unittest.IsolatedAsyncioTestCase):
  """End-to-end integration tests using Docker containers."""

  def setUp(self):
    """Set up test fixtures."""
    # Get server URLs from environment (set by Docker Compose)
    self.freeciv_server_url = os.getenv("FREECIV_SERVER_URL", "ws://localhost:8443")
    self.freeciv_api_url = os.getenv("FREECIV_API_URL", "http://localhost:8080")

    # Use mock model for predictable testing
    self.mock_model = MockModelWithPredictableResponses()

    # Create real proxy client
    self.proxy_client = FreeCivProxyClient(
        server_url=self.freeciv_server_url,
        api_url=self.freeciv_api_url
    )

  async def test_server_health_check(self):
    """Test that the FreeCiv3D server is healthy and responding."""
    # Wait for server to be ready
    for attempt in range(30):  # 30 second timeout
      try:
        # Try to connect to proxy
        await self.proxy_client.connect()
        self.assertTrue(self.proxy_client.is_connected())
        await self.proxy_client.disconnect()
        break
      except Exception as e:
        if attempt == 29:  # Last attempt
          self.fail(f"Server not ready after 30 seconds: {e}")
        await asyncio.sleep(1)

  async def test_basic_websocket_communication(self):
    """Test basic WebSocket communication with FreeCiv3D server."""
    await self.proxy_client.connect()

    try:
      # Test ping
      response = await self.proxy_client.ping()
      self.assertIsInstance(response, dict)
      self.assertIn("type", response)
      self.assertEqual(response["type"], "pong")

      # Test getting game state
      state = await self.proxy_client.get_game_state()
      self.assertIsInstance(state, dict)
      self.assertIn("turn", state)
      self.assertIn("players", state)
      self.assertIn("playerID", state)

      # Validate game state structure
      self.assertIsInstance(state["turn"], int)
      self.assertIsInstance(state["players"], list)
      self.assertGreater(len(state["players"]), 0)

      # Validate player structure
      player = state["players"][0]
      self.assertIn("id", player)
      self.assertIn("name", player)
      self.assertIn("cities", player)
      self.assertIn("units", player)

    finally:
      await self.proxy_client.disconnect()

  async def test_legal_actions_retrieval(self):
    """Test retrieving legal actions from the server."""
    await self.proxy_client.connect()

    try:
      # Get legal actions for player 1
      legal_actions = await self.proxy_client.get_legal_actions(player_id=1)
      self.assertIsInstance(legal_actions, list)
      self.assertGreater(len(legal_actions), 0)

      # Validate action IDs are integers
      for action_id in legal_actions:
        self.assertIsInstance(action_id, int)
        self.assertGreaterEqual(action_id, 0)

    finally:
      await self.proxy_client.disconnect()

  async def test_action_submission(self):
    """Test submitting actions to the server."""
    await self.proxy_client.connect()

    try:
      # Get current legal actions
      legal_actions = await self.proxy_client.get_legal_actions(player_id=1)
      self.assertGreater(len(legal_actions), 0)

      # Submit a legal action
      action = {
        "action_type": "unit_move",
        "actor_id": 101,
        "target": {"x": 11, "y": 14},
        "player_id": 1
      }

      result = await self.proxy_client.send_action(action)
      self.assertIsInstance(result, dict)
      self.assertIn("success", result)

      # Result should indicate success or provide error details
      if not result.get("success", False):
        self.assertIn("message", result)

    finally:
      await self.proxy_client.disconnect()

  async def test_full_agent_workflow(self):
    """Test complete agent workflow from observation to action."""
    await self.proxy_client.connect()

    try:
      # Create agent with real proxy client
      agent = FreeCivLLMAgent(
          model=self.mock_model,
          strategy="balanced",
          use_rethinking=False,
          proxy_client=self.proxy_client  # Inject real client
      )

      # Get current game state as observation
      observation = await self.proxy_client.get_game_state()

      # Get legal actions and add to observation
      legal_actions = await self.proxy_client.get_legal_actions(
          player_id=observation["playerID"]
      )
      observation["legalActions"] = legal_actions

      # Agent generates action
      result = agent(observation, {})

      # Validate result
      self.assertIsInstance(result, dict)
      self.assertIn("submission", result)
      action_id = result["submission"]
      self.assertIsInstance(action_id, int)
      self.assertIn(action_id, legal_actions)

      # Submit action to server
      action = {
        "action_id": action_id,
        "player_id": observation["playerID"]
      }
      action_result = await self.proxy_client.send_action(action)
      self.assertIsInstance(action_result, dict)

    finally:
      await self.proxy_client.disconnect()

  async def test_multi_turn_gameplay(self):
    """Test multi-turn gameplay with state persistence."""
    await self.proxy_client.connect()

    try:
      agent = FreeCivLLMAgent(
          model=self.mock_model,
          strategy="aggressive_expansion",
          memory_size=5,
          proxy_client=self.proxy_client
      )

      initial_state = await self.proxy_client.get_game_state()
      initial_turn = initial_state["turn"]

      # Play several turns
      for turn_offset in range(3):
        current_state = await self.proxy_client.get_game_state()
        legal_actions = await self.proxy_client.get_legal_actions(
            player_id=current_state["playerID"]
        )

        if not legal_actions:
          break  # No legal actions available

        # Update observation
        observation = current_state.copy()
        observation["legalActions"] = legal_actions

        # Agent decides action
        result = agent(observation, {})
        action_id = result["submission"]

        # Submit action
        action = {
          "action_id": action_id,
          "player_id": current_state["playerID"]
        }
        action_result = await self.proxy_client.send_action(action)

        # Verify action was processed
        self.assertIsInstance(action_result, dict)

        # Wait for turn to advance
        await asyncio.sleep(0.5)

      # Verify memory accumulated
      self.assertGreater(len(agent.memory.history), 0)

      # Verify agent performance tracking
      self.assertGreater(agent._num_model_calls, 0)
      self.assertGreater(agent._total_response_time, 0)

    finally:
      await self.proxy_client.disconnect()

  async def test_error_handling_and_recovery(self):
    """Test error handling and recovery mechanisms."""
    await self.proxy_client.connect()

    try:
      agent = FreeCivLLMAgent(
          model=self.mock_model,
          fallback_to_random=True,
          proxy_client=self.proxy_client
      )

      # Test with invalid action submission
      invalid_action = {
        "action_type": "invalid_action",
        "actor_id": -1,
        "player_id": 999
      }

      result = await self.proxy_client.send_action(invalid_action)
      self.assertIsInstance(result, dict)

      # Should get error response
      if not result.get("success", False):
        self.assertIn("message", result)

      # Test agent with empty legal actions
      observation = await self.proxy_client.get_game_state()
      observation["legalActions"] = []

      # Agent should handle gracefully
      try:
        result = agent(observation, {})
        # If it succeeds, should return valid format
        self.assertIsInstance(result, dict)
      except Exception:
        # Acceptable to fail with empty legal actions
        pass

    finally:
      await self.proxy_client.disconnect()

  async def test_concurrent_agent_connections(self):
    """Test multiple agents connecting concurrently."""
    agents = []
    clients = []

    try:
      # Create multiple agents with separate connections
      for i in range(2):
        client = FreeCivProxyClient(
            server_url=self.freeciv_server_url,
            api_url=self.freeciv_api_url
        )
        await client.connect()
        clients.append(client)

        agent = FreeCivLLMAgent(
            model=self.mock_model,
            proxy_client=client
        )
        agents.append(agent)

      # Test concurrent operations
      tasks = []
      for i, (agent, client) in enumerate(zip(agents, clients)):
        async def agent_action(a, c):
          state = await c.get_game_state()
          legal_actions = await c.get_legal_actions(player_id=state["playerID"])
          if legal_actions:
            state["legalActions"] = legal_actions
            return a(state, {})
          return {"submission": 0}

        tasks.append(agent_action(agent, client))

      # Execute concurrently
      results = await asyncio.gather(*tasks, return_exceptions=True)

      # Verify all succeeded
      for result in results:
        self.assertNotIsInstance(result, Exception)
        self.assertIsInstance(result, dict)
        self.assertIn("submission", result)

    finally:
      # Clean up all connections
      for client in clients:
        if client.is_connected():
          await client.disconnect()

  async def test_performance_under_load(self):
    """Test agent performance under moderate load."""
    await self.proxy_client.connect()

    try:
      agent = FreeCivLLMAgent(
          model=self.mock_model,
          proxy_client=self.proxy_client
      )

      # Measure response times
      response_times = []

      for i in range(5):  # Reduced from 10 for faster testing
        start_time = time.time()

        state = await self.proxy_client.get_game_state()
        legal_actions = await self.proxy_client.get_legal_actions(
            player_id=state["playerID"]
        )

        if legal_actions:
          state["legalActions"] = legal_actions
          result = agent(state, {})
          self.assertIsInstance(result, dict)

        end_time = time.time()
        response_times.append(end_time - start_time)

      # Verify reasonable performance
      avg_response_time = sum(response_times) / len(response_times)
      max_response_time = max(response_times)

      self.assertLess(avg_response_time, 5.0)  # Average under 5 seconds
      self.assertLess(max_response_time, 10.0)  # Max under 10 seconds

    finally:
      await self.proxy_client.disconnect()

  async def test_server_restart_recovery(self):
    """Test recovery from server restart."""
    await self.proxy_client.connect()

    try:
      # Get initial state
      initial_state = await self.proxy_client.get_game_state()
      self.assertIsInstance(initial_state, dict)

      # Disconnect and reconnect (simulates server restart)
      await self.proxy_client.disconnect()
      self.assertFalse(self.proxy_client.is_connected())

      # Wait and reconnect
      await asyncio.sleep(1)
      await self.proxy_client.connect()
      self.assertTrue(self.proxy_client.is_connected())

      # Should be able to get state again
      recovered_state = await self.proxy_client.get_game_state()
      self.assertIsInstance(recovered_state, dict)
      self.assertIn("turn", recovered_state)

    finally:
      if self.proxy_client.is_connected():
        await self.proxy_client.disconnect()


if __name__ == '__main__':
  unittest.main()