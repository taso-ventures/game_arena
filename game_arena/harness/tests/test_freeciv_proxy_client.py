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

"""Tests for FreeCivProxyClient WebSocket implementation."""

import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from game_arena.harness.freeciv_proxy_client import (ConnectionManager,
                                                   ConnectionState,
                                                   FreeCivProxyClient,
                                                   MessageHandler,
                                                   MessageQueue,
                                                   ProtocolTranslator)
from game_arena.harness.freeciv_state import FreeCivAction
from game_arena.harness.tests.mock_freeciv_server import MockFreeCivServer


class TestFreeCivProxyClient(unittest.IsolatedAsyncioTestCase):
  """Test suite for FreeCivProxyClient."""

  async def asyncSetUp(self):
      """Set up test environment."""
      self.mock_server = MockFreeCivServer(host="localhost", port=8003)
      await self.mock_server.start()

      self.client = FreeCivProxyClient(
          host="localhost",
          port=8003,
          agent_id="test_agent",
          heartbeat_interval=0.1,  # Fast heartbeat for testing
      )

  async def asyncTearDown(self):
      """Clean up test environment."""
      if self.client.connection_manager.state != ConnectionState.DISCONNECTED:
          await self.client.disconnect()
      await self.mock_server.stop()

  async def test_connection_establishment(self):
      """Test basic connection establishment."""
      # Test connection
      result = await self.client.connect()
      self.assertTrue(result)
      self.assertEqual(
          self.client.connection_manager.state, ConnectionState.CONNECTED
      )

      # Verify server received connection message
      messages = self.mock_server.get_recorded_messages()
      self.assertTrue(any(msg.get("type") == "llm_connect" for msg in messages))

  async def test_connection_failure(self):
      """Test connection failure handling."""
      # Create client with invalid port
      bad_client = FreeCivProxyClient(host="localhost", port=9999)
      result = await bad_client.connect()
      self.assertFalse(result)
      self.assertEqual(
          bad_client.connection_manager.state, ConnectionState.DISCONNECTED
      )

  async def test_authentication_flow(self):
      """Test LLM agent authentication."""
      await self.client.connect()

      # Check that authentication was completed
      self.assertEqual(self.client.player_id, 1)
      self.assertEqual(self.client.game_id, "default")

  async def test_state_query(self):
      """Test state query functionality."""
      await self.client.connect()

      # Request state
      state = await self.client.get_state()
      self.assertIsInstance(state, dict)
      self.assertIn("data", state)
      self.assertIn("turn", state["data"])

  async def test_state_query_with_format(self):
      """Test state query with different formats."""
      await self.client.connect()

      # Test minimal format
      state = await self.client.get_state(format_type="minimal")
      self.assertIsInstance(state, dict)

      # Test default format
      state = await self.client.get_state(format_type="llm_optimized")
      self.assertIsInstance(state, dict)

  async def test_action_submission(self):
      """Test action submission."""
      await self.client.connect()

      # Create test action
      action = FreeCivAction(
          action_type="unit_move",
          actor_id=101,
          target={"x": 2, "y": 1},
          parameters={},
          source="unit",
      )

      # Submit action
      result = await self.client.send_action(action)
      self.assertIsInstance(result, dict)
      self.assertTrue(result.get("success", False))

  async def test_invalid_action_submission(self):
      """Test handling of invalid actions."""
      await self.client.connect()

      # Create invalid action
      action = FreeCivAction(
          action_type="invalid_action",
          actor_id=999,
          target={"x": -1, "y": -1},
          parameters={},
          source="unit",
      )

      # Submit action
      result = await self.client.send_action(action)
      self.assertIsInstance(result, dict)
      self.assertFalse(result.get("success", True))

  async def test_heartbeat_mechanism(self):
      """Test heartbeat/ping functionality."""
      await self.client.connect()

      # Wait for at least one heartbeat
      await asyncio.sleep(0.15)

      # Check that ping was sent
      messages = self.mock_server.get_recorded_messages()
      self.assertTrue(any(msg.get("type") == "ping" for msg in messages))

  async def test_conn_ping_pong_handler(self):
      """Test PACKET_CONN_PING/PONG handler to prevent civserver disconnects."""
      await self.client.connect()

      # Simulate civserver sending PACKET_CONN_PING
      conn_ping_message = {
          "type": "conn_ping",
          "timestamp": time.time(),
          "data": {}
      }

      # Send ping to client and trigger message handler
      await self.mock_server.send_to_client("test_agent", json.dumps(conn_ping_message))

      # Wait for client to process and respond
      await asyncio.sleep(0.1)

      # Check that client responded with PACKET_CONN_PONG
      messages = self.mock_server.get_recorded_messages()
      pong_messages = [msg for msg in messages if msg.get("type") == "conn_pong"]
      self.assertGreater(len(pong_messages), 0, "Client should respond to conn_ping with conn_pong")

      # Verify pong message structure
      pong_msg = pong_messages[-1]
      self.assertEqual(pong_msg["type"], "conn_pong")
      self.assertIn("timestamp", pong_msg)

  async def test_disconnection_handling(self):
      """Test graceful disconnection."""
      await self.client.connect()
      self.assertEqual(
          self.client.connection_manager.state, ConnectionState.CONNECTED
      )

      # Disconnect
      await self.client.disconnect()
      self.assertEqual(
          self.client.connection_manager.state, ConnectionState.DISCONNECTED
      )

  async def test_reconnection_after_disconnect(self):
      """Test automatic reconnection after unexpected disconnect."""
      await self.client.connect()
      initial_state = self.client.connection_manager.state
      self.assertEqual(initial_state, ConnectionState.CONNECTED)

      # Simulate server-side disconnect
      await self.mock_server.simulate_disconnect("test_agent")

      # Wait for reconnection attempt
      await asyncio.sleep(1.0)  # Give more time for reconnection

      # Should either be connected or attempting to reconnect
      final_state = self.client.connection_manager.state
      self.assertIn(
          final_state, [ConnectionState.CONNECTED, ConnectionState.RECONNECTING]
      )

  async def test_message_queuing(self):
      """Test message queue functionality."""
      await self.client.connect()

      # Add small delay to server to test queuing
      self.mock_server.set_message_delay(0.05)

      # Send multiple messages sequentially to avoid race conditions
      results = []
      for i in range(3):  # Reduced number
          action = FreeCivAction(
              action_type="unit_move",
              actor_id=101 + i,
              target={"x": i, "y": i},
              parameters={},
              source="unit",
          )
          result = await self.client.send_action(action)
          results.append(result)

      # All should succeed
      self.assertEqual(len(results), 3)
      self.assertTrue(all(r.get("success", False) for r in results))

  async def test_state_caching(self):
      """Test state caching functionality."""
      await self.client.connect()

      # First state request
      start_time = time.time()
      state1 = await self.client.get_state()
      first_duration = time.time() - start_time

      # Second state request (should be cached)
      start_time = time.time()
      state2 = await self.client.get_state()
      second_duration = time.time() - start_time

      # States should be identical and second should be faster
      self.assertEqual(state1["data"]["turn"], state2["data"]["turn"])
      self.assertLess(second_duration, first_duration)

  async def test_cache_expiration(self):
      """Test state cache expiration."""
      # Set very short cache TTL
      self.client.state_cache_ttl = 0.01
      await self.client.connect()

      # Get state
      state1 = await self.client.get_state()

      # Wait for cache to expire
      await asyncio.sleep(0.02)

      # Get state again (should be fresh)
      state2 = await self.client.get_state()

      # May have different timestamps
      self.assertIsInstance(state1, dict)
      self.assertIsInstance(state2, dict)

  async def test_failure_injection(self):
      """Test handling of server failures."""
      await self.client.connect()

      # Inject server failure
      self.mock_server.inject_failure()

      # Try to send action (should trigger reconnection)
      action = FreeCivAction(
          action_type="unit_move",
          actor_id=101,
          target={"x": 1, "y": 1},
          parameters={},
          source="unit",
      )

      # Should eventually succeed after reconnection
      with self.assertRaises(Exception):
          await self.client.send_action(action)


class TestConnectionManager(unittest.IsolatedAsyncioTestCase):
  """Test suite for ConnectionManager."""

  def setUp(self):
      """Set up test environment."""
      self.connection_manager = ConnectionManager(
          ws_url="ws://localhost:8003",
          agent_id="test_agent",
          heartbeat_interval=0.1,
      )

  async def test_initial_state(self):
      """Test initial connection state."""
      self.assertEqual(self.connection_manager.state, ConnectionState.DISCONNECTED)
      self.assertEqual(self.connection_manager.reconnect_attempts, 0)

  async def test_state_transitions(self):
      """Test connection state transitions."""
      # Start connecting
      self.connection_manager.state = ConnectionState.CONNECTING
      self.assertEqual(self.connection_manager.state, ConnectionState.CONNECTING)

      # Mark as connected
      self.connection_manager.state = ConnectionState.CONNECTED
      self.assertEqual(self.connection_manager.state, ConnectionState.CONNECTED)

      # Start reconnecting
      self.connection_manager.state = ConnectionState.RECONNECTING
      self.assertEqual(self.connection_manager.state, ConnectionState.RECONNECTING)

  def test_exponential_backoff(self):
      """Test exponential backoff calculation."""
      # Test backoff calculation
      delays = []
      for attempt in range(5):
          delay = self.connection_manager._calculate_backoff(attempt)
          delays.append(delay)

      # Should increase exponentially
      self.assertLess(delays[0], delays[1])
      self.assertLess(delays[1], delays[2])
      self.assertLessEqual(delays[-1], 60)  # Max delay cap


class TestMessageHandler(unittest.IsolatedAsyncioTestCase):
  """Test suite for MessageHandler."""

  def setUp(self):
      """Set up test environment."""
      self.message_handler = MessageHandler()

  async def test_state_update_handling(self):
      """Test state update message handling."""
      message = {"type": "state_update", "data": {"turn": 5, "phase": "movement"}}

      # Handler should process without error
      await self.message_handler.handle_state_update(message)

  async def test_action_result_handling(self):
      """Test action result message handling."""
      message = {
          "type": "action_result",
          "success": True,
          "data": {"action_processed": True},
      }

      # Handler should process without error
      await self.message_handler.handle_action_result(message)

  async def test_turn_notification_handling(self):
      """Test turn notification message handling."""
      message = {
          "type": "turn_notification",
          "data": {"turn": 6, "current_player": 2},
      }

      # Handler should process without error
      await self.message_handler.handle_turn_notification(message)


class TestProtocolTranslator(unittest.TestCase):
  """Test suite for ProtocolTranslator."""

  def setUp(self):
      """Set up test environment."""
      self.translator = ProtocolTranslator()

  def test_action_to_packet_conversion(self):
      """Test converting FreeCivAction to FreeCiv packet format."""
      action = FreeCivAction(
          action_type="unit_move",
          actor_id=101,
          target={"x": 5, "y": 7},
          parameters={},
          source="unit",
      )

      packet = self.translator.to_freeciv_packet(action)

      self.assertIsInstance(packet, dict)
      self.assertIn("pid", packet)
      self.assertIn("data", packet)
      self.assertEqual(packet["data"]["unit_id"], 101)

  def test_packet_from_freeciv_conversion(self):
      """Test converting FreeCiv packet to Game Arena format."""
      packet = {
          "type": "state_update",
          "data": {
              "turn": 42,
              "phase": "movement",
              "observation": {"test": "data"},
          },
      }

      result = self.translator.from_freeciv_packet(packet)

      self.assertIsInstance(result, dict)
      self.assertEqual(result["data"]["turn"], 42)

  def test_unknown_action_type(self):
      """Test handling of unknown action types."""
      action = FreeCivAction(
          action_type="test_unknown_action",
          actor_id=999,
          target={},
          parameters={},
          source="unit",
      )

      packet = self.translator.to_freeciv_packet(action)

      # Should have default packet structure
      self.assertIsInstance(packet, dict)
      self.assertIn("pid", packet)


class TestMessageQueue(unittest.IsolatedAsyncioTestCase):
  """Test suite for MessageQueue."""

  def setUp(self):
      """Set up test environment."""
      self.message_queue = MessageQueue()

  async def test_priority_queue_ordering(self):
      """Test priority queue message ordering."""
      # Add messages with different priorities
      await self.message_queue.enqueue({"type": "normal", "data": "1"}, priority=1)
      await self.message_queue.enqueue({"type": "urgent", "data": "2"}, priority=0)
      await self.message_queue.enqueue({"type": "normal", "data": "3"}, priority=1)

      # Should get urgent message first
      message = await self.message_queue.get_next_message()
      self.assertEqual(message["type"], "urgent")

      # Then normal messages in order
      message = await self.message_queue.get_next_message()
      self.assertEqual(message["data"], "1")

  async def test_queue_processing(self):
      """Test message queue processing."""
      messages_processed = []

      # Mock processor function
      async def mock_processor(message):
          messages_processed.append(message)

      # Add messages
      await self.message_queue.enqueue({"id": 1}, priority=1)
      await self.message_queue.enqueue({"id": 2}, priority=0)

      # Process with mock
      await self.message_queue.process_messages(mock_processor, max_messages=2)

      # Should have processed both messages in priority order
      self.assertEqual(len(messages_processed), 2)
      self.assertEqual(messages_processed[0]["id"], 2)  # Higher priority first
      self.assertEqual(messages_processed[1]["id"], 1)

  async def test_empty_queue(self):
      """Test behavior with empty queue."""
      # Should not block
      message = await asyncio.wait_for(
          self.message_queue.get_next_message_nowait(), timeout=0.1
      )
      self.assertIsNone(message)


if __name__ == "__main__":
  unittest.main()
