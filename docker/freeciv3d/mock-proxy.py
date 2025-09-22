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

"""Mock FreeCiv proxy server for end-to-end testing."""

import asyncio
import json
import logging
import signal
import sys
from typing import Any, Dict, List, Optional

import websockets


class MockFreeCivProxy:
  """Mock proxy server that translates between Game Arena and FreeCiv3D."""

  def __init__(self, port: int = 8443):
    self.port = port
    self.connected_clients = set()
    self.freeciv_connection = None
    self.running = False

    logging.basicConfig(level=logging.INFO)
    self.logger = logging.getLogger(__name__)

  async def start_server(self):
    """Start the proxy server."""
    self.running = True
    self.logger.info(f"Starting mock FreeCiv proxy on port {self.port}")

    async def handle_client(websocket, path):
      await self.handle_client_connection(websocket, path)

    server = await websockets.serve(handle_client, "0.0.0.0", self.port)
    self.logger.info(f"Mock FreeCiv proxy listening on port {self.port}")

    # Connect to FreeCiv3D server
    await self.connect_to_freeciv()

    return server

  async def connect_to_freeciv(self):
    """Connect to the mock FreeCiv3D server."""
    try:
      self.freeciv_connection = await websockets.connect("ws://localhost:7000")
      self.logger.info("Connected to FreeCiv3D server")

      # Start listening to FreeCiv3D messages
      asyncio.create_task(self.listen_to_freeciv())

    except Exception as e:
      self.logger.error(f"Failed to connect to FreeCiv3D: {e}")

  async def listen_to_freeciv(self):
    """Listen for messages from FreeCiv3D server."""
    if not self.freeciv_connection:
      return

    try:
      async for message in self.freeciv_connection:
        data = json.loads(message)
        self.logger.debug(f"Received from FreeCiv3D: {data['type']}")

        # Broadcast relevant updates to connected clients
        if data['type'] in ['game_state_update', 'game_state']:
          await self.broadcast_to_clients({
            "type": "state_update",
            "data": data['data']
          })

    except websockets.exceptions.ConnectionClosed:
      self.logger.warning("FreeCiv3D connection closed")
    except Exception as e:
      self.logger.error(f"Error listening to FreeCiv3D: {e}")

  async def handle_client_connection(self, websocket, path):
    """Handle a new client connection."""
    client_id = f"client_{len(self.connected_clients)}"
    self.connected_clients.add(websocket)
    self.logger.info(f"Client {client_id} connected")

    try:
      # Send initial handshake
      await websocket.send(json.dumps({
        "type": "connected",
        "client_id": client_id,
        "server": "mock-freeciv-proxy"
      }))

      async for message in websocket:
        await self.handle_client_message(websocket, message)

    except websockets.exceptions.ConnectionClosed:
      self.logger.info(f"Client {client_id} disconnected")
    except Exception as e:
      self.logger.error(f"Error handling client {client_id}: {e}")
    finally:
      self.connected_clients.discard(websocket)

  async def handle_client_message(self, websocket, message: str):
    """Handle message from Game Arena client."""
    try:
      data = json.loads(message)
      message_type = data.get("type")

      self.logger.debug(f"Received from client: {message_type}")

      if message_type == "get_game_state":
        await self.handle_get_game_state(websocket, data)
      elif message_type == "get_legal_actions":
        await self.handle_get_legal_actions(websocket, data)
      elif message_type == "send_action":
        await self.handle_send_action(websocket, data)
      elif message_type == "ping":
        await self.handle_ping(websocket, data)
      else:
        await websocket.send(json.dumps({
          "type": "error",
          "message": f"Unknown message type: {message_type}"
        }))

    except json.JSONDecodeError:
      await websocket.send(json.dumps({
        "type": "error",
        "message": "Invalid JSON format"
      }))
    except Exception as e:
      self.logger.error(f"Error processing message: {e}")
      await websocket.send(json.dumps({
        "type": "error",
        "message": "Internal server error"
      }))

  async def handle_get_game_state(self, websocket, data):
    """Handle game state request."""
    if not self.freeciv_connection:
      await websocket.send(json.dumps({
        "type": "error",
        "message": "Not connected to FreeCiv3D server"
      }))
      return

    # Request state from FreeCiv3D
    await self.freeciv_connection.send(json.dumps({
      "type": "get_game_state"
    }))

    # Wait for response (simplified - in real implementation would use proper async handling)
    try:
      response = await asyncio.wait_for(self.freeciv_connection.recv(), timeout=5.0)
      freeciv_data = json.loads(response)

      if freeciv_data['type'] == 'game_state':
        # Transform FreeCiv3D state to Game Arena format
        game_arena_state = self.transform_state_to_game_arena(freeciv_data['data'])

        await websocket.send(json.dumps({
          "type": "game_state",
          "data": game_arena_state
        }))
      else:
        await websocket.send(json.dumps({
          "type": "error",
          "message": "Unexpected response from FreeCiv3D"
        }))

    except asyncio.TimeoutError:
      await websocket.send(json.dumps({
        "type": "error",
        "message": "Timeout waiting for FreeCiv3D response"
      }))

  async def handle_get_legal_actions(self, websocket, data):
    """Handle legal actions request."""
    player_id = data.get("player_id", 1)

    if not self.freeciv_connection:
      await websocket.send(json.dumps({
        "type": "error",
        "message": "Not connected to FreeCiv3D server"
      }))
      return

    # Request legal actions from FreeCiv3D
    await self.freeciv_connection.send(json.dumps({
      "type": "get_legal_actions",
      "player_id": player_id
    }))

    try:
      response = await asyncio.wait_for(self.freeciv_connection.recv(), timeout=5.0)
      freeciv_data = json.loads(response)

      if freeciv_data['type'] == 'legal_actions':
        # Transform to Game Arena action IDs
        action_ids = self.transform_actions_to_ids(freeciv_data['data'])

        await websocket.send(json.dumps({
          "type": "legal_actions",
          "data": action_ids
        }))
      else:
        await websocket.send(json.dumps({
          "type": "error",
          "message": "Unexpected response from FreeCiv3D"
        }))

    except asyncio.TimeoutError:
      await websocket.send(json.dumps({
        "type": "error",
        "message": "Timeout waiting for FreeCiv3D response"
      }))

  async def handle_send_action(self, websocket, data):
    """Handle action submission."""
    action = data.get("action")

    if not self.freeciv_connection:
      await websocket.send(json.dumps({
        "type": "error",
        "message": "Not connected to FreeCiv3D server"
      }))
      return

    # Transform Game Arena action to FreeCiv3D format
    freeciv_action = self.transform_action_to_freeciv(action)

    # Send to FreeCiv3D
    await self.freeciv_connection.send(json.dumps({
      "type": "send_action",
      "action": freeciv_action
    }))

    try:
      response = await asyncio.wait_for(self.freeciv_connection.recv(), timeout=10.0)
      freeciv_result = json.loads(response)

      if freeciv_result['type'] == 'action_result':
        await websocket.send(json.dumps({
          "type": "action_result",
          "data": freeciv_result['data']
        }))
      else:
        await websocket.send(json.dumps({
          "type": "error",
          "message": "Unexpected response from FreeCiv3D"
        }))

    except asyncio.TimeoutError:
      await websocket.send(json.dumps({
        "type": "error",
        "message": "Timeout waiting for action result"
      }))

  async def handle_ping(self, websocket, data):
    """Handle ping request."""
    await websocket.send(json.dumps({
      "type": "pong",
      "timestamp": data.get("timestamp"),
      "server_time": asyncio.get_event_loop().time()
    }))

  def transform_state_to_game_arena(self, freeciv_state: Dict[str, Any]) -> Dict[str, Any]:
    """Transform FreeCiv3D state to Game Arena observation format."""
    return {
      "turn": freeciv_state["turn"],
      "playerID": freeciv_state["active_player"],
      "players": freeciv_state["players"],
      "map": freeciv_state["map"],
      "phase": freeciv_state["phase"],
      "game": {
        "name": "FreeCiv3D",
        "version": "mock-1.0",
        "state": "active"
      }
    }

  def transform_actions_to_ids(self, action_strings: List[str]) -> List[int]:
    """Transform action strings to integer IDs for Game Arena."""
    # Simple hash-based mapping for testing
    action_ids = []
    for i, action_str in enumerate(action_strings):
      # Use hash to create consistent ID mapping
      action_id = hash(action_str) % 10000
      if action_id < 0:
        action_id = -action_id
      action_ids.append(action_id)

    return action_ids

  def transform_action_to_freeciv(self, action: Dict[str, Any]) -> Dict[str, Any]:
    """Transform Game Arena action to FreeCiv3D format."""
    return {
      "action_type": action.get("action_type", "unknown"),
      "actor_id": action.get("actor_id"),
      "target": action.get("target"),
      "parameters": action.get("parameters", {}),
      "player_id": action.get("player_id", 1)
    }

  async def broadcast_to_clients(self, message: Dict[str, Any]):
    """Broadcast message to all connected clients."""
    if not self.connected_clients:
      return

    message_str = json.dumps(message)
    disconnected = set()

    for client in self.connected_clients:
      try:
        await client.send(message_str)
      except websockets.exceptions.ConnectionClosed:
        disconnected.add(client)
      except Exception as e:
        self.logger.error(f"Error broadcasting to client: {e}")
        disconnected.add(client)

    # Remove disconnected clients
    self.connected_clients -= disconnected

  async def shutdown(self):
    """Shutdown the proxy server."""
    self.logger.info("Shutting down mock FreeCiv proxy")
    self.running = False

    if self.freeciv_connection:
      await self.freeciv_connection.close()

    # Close all client connections
    for client in self.connected_clients.copy():
      await client.close()

    self.logger.info("Mock FreeCiv proxy stopped")


async def main():
  """Main entry point."""
  proxy = MockFreeCivProxy()

  def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}, shutting down...")
    asyncio.create_task(proxy.shutdown())

  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)

  try:
    server = await proxy.start_server()
    await server.wait_closed()
  except Exception as e:
    logging.error(f"Server error: {e}")
    await proxy.shutdown()
  finally:
    sys.exit(0)


if __name__ == "__main__":
  asyncio.run(main())