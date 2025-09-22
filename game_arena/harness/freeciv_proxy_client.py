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

"""FreeCiv Proxy Client for WebSocket communication with FreeCiv3D server."""

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections import OrderedDict
from enum import Enum
from typing import Any, Callable, Dict, Optional

import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed, WebSocketException

from game_arena.harness.freeciv_state import FreeCivAction

logger = logging.getLogger(__name__)

# Security constants
MAX_JSON_SIZE = 1_000_000  # 1MB limit for JSON messages
MAX_JSON_DEPTH = 100  # Maximum nesting depth
MAX_WEBSOCKET_SIZE = 10**6  # 1MB limit for WebSocket messages
MAX_WEBSOCKET_QUEUE = 32  # Maximum queued messages
WEBSOCKET_CLOSE_TIMEOUT = 5.0  # Timeout for WebSocket close operations

# Cache key validation pattern
CACHE_KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_.-]+$')


class PacketID(Enum):
  """FreeCiv packet type identifiers."""

  UNIT_ORDERS = 31
  CITY_CHANGE_PRODUCTION = 85
  GENERIC = 0


# Configuration constants
DEFAULT_HEARTBEAT_INTERVAL = 30.0
DEFAULT_STATE_CACHE_TTL = 5.0
DEFAULT_MAX_CACHE_ENTRIES = 10
DEFAULT_MAX_RECONNECT_ATTEMPTS = 3
BACKGROUND_TASK_SHUTDOWN_TIMEOUT = 5.0


def _count_json_depth(obj: Any, depth: int = 0) -> int:
  """Count the maximum nesting depth of a JSON object."""
  if depth > MAX_JSON_DEPTH:
      raise ValueError(f"JSON depth exceeds maximum of {MAX_JSON_DEPTH}")

  if isinstance(obj, dict):
      return max(
          (_count_json_depth(v, depth + 1) for v in obj.values()),
          default=depth
      )
  elif isinstance(obj, list):
      return max(
          (_count_json_depth(item, depth + 1) for item in obj),
          default=depth
      )
  return depth


def safe_json_loads(json_string: str) -> Dict[str, Any]:
  """Safely parse JSON with size and depth validation.

  Args:
      json_string: JSON string to parse

  Returns:
      Parsed JSON object

  Raises:
      ValueError: If JSON is invalid, too large, or too deeply nested
      json.JSONDecodeError: If JSON syntax is invalid
  """
  if not isinstance(json_string, str):
      raise ValueError("Input must be a string")

  # Check size limit
  if len(json_string.encode('utf-8')) > MAX_JSON_SIZE:
      raise ValueError(f"JSON size {len(json_string)} exceeds maximum of {MAX_JSON_SIZE}")

  # Parse JSON
  try:
      data = json.loads(json_string)
  except json.JSONDecodeError as e:
      logger.warning(f"Invalid JSON received: {e}")
      raise

  # Validate depth
  try:
      _count_json_depth(data)
  except ValueError as e:
      logger.warning(f"JSON depth validation failed: {e}")
      raise

  # Basic schema validation - ensure it's a dict with reasonable structure
  if not isinstance(data, dict):
      raise ValueError("JSON must be an object at root level")

  return data


def create_secure_cache_key(prefix: str, identifier: str) -> str:
  """Create a secure cache key with validation.

  Args:
      prefix: Cache key prefix
      identifier: Unique identifier for the cached item

  Returns:
      Secure cache key

  Raises:
      ValueError: If inputs contain invalid characters
  """
  # Validate inputs
  if not isinstance(prefix, str) or not isinstance(identifier, str):
      raise ValueError("Prefix and identifier must be strings")

  # Sanitize inputs - only allow alphanumeric and safe characters
  clean_prefix = re.sub(r'[^a-zA-Z0-9_]', '', prefix)
  clean_identifier = re.sub(r'[^a-zA-Z0-9_]', '', identifier)

  if not clean_prefix or not clean_identifier:
      raise ValueError("Invalid characters in cache key components")

  # Create hash of original identifier to prevent collision attacks
  identifier_hash = hashlib.sha256(identifier.encode('utf-8')).hexdigest()[:16]

  return f"{clean_prefix}_{clean_identifier}_{identifier_hash}"


class ConnectionState(Enum):
  """Connection state enumeration."""

  DISCONNECTED = "disconnected"
  CONNECTING = "connecting"
  CONNECTED = "connected"
  RECONNECTING = "reconnecting"


class FreeCivProxyClient:
  """WebSocket client for communicating with FreeCiv3D proxy server."""

  def __init__(
      self,
      host: str = "localhost",
      port: int = 8002,
      agent_id: Optional[str] = None,
      game_id: str = "default",
      api_token: Optional[str] = None,
      endpoint: str = "/llmsocket",
      heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
      state_cache_ttl: float = DEFAULT_STATE_CACHE_TTL,
      max_cache_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
  ):
      """Initialize FreeCiv proxy client.

      Args:
          host: FreeCiv3D server host
          port: FreeCiv3D server port
          agent_id: Unique agent identifier
          game_id: Game session identifier
          api_token: API token for authentication with FreeCiv3D LLM gateway
          endpoint: WebSocket endpoint path (default: /llmsocket for FreeCiv3D LLM gateway)
          heartbeat_interval: Heartbeat interval in seconds
          state_cache_ttl: State cache TTL in seconds
          max_cache_entries: Maximum number of cache entries
      """
      self.host = host
      self.port = port
      self.agent_id = agent_id or f"agent_{uuid.uuid4().hex[:8]}"
      self.game_id = game_id
      self.api_token = api_token
      self.endpoint = endpoint
      self.heartbeat_interval = heartbeat_interval
      self.state_cache_ttl = state_cache_ttl
      self.max_cache_entries = max_cache_entries

      # Connection management - construct full WebSocket URL with endpoint
      ws_url = f"ws://{host}:{port}{endpoint}/{port}"
      self.connection_manager = ConnectionManager(
          ws_url=ws_url,
          agent_id=self.agent_id,
          heartbeat_interval=heartbeat_interval,
      )

      # Message handling
      self.message_handler = MessageHandler()
      self.message_queue = MessageQueue()
      self.protocol_translator = ProtocolTranslator()

      # State management
      self.player_id: Optional[int] = None
      self.state_cache: OrderedDict[str, Any] = OrderedDict()
      self.last_state_update = 0

      # Background tasks
      self._heartbeat_task: Optional[asyncio.Task] = None
      self._message_processor_task: Optional[asyncio.Task] = None

  async def connect(self) -> bool:
      """Connect to FreeCiv3D server.

      Returns:
          True if connection successful, False otherwise
      """
      try:
          # Establish WebSocket connection
          success = await self.connection_manager.connect()
          if not success:
              return False

          # Wait for welcome message
          welcome_response = await self.connection_manager.receive_message()
          if welcome_response:
              try:
                  welcome_data = safe_json_loads(welcome_response)
                  logger.debug(f"Received welcome: {welcome_data}")
              except (ValueError, json.JSONDecodeError) as e:
                  logger.error(f"Invalid welcome message: {e}")
                  return False

          # Send authentication message
          auth_message = {
              "type": "llm_connect",
              "agent_id": self.agent_id,
              "timestamp": time.time(),
              "data": {
                  "api_token": self.api_token or "test-token-fc3d-001",
                  "model": "gpt-4",
                  "game_id": self.game_id,
                  "capabilities": ["move", "build", "research"]
              }
          }

          # Log the authentication message without sensitive api_token
          auth_message_log = dict(auth_message)
          # Remove or redact sensitive data before logging
          if "data" in auth_message_log:
              auth_message_log["data"] = dict(auth_message_log["data"])
              auth_message_log["data"]["api_token"] = "***REDACTED***"
          logger.debug(f"Sending auth message: {auth_message_log}")
          await self.connection_manager.send_message(json.dumps(auth_message))

          # Wait for authentication response
          auth_response = await self.connection_manager.receive_message()
          if auth_response:
              try:
                  auth_data = safe_json_loads(auth_response)
              except (ValueError, json.JSONDecodeError) as e:
                  logger.error(f"Invalid authentication response: {e}")
                  return False

              if auth_data.get("type") == "auth_success":
                  self.player_id = auth_data.get("player_id")
                  logger.info(
                      f"Successfully authenticated as player {self.player_id}"
                  )

                  # Start background tasks
                  await self._start_background_tasks()
                  return True
              else:
                  logger.error(f"Authentication failed: {auth_data}")

          return False

      except Exception as e:
          logger.error(f"Failed to connect: {e}")
          return False

  async def disconnect(self) -> None:
      """Disconnect from FreeCiv3D server."""
      # Stop background tasks
      await self._stop_background_tasks()

      # Disconnect WebSocket
      await self.connection_manager.disconnect()

      # Clear state
      self.player_id = None
      self.state_cache.clear()

  async def get_state(self, format_type: str = "llm_optimized") -> Dict[str, Any]:
      """Get current game state.

      Args:
          format_type: State format ("llm_optimized", "minimal", etc.)

      Returns:
          Current game state dictionary

      Raises:
          RuntimeError: If not connected to FreeCiv server or failed to get state
          ConnectionClosed: If WebSocket connection is lost during request
          json.JSONDecodeError: If server response cannot be parsed
          asyncio.TimeoutError: If state request times out
      """
      if self.connection_manager.state != ConnectionState.CONNECTED:
          raise RuntimeError("Not connected to FreeCiv server")

      # Check cache first
      try:
          cache_key = create_secure_cache_key("state", format_type)
      except ValueError as e:
          logger.error(f"Invalid format_type for cache key: {e}")
          raise ValueError(f"Invalid format_type: {format_type}")
      current_time = time.time()

      if (
          cache_key in self.state_cache
          and current_time - self.state_cache[cache_key].get("_timestamp", 0)
          < self.state_cache_ttl
      ):
          # Move to end (most recently used) for LRU
          cached_item = self.state_cache.pop(cache_key)
          self.state_cache[cache_key] = cached_item
          return {
              k: v
              for k, v in cached_item.items()
              if k != "_timestamp"
          }

      # Request fresh state
      state_request = {
          "type": "state_query",
          "format": format_type,
          "agent_id": self.agent_id,
      }

      # Send directly for now (message queue can be used for batching later)
      await self.connection_manager.send_message(json.dumps(state_request))
      response = await self._wait_for_response("state_update")

      if response:
          # Evict old cache entries if limit exceeded using LRU
          if len(self.state_cache) >= self.max_cache_entries:
              # Remove least recently used entry (first item in OrderedDict)
              oldest_key, _ = self.state_cache.popitem(last=False)
              logger.debug(f"Evicted LRU cache entry: {oldest_key}")

          # Cache the response with timestamp
          self.state_cache[cache_key] = {**response, "_timestamp": current_time}
          self.last_state_update = current_time
          return response

      raise RuntimeError("Failed to get game state")

  async def send_action(self, action: FreeCivAction) -> Dict[str, Any]:
      """Send action to FreeCiv server.

      Args:
          action: FreeCivAction to send

      Returns:
          Action result dictionary

      Raises:
          RuntimeError: If not connected to FreeCiv server or failed to send action
          ConnectionClosed: If WebSocket connection is lost during request
          json.JSONDecodeError: If server response cannot be parsed
          asyncio.TimeoutError: If action request times out
          ValueError: If action format is invalid
      """
      if self.connection_manager.state != ConnectionState.CONNECTED:
          raise RuntimeError("Not connected to FreeCiv server")

      # Convert action to packet format
      packet = self.protocol_translator.to_freeciv_packet(action)

      action_request = {
          "type": "action",
          "agent_id": self.agent_id,
          "data": packet,
      }

      # Send directly for now (message queue can be used for batching later)
      await self.connection_manager.send_message(json.dumps(action_request))
      response = await self._wait_for_response("action_result")

      if response:
          return response

      raise RuntimeError("Failed to send action")

  async def _start_background_tasks(self) -> None:
      """Start background tasks for heartbeat and message processing."""
      self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
      self._message_processor_task = asyncio.create_task(
          self._message_processor_loop()
      )

  async def _stop_background_tasks(self) -> None:
      """Stop background tasks with graceful shutdown."""
      tasks = []

      # Cancel tasks
      if self._heartbeat_task and not self._heartbeat_task.done():
          self._heartbeat_task.cancel()
          tasks.append(self._heartbeat_task)

      if self._message_processor_task and not self._message_processor_task.done():
          self._message_processor_task.cancel()
          tasks.append(self._message_processor_task)

      # Wait for all tasks with timeout
      if tasks:
          try:
              done, pending = await asyncio.wait(
                  tasks,
                  timeout=BACKGROUND_TASK_SHUTDOWN_TIMEOUT,
                  return_when=asyncio.ALL_COMPLETED,
              )

              # Force cancel any remaining tasks and await them
              for task in pending:
                  task.cancel()
                  try:
                      await task
                  except asyncio.CancelledError:
                      pass  # Expected for cancelled tasks
                  except Exception as e:
                      logger.warning(f"Error in cancelled task {task.get_name()}: {e}")

              # Ensure all done tasks are properly awaited to handle any exceptions
              for task in done:
                  try:
                      await task
                  except asyncio.CancelledError:
                      pass  # Expected for cancelled tasks
                  except Exception as e:
                      logger.warning(f"Error in completed task {task.get_name()}: {e}")

          except Exception as e:
              logger.error(f"Error during background task shutdown: {e}")
          finally:
              # Ensure all task references are cleared
              self._heartbeat_task = None
              self._message_processor_task = None

  async def _heartbeat_loop(self) -> None:
      """Background heartbeat loop."""
      while self.connection_manager.state == ConnectionState.CONNECTED:
          try:
              ping_message = {
                  "type": "ping",
                  "timestamp": int(time.time()),
                  "echo": f"heartbeat_{uuid.uuid4().hex[:8]}",
              }
              await self.message_queue.enqueue(ping_message, priority=2)
              await asyncio.sleep(self.heartbeat_interval)
          except asyncio.CancelledError:
              break
          except Exception as e:
              logger.warning(f"Heartbeat error: {e}")

  async def _message_processor_loop(self) -> None:
      """Background message processing loop."""
      while self.connection_manager.state == ConnectionState.CONNECTED:
          try:
              await self.message_queue.process_messages(
                  self._send_message_to_server, max_messages=10
              )
              await asyncio.sleep(0.01)  # Small delay to prevent busy loop
          except asyncio.CancelledError:
              break
          except Exception as e:
              logger.warning(f"Message processing error: {e}")

  async def _send_message_to_server(self, message: Dict[str, Any]) -> None:
      """Send message to server via WebSocket.

      Args:
          message: Message to send
      """
      try:
          message_str = json.dumps(message)
          await self.connection_manager.send_message(message_str)
      except Exception as e:
          logger.error(f"Failed to send message: {e}")
          # Try to reconnect if connection is lost
          if self.connection_manager.state == ConnectionState.CONNECTED:
              await self.connection_manager.reconnect()

  async def _wait_for_response(
      self, expected_type: str, timeout: float = 30.0
  ) -> Optional[Dict[str, Any]]:
      """Wait for specific response type.

      Args:
          expected_type: Expected message type
          timeout: Timeout in seconds

      Returns:
          Response message or None if timeout/error

      Raises:
          RuntimeError: If connection is lost during wait
          json.JSONDecodeError: If response cannot be parsed as JSON
          ConnectionClosed: If WebSocket connection is closed unexpectedly
          WebSocketException: If other WebSocket errors occur
      """
      start_time = time.time()

      while time.time() - start_time < timeout:
          # Check if still connected
          if self.connection_manager.state != ConnectionState.CONNECTED:
              logger.warning("Connection lost while waiting for response")
              return None

          try:
              response = await asyncio.wait_for(
                  self.connection_manager.receive_message(), timeout=1.0
              )
              if response:
                  try:
                      data = safe_json_loads(response)
                      if data.get("type") == expected_type:
                          return data
                      else:
                          # Handle other message types
                          await self.message_handler.handle_message(data)
                  except (ValueError, json.JSONDecodeError) as e:
                      logger.warning(f"Invalid JSON response: {e}")
                      continue
          except asyncio.TimeoutError:
              continue
          except Exception as e:
              logger.warning(f"Error waiting for response: {e}")
              # If we get repeated errors, break to prevent infinite loops
              break

      return None


class ConnectionManager:
  """Manages WebSocket connection with reconnection logic."""

  def __init__(
      self,
      ws_url: str,
      agent_id: str,
      heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
      max_reconnect_attempts: int = DEFAULT_MAX_RECONNECT_ATTEMPTS,
  ):
      """Initialize connection manager.

      Args:
          ws_url: WebSocket URL
          agent_id: Agent identifier
          heartbeat_interval: Heartbeat interval in seconds
          max_reconnect_attempts: Maximum reconnection attempts
      """
      self.ws_url = ws_url
      self.agent_id = agent_id
      self.heartbeat_interval = heartbeat_interval
      self.max_reconnect_attempts = max_reconnect_attempts

      self.websocket: Optional[WebSocketClientProtocol] = None
      self.state = ConnectionState.DISCONNECTED
      self.reconnect_attempts = 0

  async def connect(self) -> bool:
      """Establish WebSocket connection.

      Returns:
          True if successful, False otherwise
      """
      self.state = ConnectionState.CONNECTING

      try:
          self.websocket = await websockets.connect(
              self.ws_url,
              ping_interval=self.heartbeat_interval,
              ping_timeout=10,
              max_size=MAX_WEBSOCKET_SIZE,
              max_queue=MAX_WEBSOCKET_QUEUE
          )
          self.state = ConnectionState.CONNECTED
          self.reconnect_attempts = 0
          logger.info(f"Connected to {self.ws_url}")
          return True

      except Exception as e:
          logger.error(f"Connection failed: {e}")
          self.state = ConnectionState.DISCONNECTED
          return False

  async def disconnect(self) -> None:
      """Close WebSocket connection."""
      if self.websocket:
          try:
              await asyncio.wait_for(
                  self.websocket.close(), timeout=WEBSOCKET_CLOSE_TIMEOUT
              )
          except asyncio.TimeoutError:
              logger.warning("WebSocket close timed out")
          except Exception as e:
              logger.warning(f"Error closing WebSocket: {e}")
          finally:
              self.websocket = None
      self.state = ConnectionState.DISCONNECTED

  async def reconnect(self) -> bool:
      """Attempt to reconnect with exponential backoff.

      Returns:
          True if reconnection successful, False otherwise
      """
      if self.reconnect_attempts >= self.max_reconnect_attempts:
          logger.error("Max reconnection attempts reached")
          # Reset for future manual reconnect attempts
          self.reconnect_attempts = 0
          return False

      self.state = ConnectionState.RECONNECTING
      self.reconnect_attempts += 1

      # Exponential backoff
      backoff_delay = self._calculate_backoff(self.reconnect_attempts)
      logger.info(
          f"Reconnecting in {backoff_delay}s (attempt {self.reconnect_attempts})"
      )
      await asyncio.sleep(backoff_delay)

      return await self.connect()

  def _calculate_backoff(self, attempt: int) -> float:
      """Calculate exponential backoff delay.

      Args:
          attempt: Attempt number

      Returns:
          Delay in seconds
      """
      return min(2**attempt, 60)  # Cap at 60 seconds

  async def send_message(self, message: str) -> None:
      """Send message via WebSocket.

      Args:
          message: Message string to send

      Raises:
          RuntimeError: If not connected
      """
      if not self.websocket or self.state != ConnectionState.CONNECTED:
          raise RuntimeError("Not connected")

      try:
          await self.websocket.send(message)
      except (ConnectionClosed, WebSocketException) as e:
          logger.warning(f"Send failed, attempting reconnect: {e}")
          await self.reconnect()
          raise

  async def receive_message(self) -> Optional[str]:
      """Receive message from WebSocket.

      Returns:
          Message string or None if error

      Raises:
          RuntimeError: If not connected
      """
      if not self.websocket or self.state != ConnectionState.CONNECTED:
          raise RuntimeError("Not connected")

      try:
          message = await self.websocket.recv()
          return message
      except (ConnectionClosed, WebSocketException) as e:
          logger.warning(f"Receive failed, attempting reconnect: {e}")
          await self.reconnect()
          return None


class MessageHandler:
  """Handles incoming messages from FreeCiv server."""

  async def handle_message(self, message: Dict[str, Any]) -> None:
      """Route message to appropriate handler.

      Args:
          message: Parsed message dictionary
      """
      msg_type = message.get("type", "")

      if msg_type == "state_update":
          await self.handle_state_update(message)
      elif msg_type == "action_result":
          await self.handle_action_result(message)
      elif msg_type == "turn_notification":
          await self.handle_turn_notification(message)
      elif msg_type == "pong":
          await self.handle_pong(message)
      else:
          logger.warning(f"Unknown message type: {msg_type}")

  async def handle_state_update(self, message: Dict[str, Any]) -> None:
      """Handle state update from server.

      Args:
          message: State update message
      """
      logger.debug("Received state update")
      # State updates are handled by the client's response waiting logic

  async def handle_action_result(self, message: Dict[str, Any]) -> None:
      """Handle action result from server.

      Args:
          message: Action result message
      """
      success = message.get("success", False)
      logger.debug(f"Action result: success={success}")

  async def handle_turn_notification(self, message: Dict[str, Any]) -> None:
      """Handle turn notification from server.

      Args:
          message: Turn notification message
      """
      turn = message.get("data", {}).get("turn", "unknown")
      logger.info(f"Turn notification: {turn}")

  async def handle_pong(self, message: Dict[str, Any]) -> None:
      """Handle pong response from server.

      Args:
          message: Pong message
      """
      logger.debug("Received pong")


class ProtocolTranslator:
  """Translates between Game Arena and FreeCiv protocol formats."""

  def to_freeciv_packet(self, action: FreeCivAction) -> Dict[str, Any]:
      """Convert FreeCivAction to FreeCiv packet format.

      Args:
          action: FreeCivAction to convert

      Returns:
          FreeCiv packet dictionary
      """
      if action.action_type == "unit_move":
          return {
              "pid": PacketID.UNIT_ORDERS.value,
              "data": {
                  "unit_id": action.actor_id,
                  "dest_x": action.target.get("x", 0),
                  "dest_y": action.target.get("y", 0),
                  **action.parameters,
              },
          }
      elif action.action_type == "city_production":
          return {
              "pid": PacketID.CITY_CHANGE_PRODUCTION.value,
              "data": {
                  "city_id": action.actor_id,
                  "production_id": action.target,
                  **action.parameters,
              },
          }
      else:
          # Default packet format for unknown actions
          return {
              "pid": PacketID.GENERIC.value,
              "data": {
                  "action_type": action.action_type,
                  "actor_id": action.actor_id,
                  "target": action.target,
                  "parameters": action.parameters,
              },
          }

  def from_freeciv_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
      """Convert FreeCiv packet to Game Arena format.

      Args:
          packet: FreeCiv packet dictionary

      Returns:
          Game Arena format dictionary
      """
      # For now, pass through as-is since FreeCiv3D proxy
      # already sends in a compatible format
      return packet


class MessageQueue:
  """Priority message queue for outgoing messages."""

  def __init__(self):
      """Initialize message queue."""
      self.priority_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
      self.normal_queue: asyncio.Queue = asyncio.Queue()

  async def enqueue(self, message: Dict[str, Any], priority: int = 1) -> None:
      """Add message to queue.

      Args:
          message: Message to queue
          priority: Priority level (0=highest, 1=normal, 2=lowest)
      """
      if priority == 0:
          await self.priority_queue.put((priority, time.time(), message))
      else:
          await self.normal_queue.put(message)

  async def get_next_message(self) -> Optional[Dict[str, Any]]:
      """Get next message from queue.

      Returns:
          Next message or None if empty
      """
      # Check priority queue first
      try:
          _, _, message = self.priority_queue.get_nowait()
          return message
      except asyncio.QueueEmpty:
          pass

      # Then check normal queue
      try:
          message = self.normal_queue.get_nowait()
          return message
      except asyncio.QueueEmpty:
          return None

  async def get_next_message_nowait(self) -> Optional[Dict[str, Any]]:
      """Get next message without waiting.

      Returns:
          Next message or None if empty
      """
      try:
          # Check priority queue first
          if not self.priority_queue.empty():
              _, _, message = self.priority_queue.get_nowait()
              return message

          # Then normal queue
          if not self.normal_queue.empty():
              return self.normal_queue.get_nowait()

      except asyncio.QueueEmpty:
          pass

      return None

  async def process_messages(
      self, processor: Callable, max_messages: int = 10
  ) -> None:
      """Process queued messages.

      Args:
          processor: Async function to process messages
          max_messages: Maximum messages to process per call
      """
      processed = 0
      while processed < max_messages:
          message = await self.get_next_message_nowait()
          if not message:
              break

          try:
              await processor(message)
              processed += 1
          except Exception as e:
              logger.error(f"Error processing message: {e}")
              break
