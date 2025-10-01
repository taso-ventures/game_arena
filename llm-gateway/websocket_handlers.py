#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WebSocket handlers for LLM Gateway
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect, FastAPI
import websockets

try:
    from .connection_manager import connection_manager
    from .config import settings
    from .utils.origin_validator import validate_websocket_origin, get_websocket_origin_info, create_origin_rejection_response
    from .utils.rate_limiter import comprehensive_rate_limiter
    from .utils.constants import MAX_MESSAGE_SIZE_BYTES, ERROR_CODE_VALIDATION
except ImportError:
    from connection_manager import connection_manager
    from config import settings
    from utils.origin_validator import validate_websocket_origin, get_websocket_origin_info, create_origin_rejection_response
    from utils.rate_limiter import comprehensive_rate_limiter
    from utils.constants import MAX_MESSAGE_SIZE_BYTES, ERROR_CODE_VALIDATION

# Gateway will be injected by main.py to avoid circular imports
gateway = None

logger = logging.getLogger("llm-gateway")


class AgentWebSocketHandler:
    """Handler for agent WebSocket connections - pure pass-through to proxy"""

    def __init__(self, websocket: WebSocket, agent_id: str):
        self.websocket = websocket
        self.agent_id = agent_id
        self.connection_id: Optional[str] = None
        self.authenticated = False
        self.player_id: Optional[int] = None
        self.game_id: Optional[str] = None
        self.proxy_connection: Optional[WebSocket] = None  # Connection to freeciv-proxy LLM handler

    async def handle_connection(self):
        """Handle the WebSocket connection lifecycle"""
        # Validate origin before accepting connection
        if not self._validate_origin():
            await self.websocket.close(code=1008, reason="Unauthorized origin")
            return

        await self.websocket.accept()

        try:
            # Check connection limits
            if not await comprehensive_rate_limiter.track_connection(self.agent_id):
                await self._send_error("Connection limit exceeded for agent")
                return

            # Add to connection manager
            self.connection_id = await connection_manager.add_connection(
                self.websocket, "agent", self.agent_id
            )

            # Send welcome message
            await self._send_welcome()

            # Message handling loop
            while True:
                try:
                    data = await self.websocket.receive_text()

                    # WebSocket-level message size enforcement
                    message_size = len(data.encode('utf-8'))
                    if message_size > MAX_MESSAGE_SIZE_BYTES:
                        await self._send_error(
                            f"Message size {message_size} bytes exceeds limit of {MAX_MESSAGE_SIZE_BYTES} bytes",
                            error_code=ERROR_CODE_VALIDATION
                        )
                        logger.warning(
                            f"Agent {self.agent_id} sent oversized message: {message_size} bytes"
                        )
                        continue

                    # Rate limiting and message size validation
                    is_allowed, reason = await comprehensive_rate_limiter.check_rate_limits(
                        self.agent_id,
                        message_size=message_size,
                        message_content=data
                    )

                    if not is_allowed:
                        await self._send_rate_limit_error(reason)
                        continue

                    message = json.loads(data)
                    await self._handle_message(message)

                except WebSocketDisconnect:
                    logger.info(f"Agent {self.agent_id} disconnected")
                    break

                except json.JSONDecodeError:
                    await self._send_error("Invalid JSON format")

                except Exception as e:
                    logger.error(f"Error handling message from {self.agent_id}: {e}")
                    await self._send_error("Message processing failed")

        except Exception as e:
            logger.error(f"Error in agent connection {self.agent_id}: {e}")

        finally:
            # Cleanup proxy connection
            if self.proxy_connection:
                await self.proxy_connection.close()
                self.proxy_connection = None

            # Cleanup connection manager
            if self.connection_id:
                await connection_manager.handle_disconnect(self.connection_id)

            # Release connection tracking
            await comprehensive_rate_limiter.release_connection(self.agent_id)

    async def _send_welcome(self):
        """Send welcome message"""
        welcome = {
            "type": "welcome",
            "handler_id": self.connection_id,
            "message": "LLM agent gateway ready. Send llm_connect message to authenticate."
        }
        await self.websocket.send_text(json.dumps(welcome))

    async def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming message from agent - pass through to proxy"""
        message_type = message.get("type")

        if message_type == "llm_connect":
            # First message - establish proxy connection and forward
            await self._connect_to_proxy_and_forward(message)
        elif self.authenticated and self.proxy_connection:
            # All other messages - forward directly to proxy
            await self._forward_to_proxy(message)
        elif message_type == "ping":
            # Handle pings locally for connection health
            await self._handle_ping(message)
        else:
            await self._send_error("Not connected to proxy")

    async def _connect_to_proxy_and_forward(self, message: Dict[str, Any]):
        """Connect to proxy LLM handler and forward the connect message"""
        try:
            # Connect to the freeciv-proxy LLM handler endpoint
            proxy_url = f"ws://{settings.freeciv_proxy_host}:{settings.freeciv_proxy_port}{settings.freeciv_proxy_ws_path}"
            logger.info(f"Connecting agent {self.agent_id} to proxy: {proxy_url}")

            self.proxy_connection = await websockets.connect(proxy_url)
            logger.info(f"Connected to proxy for agent {self.agent_id}")

            # Start listening for proxy messages in background
            asyncio.create_task(self._listen_to_proxy())

            # Forward the connect message to proxy
            await self.proxy_connection.send(json.dumps(message))
            logger.info(f"Forwarded llm_connect message for agent {self.agent_id}")

            # Mark as authenticated to allow further pass-through
            self.authenticated = True

            # Extract game_id for connection tracking
            if "data" in message:
                self.game_id = message["data"].get("game_id")

        except Exception as e:
            logger.error(f"Failed to connect to proxy for agent {self.agent_id}: {e}")
            await self._send_error(f"Failed to connect to game server: {e}")

    async def _listen_to_proxy(self):
        """Listen for messages from proxy and forward to agent"""
        try:
            while self.proxy_connection:
                try:
                    # Receive message from proxy
                    proxy_message = await self.proxy_connection.recv()

                    # Forward directly to agent
                    await self.websocket.send_text(proxy_message)
                    logger.debug(f"Forwarded proxy message to agent {self.agent_id}")

                except websockets.ConnectionClosed:
                    logger.info(f"Proxy connection closed for agent {self.agent_id}")
                    break
                except Exception as e:
                    logger.error(f"Error forwarding proxy message to agent {self.agent_id}: {e}")

        except Exception as e:
            logger.error(f"Error in proxy listener for agent {self.agent_id}: {e}")
        finally:
            # Clean up proxy connection
            if self.proxy_connection:
                await self.proxy_connection.close()
                self.proxy_connection = None

    async def _forward_to_proxy(self, message: Dict[str, Any]):
        """Forward message from agent to proxy"""
        try:
            if self.proxy_connection:
                await self.proxy_connection.send(json.dumps(message))
                logger.debug(f"Forwarded agent message to proxy: {message.get('type')}")
            else:
                await self._send_error("Proxy connection lost")
        except Exception as e:
            logger.error(f"Failed to forward message to proxy: {e}")
            await self._send_error(f"Failed to forward message: {e}")

    # Removed complex state query and action handlers - now handled by pass-through

    async def _handle_ping(self, message: Dict[str, Any]):
        """Handle ping message"""
        pong = {
            "type": "pong",
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
        await self.websocket.send_text(json.dumps(pong))

    async def _send_error(self, error_message: str, error_code: str = "E500"):
        """Send error message"""
        error = {
            "type": "error",
            "agent_id": self.agent_id,
            "timestamp": time.time(),
            "data": {
                "type": "error",
                "success": False,
                "error_code": error_code,
                "error_message": error_message
            }
        }
        await self.websocket.send_text(json.dumps(error))

    async def _send_rate_limit_error(self, reason: str):
        """Send rate limit error message"""
        error = {
            "type": "rate_limit_error",
            "agent_id": self.agent_id,
            "timestamp": time.time(),
            "data": {
                "type": "error",
                "success": False,
                "error_code": "E429",
                "error_message": f"Rate limit exceeded: {reason}"
            }
        }
        await self.websocket.send_text(json.dumps(error))

    def _validate_origin(self) -> bool:
        """
        Validate WebSocket origin for security

        Returns:
            bool: True if origin is allowed, False otherwise
        """
        try:
            # Get origin info for logging
            origin_info = get_websocket_origin_info(self.websocket)
            logger.debug(f"Agent {self.agent_id} connection from: {origin_info}")

            # For LLM agents, allow null origins (non-browser clients)
            # but validate against allowed origins if present
            is_valid = validate_websocket_origin(
                self.websocket,
                settings.allowed_origins,
                strict_mode=False  # Allow null origins for LLM agents
            )

            if not is_valid:
                logger.warning(
                    f"Rejected agent {self.agent_id} connection from unauthorized origin: "
                    f"{origin_info.get('origin', 'null')}"
                )

            return is_valid

        except Exception as e:
            logger.error(f"Error validating origin for agent {self.agent_id}: {e}")
            return False


class SpectatorWebSocketHandler:
    """Handler for spectator WebSocket connections"""

    def __init__(self, websocket: WebSocket, game_id: str):
        self.websocket = websocket
        self.game_id = game_id
        self.connection_id: Optional[str] = None

    async def handle_connection(self):
        """Handle the WebSocket connection lifecycle"""
        # Validate origin before accepting connection (strict mode for spectators)
        if not self._validate_origin():
            await self.websocket.close(code=1008, reason="Unauthorized origin")
            return

        await self.websocket.accept()

        try:
            # Check if game exists - allow spectators to connect even if game doesn't exist yet
            game_exists = hasattr(gateway, 'game_sessions') and self.game_id in gateway.game_sessions
            if not game_exists:
                logger.info(f"Spectator connecting to game {self.game_id} - game session not found, will wait for game to start")

            # Add to connection manager
            self.connection_id = await connection_manager.add_connection(
                self.websocket, "spectator", self.game_id
            )

            # Send welcome message
            await self._send_welcome()

            # Keep connection alive and handle messages
            while True:
                try:
                    # Spectators mainly receive updates, minimal message handling
                    data = await self.websocket.receive_text()

                    # WebSocket-level message size enforcement
                    message_size = len(data.encode('utf-8'))
                    if message_size > MAX_MESSAGE_SIZE_BYTES:
                        await self._send_error(
                            f"Message size {message_size} bytes exceeds limit of {MAX_MESSAGE_SIZE_BYTES} bytes"
                        )
                        logger.warning(
                            f"Spectator on game {self.game_id} sent oversized message: {message_size} bytes"
                        )
                        continue

                    message = json.loads(data)

                    # Handle spectator messages
                    message_type = message.get("type")
                    if message_type == "ping":
                        await self._handle_ping()
                    elif message_type == "spectator_join":
                        await self._handle_spectator_join(message)
                    else:
                        logger.debug(f"Unknown spectator message type: {message_type}")

                except WebSocketDisconnect:
                    logger.info(f"Spectator disconnected from game {self.game_id}")
                    break

                except json.JSONDecodeError:
                    await self._send_error("Invalid JSON format")

                except Exception as e:
                    logger.error(f"Error handling spectator message: {e}")

        except Exception as e:
            logger.error(f"Error in spectator connection for game {self.game_id}: {e}")

        finally:
            # Cleanup
            if self.connection_id:
                await connection_manager.handle_disconnect(self.connection_id)

    async def _send_welcome(self):
        """Send welcome message to spectator"""
        welcome = {
            "type": "spectator_joined",
            "game_id": self.game_id,
            "message": f"Connected as spectator to game {self.game_id}",
            "timestamp": time.time()
        }
        await self.websocket.send_text(json.dumps(welcome))

        # Send initial game state if available
        await self._send_initial_game_state()

    async def _handle_ping(self):
        """Handle ping from spectator"""
        pong = {
            "type": "pong",
            "game_id": self.game_id,
            "timestamp": time.time()
        }
        await self.websocket.send_text(json.dumps(pong))

    async def _handle_spectator_join(self, message: Dict[str, Any]):
        """Handle spectator join message"""
        # Already handled in _send_welcome, but can send confirmation
        response = {
            "type": "spectator_joined",
            "game_id": self.game_id,
            "spectator_id": message.get("spectator_id"),
            "message": "Successfully joined as spectator",
            "timestamp": time.time()
        }
        await self.websocket.send_text(json.dumps(response))

    async def _send_initial_game_state(self):
        """Send initial game state to spectator"""
        try:
            if hasattr(gateway, 'get_spectator_game_state'):
                game_state = await gateway.get_spectator_game_state(self.game_id)
            elif hasattr(gateway, 'game_sessions') and self.game_id in gateway.game_sessions:
                # Get basic game info from session
                session = gateway.game_sessions[self.game_id]
                game_state = {
                    "turn": getattr(session, 'current_turn', 1),
                    "players": getattr(session, 'players', {}),
                    "game_info": {
                        "status": getattr(session, 'status', 'running'),
                        "turn": getattr(session, 'current_turn', 1)
                    }
                }
            else:
                # Fallback state
                game_state = {
                    "turn": 1,
                    "players": {},
                    "game_info": {"status": "running", "turn": 1}
                }

            state_message = {
                "type": "game_state",
                "game_id": self.game_id,
                "data": game_state,
                "timestamp": time.time()
            }
            await self.websocket.send_text(json.dumps(state_message))

        except Exception as e:
            logger.error(f"Error sending initial game state to spectator: {e}")

    async def send_game_update(self, update_data: Dict[str, Any]):
        """Send game update to spectator (called by game session)"""
        try:
            message = {
                "type": "game_state",
                "game_id": self.game_id,
                "data": update_data,
                "timestamp": time.time()
            }
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending game update to spectator: {e}")

    async def send_turn_update(self, turn_number: int):
        """Send turn update to spectator"""
        try:
            message = {
                "type": "turn_update",
                "game_id": self.game_id,
                "turn": turn_number,
                "timestamp": time.time()
            }
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending turn update to spectator: {e}")

    async def send_player_action(self, player_id: int, action_data: Dict[str, Any]):
        """Send player action to spectator"""
        try:
            message = {
                "type": "player_action",
                "game_id": self.game_id,
                "player_id": player_id,
                "action": action_data,
                "timestamp": time.time()
            }
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending player action to spectator: {e}")

    async def send_game_ended(self, result_data: Dict[str, Any]):
        """Send game ended message to spectator"""
        try:
            message = {
                "type": "game_ended",
                "game_id": self.game_id,
                "result": result_data,
                "timestamp": time.time()
            }
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending game ended to spectator: {e}")

    async def _send_error(self, error_message: str):
        """Send error message"""
        error = {
            "type": "error",
            "game_id": self.game_id,
            "message": error_message
        }
        await self.websocket.send_text(json.dumps(error))

    def _validate_origin(self) -> bool:
        """
        Validate WebSocket origin for security (strict mode for spectators)

        Returns:
            bool: True if origin is allowed, False otherwise
        """
        try:
            # Get origin info for logging
            origin_info = get_websocket_origin_info(self.websocket)
            logger.debug(f"Spectator connection to game {self.game_id} from: {origin_info}")

            # For spectators (browsers), require valid origin
            is_valid = validate_websocket_origin(
                self.websocket,
                settings.allowed_origins,
                strict_mode=True  # Strict mode for browser clients
            )

            if not is_valid:
                logger.warning(
                    f"Rejected spectator connection to game {self.game_id} from unauthorized origin: "
                    f"{origin_info.get('origin', 'null')}"
                )

            return is_valid

        except Exception as e:
            logger.error(f"Error validating origin for spectator connection to game {self.game_id}: {e}")
            return False


# WebSocket route registration
def register_websocket_routes(app: FastAPI):
    """Register WebSocket routes with the FastAPI app"""

    @app.websocket("/ws/agent/{agent_id}")
    async def agent_websocket_endpoint(websocket: WebSocket, agent_id: str):
        """WebSocket endpoint for LLM agents"""
        handler = AgentWebSocketHandler(websocket, agent_id)
        await handler.handle_connection()

    @app.websocket("/ws/spectator/{game_id}")
    async def spectator_websocket_endpoint(websocket: WebSocket, game_id: str):
        """WebSocket endpoint for game spectators"""
        if not settings.enable_spectator_mode:
            await websocket.close(code=1008, reason="Spectator mode disabled")
            return

        handler = SpectatorWebSocketHandler(websocket, game_id)
        await handler.handle_connection()

    logger.info("WebSocket routes registered")