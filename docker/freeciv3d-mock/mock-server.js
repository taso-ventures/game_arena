/**
 * Copyright 2025 The game_arena Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

const WebSocket = require('ws');
const http = require('http');

// Mock FreeCiv3D web server for end-to-end testing
class MockFreeCivServer {
  constructor() {
    this.gameState = {
      turn: 1,
      players: [
        {
          id: 1,
          name: "Player1",
          nation: "Romans",
          score: 100,
          cities: [
            {
              id: 1,
              name: "Rome",
              x: 10,
              y: 10,
              population: 5,
              production: 10,
              trade: 8
            }
          ],
          units: [
            {
              id: 101,
              type: "warrior",
              x: 10,
              y: 14,
              health: 100,
              moves_left: 1
            }
          ],
          technologies: ["pottery", "bronze_working"],
          gold: 50,
          science_rate: 70,
          tax_rate: 20,
          luxury_rate: 10
        },
        {
          id: 2,
          name: "Player2",
          nation: "Greeks",
          score: 95,
          cities: [
            {
              id: 2,
              name: "Athens",
              x: 20,
              y: 20,
              population: 4,
              production: 8,
              trade: 10
            }
          ],
          units: [
            {
              id: 201,
              type: "warrior",
              x: 20,
              y: 24,
              health: 100,
              moves_left: 1
            }
          ],
          technologies: ["pottery", "alphabet"],
          gold: 45,
          science_rate: 80,
          tax_rate: 20,
          luxury_rate: 0
        }
      ],
      map: {
        width: 50,
        height: 50,
        tiles: this.generateMockMap(50, 50)
      },
      phase: "movement",
      active_player: 1
    };

    this.clients = new Map();
    this.actionHistory = [];
  }

  generateMockMap(width, height) {
    const tiles = [];
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        tiles.push({
          x: x,
          y: y,
          terrain: Math.random() > 0.7 ? "forest" : "grassland",
          special: Math.random() > 0.9 ? "river" : null,
          city_id: null,
          unit_ids: []
        });
      }
    }
    return tiles;
  }

  start() {
    // HTTP server for REST API
    this.httpServer = http.createServer((req, res) => {
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

      if (req.method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
      }

      if (req.url === '/api/game/state' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(this.gameState));
      } else if (req.url === '/api/game/legal_actions' && req.method === 'GET') {
        const url = new URL(req.url, `http://${req.headers.host}`);
        const playerId = parseInt(url.searchParams.get('player_id') || '1');
        const legalActions = this.getLegalActions(playerId);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ legal_actions: legalActions }));
      } else if (req.url === '/api/health' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'healthy', server: 'mock-freeciv3d' }));
      } else {
        res.writeHead(404);
        res.end('Not Found');
      }
    });

    // WebSocket server for real-time communication
    this.wsServer = new WebSocket.Server({ port: 7000 });

    this.wsServer.on('connection', (ws, req) => {
      const clientId = Math.random().toString(36).substr(2, 9);
      this.clients.set(clientId, ws);

      console.log(`Client ${clientId} connected`);

      ws.on('message', (data) => {
        try {
          const message = JSON.parse(data.toString());
          this.handleMessage(clientId, message, ws);
        } catch (error) {
          console.error('Invalid JSON message:', error);
          ws.send(JSON.stringify({
            type: 'error',
            message: 'Invalid JSON format'
          }));
        }
      });

      ws.on('close', () => {
        console.log(`Client ${clientId} disconnected`);
        this.clients.delete(clientId);
      });

      // Send initial game state
      ws.send(JSON.stringify({
        type: 'game_state',
        data: this.gameState
      }));
    });

    this.httpServer.listen(8080, () => {
      console.log('Mock FreeCiv3D HTTP server listening on port 8080');
    });

    console.log('Mock FreeCiv3D WebSocket server listening on port 7000');
  }

  handleMessage(clientId, message, ws) {
    console.log(`Received from ${clientId}:`, message);

    switch (message.type) {
      case 'get_game_state':
        ws.send(JSON.stringify({
          type: 'game_state',
          data: this.gameState
        }));
        break;

      case 'get_legal_actions':
        const playerId = message.player_id || 1;
        const legalActions = this.getLegalActions(playerId);
        ws.send(JSON.stringify({
          type: 'legal_actions',
          data: legalActions
        }));
        break;

      case 'send_action':
        const result = this.processAction(message.action);
        this.actionHistory.push({
          timestamp: new Date().toISOString(),
          client_id: clientId,
          action: message.action,
          result: result
        });

        ws.send(JSON.stringify({
          type: 'action_result',
          data: result
        }));

        // Broadcast state update to all clients
        this.broadcastGameState();
        break;

      case 'ping':
        ws.send(JSON.stringify({
          type: 'pong',
          timestamp: new Date().toISOString()
        }));
        break;

      default:
        ws.send(JSON.stringify({
          type: 'error',
          message: `Unknown message type: ${message.type}`
        }));
    }
  }

  getLegalActions(playerId) {
    const player = this.gameState.players.find(p => p.id === playerId);
    if (!player) return [];

    const actions = [];

    // Unit actions
    player.units.forEach(unit => {
      if (unit.moves_left > 0) {
        // Movement actions
        const directions = [
          { x: unit.x + 1, y: unit.y },
          { x: unit.x - 1, y: unit.y },
          { x: unit.x, y: unit.y + 1 },
          { x: unit.x, y: unit.y - 1 }
        ];

        directions.forEach(pos => {
          if (this.isValidPosition(pos.x, pos.y)) {
            actions.push(`unit_move_${unit.type}(${unit.id})_to(${pos.x},${pos.y})`);
          }
        });

        // Other unit actions
        actions.push(`unit_fortify_${unit.type}(${unit.id})`);
        actions.push(`unit_wait_${unit.type}(${unit.id})`);
      }
    });

    // City actions
    player.cities.forEach(city => {
      actions.push(`city_change_production(${city.id})_to(warrior)`);
      actions.push(`city_change_production(${city.id})_to(settler)`);
      actions.push(`city_buy_unit(${city.id})_type(warrior)`);
    });

    // Player actions
    actions.push('player_end_turn');
    actions.push('player_change_government_to(republic)');
    actions.push('player_set_science_rate(80)');

    return actions;
  }

  isValidPosition(x, y) {
    return x >= 0 && x < this.gameState.map.width &&
           y >= 0 && y < this.gameState.map.height;
  }

  processAction(action) {
    console.log('Processing action:', action);

    // Simulate action processing
    const success = Math.random() > 0.1; // 90% success rate

    if (success) {
      // Update game state based on action
      if (action.action_type === 'unit_move') {
        this.updateUnitPosition(action);
      } else if (action.action_type === 'player_end_turn') {
        this.advanceTurn();
      }

      return {
        success: true,
        message: 'Action executed successfully',
        score_delta: Math.floor(Math.random() * 10) - 5
      };
    } else {
      return {
        success: false,
        message: 'Action failed due to game rules',
        error_code: 'INVALID_ACTION'
      };
    }
  }

  updateUnitPosition(action) {
    const player = this.gameState.players.find(p => p.id === action.player_id);
    if (player) {
      const unit = player.units.find(u => u.id === action.actor_id);
      if (unit && action.target) {
        unit.x = action.target.x;
        unit.y = action.target.y;
        unit.moves_left = Math.max(0, unit.moves_left - 1);
      }
    }
  }

  advanceTurn() {
    this.gameState.turn += 1;
    this.gameState.active_player = this.gameState.active_player === 1 ? 2 : 1;

    // Reset unit moves
    this.gameState.players.forEach(player => {
      player.units.forEach(unit => {
        unit.moves_left = 1;
      });
    });
  }

  broadcastGameState() {
    const message = JSON.stringify({
      type: 'game_state_update',
      data: this.gameState
    });

    this.clients.forEach((ws, clientId) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(message);
      }
    });
  }
}

// Start the mock server
const server = new MockFreeCivServer();
server.start();

console.log('Mock FreeCiv3D server started');
console.log('HTTP API: http://localhost:8080');
console.log('WebSocket: ws://localhost:7000');