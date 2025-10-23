# FreeCiv3D Mock Server (Testing Only)

This directory contains a **mock FreeCiv3D server** for isolated testing purposes only.

## ⚠️ Important: This is NOT the production FreeCiv3D server

This mock server provides stub implementations for testing Game Arena's FreeCiv integration **without requiring the full FreeCiv3D stack**.

### Mock vs Real FreeCiv3D

| Aspect | Mock (this directory) | Real FreeCiv3D (../freeciv3d) |
|--------|----------------------|-------------------------------|
| Purpose | Unit/E2E testing | Actual gameplay |
| Components | Stub HTTP/WS servers | Full civserver, web client, LLM Gateway |
| Build time | Fast (~30s) | Slow (~5-10 min) |
| Features | Minimal fake responses | Complete FreeCiv game engine |
| Used by | `docker-compose.e2e.yml` | `run_freeciv_game.py` production orchestration |

### What's Inside

- `Dockerfile` - Lightweight Node.js container for mock servers
- `mock-server.js` - HTTP API stub (port 8080)
- `mock-proxy.py` - WebSocket proxy stub (port 8443)

### Usage

Only used for automated testing:

```bash
# Run E2E tests with mock server
docker-compose -f docker-compose.e2e.yml up --build

# Run E2E test suite
docker-compose -f docker-compose.e2e.yml run game-arena-test
```

### For Production Gameplay

**Do NOT use this mock server for actual LLM gameplay.**

Instead, use the real FreeCiv3D server from the sibling repository:

```bash
# Start the real FreeCiv3D server
cd ../freeciv3d
docker-compose up

# Run game orchestration (in this repo)
cd ../game_arena
docker exec game-arena python run_freeciv_game.py
```

See [run_freeciv_game.py](../../run_freeciv_game.py) for the production orchestration setup.
