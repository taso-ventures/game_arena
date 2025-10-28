# FreeCiv LLM vs LLM Gameplay Guide

This guide explains how to use Game Arena to run LLMs playing FreeCiv against each other using the FreeCiv3D Docker container.

## Prerequisites ✅

All prerequisites are already set up in your environment:

1. **✅ FreeCiv3D Docker container** - Available in `../freeciv3d`
2. **✅ Game Arena with OpenSpiel** - Docker container with full framework
3. **✅ API Keys configured** - OpenAI, Gemini, and Anthropic keys in `.env`
4. **✅ FreeCiv LLM Agent** - Complete implementation with rethinking support

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FreeCiv3D     │    │   Game Arena     │    │  LLM APIs       │
│   Docker        │    │   Docker         │    │  (OpenAI,       │
│                 │◄──►│                  │◄──►│   Gemini,       │
│  • Port 8080    │    │  • FreeCivLLM    │    │   Anthropic)    │
│  • Port 8002    │    │    Agent         │    │                 │
│  • LLM Gateway  │    │  • Full Framework│    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Start FreeCiv3D Server
```bash
cd ../freeciv3d
docker-compose up -d
```

Wait for the server to be ready (check logs or visit http://localhost:8080).

### 2. Start Game Arena
```bash
cd game_arena  # (you're already here)
docker-compose up -d game-arena
```

### 3. Run LLM vs LLM Game

**Option A: Use the Custom Game Runner (Recommended)**
```bash
# Basic game: Gemini vs GPT-4
docker exec game-arena python freeciv/run_freeciv_game.py

# Custom game with different models and strategies
docker exec game-arena python freeciv/run_freeciv_game.py \\
  --player1=openai --player2=anthropic \\
  --strategy1=balanced --strategy2=science_victory \\
  --turns=100 --verbose=true
```

**Option B: Use the Existing Demo**
```bash
# Run existing FreeCiv demo (fewer options but simpler)
docker exec game-arena python -m game_arena.harness.freeciv_harness_demo \\
  --num_moves=50 \\
  --gemini_model=gemini-2.5-flash \\
  --openai_model=gpt-4.1
```

## Game Runner Options

### Models Available
- `gemini` - Uses Gemini 2.5 Flash
- `openai` - Uses GPT-4.1
- `anthropic` - Uses Claude 3 Sonnet

### Strategies Available
- `balanced` - Balanced expansion and development
- `aggressive_expansion` - Focus on territorial expansion
- `economic_focus` - Prioritize economic development
- `defensive_turtle` - Defensive, fortification-focused
- `science_victory` - Focus on technological advancement
- `opportunistic` - Adaptive strategy based on circumstances

### Example Commands

**Gemini vs GPT-4 (Balanced vs Aggressive)**
```bash
docker exec game-arena python freeciv/run_freeciv_game.py \\
  --player1=gemini --player2=openai \\
  --strategy1=balanced --strategy2=aggressive_expansion \\
  --turns=75
```

**Claude vs GPT-4 (Science vs Economic)**
```bash
docker exec game-arena python freeciv/run_freeciv_game.py \\
  --player1=anthropic --player2=openai \\
  --strategy1=science_victory --strategy2=economic_focus \\
  --turns=100
```

**Fast Test Game**
```bash
docker exec game-arena python freeciv/run_freeciv_game.py \\
  --turns=10 --verbose=true
```

## What Happens During the Game

1. **Connection**: Game Arena connects to FreeCiv3D LLM Gateway (port 8002)
2. **Agent Creation**: Two LLM agents are created with different models and strategies
3. **Game Loop**:
   - Each agent takes turns analyzing the game state
   - LLM generates moves based on current situation and strategy
   - Game Arena validates moves and handles retries if needed
   - FreeCiv3D processes the moves and updates the game state
4. **Full Framework Features**:
   - **Rethinking**: If LLM generates illegal moves, agent rethinks and tries again
   - **Memory**: Agents remember previous actions for context
   - **Strategy Adaptation**: Agents can adjust strategy based on game progress
   - **Move Validation**: Full OpenSpiel integration for move validation

## Troubleshooting

### FreeCiv3D Not Running
```
✗ FreeCiv3D server not running on host.docker.internal:8080
Start FreeCiv3D with: cd ../freeciv3d && docker-compose up
```

**Solution**: Start the FreeCiv3D server first.

### Connection Issues
```
✗ Connection failed: server rejected WebSocket connection: HTTP 404
```

**Solution**: Wait for FreeCiv3D to fully initialize, or check if it's running on the correct port.

### Missing API Keys
```
✗ Missing API key for OPENAI
Please set OPENAI_API_KEY in your .env file
```

**Solution**: Your API keys are already configured in `.env`, but check that the model type matches available keys.

## Advanced Usage

### Tournament Mode
To run multiple games and collect statistics:

```bash
# Run 5 games between different model pairs
for i in {1..5}; do
  echo "Game $i"
  docker exec game-arena python freeciv/run_freeciv_game.py \\
    --turns=50 --verbose=false
  sleep 5  # Brief pause between games
done
```

### Custom Strategies
You can modify the `_STRATEGY1` and `_STRATEGY2` values in `run_freeciv_game.py` or create new strategy configurations.

### Logging and Analysis
Game logs are available in the Docker containers:
```bash
# View Game Arena logs
docker logs game-arena

# View FreeCiv3D logs
docker logs fciv-net
```

## Current Limitations

1. **Game State Integration**: The current implementation uses a simplified game loop. Full integration with FreeCiv's complex state system is partially implemented.

2. **Multi-Player Support**: Currently designed for 2-player games. FreeCiv supports more players but would require additional agent coordination.

3. **Save/Load Games**: Game persistence across container restarts is not currently implemented.

## Next Steps

1. **Start FreeCiv3D**: `cd ../freeciv3d && docker-compose up -d`
2. **Run Your First Game**: `docker exec game-arena python freeciv/run_freeciv_game.py`
3. **Experiment**: Try different model combinations and strategies
4. **Watch the Magic**: See LLMs playing civilization against each other! 🎮

## Architecture Benefits

This setup gives you:
- **Full Game Arena Framework**: Tournament system, move validation, retry logic
- **OpenSpiel Integration**: Proper game state management and validation
- **Multiple LLM Support**: Easy switching between OpenAI, Gemini, and Anthropic
- **Strategy System**: Different AI personalities and approaches
- **Production Ready**: Docker containers, proper error handling, logging
- **Extensible**: Easy to add new models, strategies, or game modes

You're now ready to watch AIs play Civilization! 🏛️⚔️🤖