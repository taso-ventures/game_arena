# FreeCiv Testing Guide

This guide explains how to test the FreeCiv integration in Game Arena, including setup, running games, and troubleshooting.

## Prerequisites

### 1. FreeCiv3D Server

The FreeCiv integration requires a running FreeCiv3D server. This should be running on port 8080 from the adjacent `freeciv3d` repository.

**Start FreeCiv3D server:**
```bash
cd ../freeciv3d
docker-compose up
```

**Verify FreeCiv3D is running:**
```bash
curl http://localhost:8080/status
```

### 2. API Keys

Set up your LLM API keys by copying the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 3. Python Dependencies

Install Game Arena with FreeCiv dependencies:
```bash
pip install -e .[dev]
pip install websockets aiohttp requests
```

## Docker Setup

### Build and Run Game Arena Container

The Docker setup is configured to connect to FreeCiv3D server running on the host:

```bash
# Build the container
docker-compose build

# Start the container (with environment file)
docker-compose up

# Access the container for testing
docker-compose exec game-arena bash
```

### Environment Configuration

The Docker setup automatically configures:
- Connection to FreeCiv3D server on `host.docker.internal:8080`
- WebSocket connection to `host.docker.internal:4002`
- Volume mount for live code development
- API key environment variables

## Testing FreeCiv Integration

### 1. Quick Connection Test

Test the basic connection to FreeCiv3D:

```bash
python scripts/test_freeciv_connection.py
```

This will test:
- HTTP connection to FreeCiv server
- WebSocket connection (if available)
- FreeCiv client functionality
- Game state retrieval and parsing

### 2. Run a FreeCiv Demo Game

Run a short FreeCiv game between two LLM agents:

```bash
python scripts/run_game.py freeciv --num_moves=5
```

Options:
- `--num_moves=N`: Number of moves to play (default: 10)
- `--parser_choice=rule_then_soft|llm_only`: Parser strategy
- `--gemini_model=MODEL`: Gemini model name
- `--openai_model=MODEL`: OpenAI model name

### 3. Test All Games

Run comprehensive tests for all supported games:

```bash
python scripts/test_all_games.py --games chess freeciv
```

Options:
- `--games GAMES`: Specify which games to test
- `--num_moves=N`: Moves per test (default: 3)
- `--verbose`: Detailed output
- `--skip_setup_check`: Skip pre-test validation

### 4. Direct FreeCiv Demo

Run the FreeCiv-specific demo directly:

```bash
python -m game_arena.harness.freeciv_harness_demo \
  --num_moves=10 \
  --parser_choice=rule_then_soft
```

## Game Arena vs OpenSpiel

Game Arena now supports three types of games:

1. **Chess & Go**: Use OpenSpiel's pyspiel.load_game()
2. **FreeCiv**: Uses FreeCivState adapter connecting to external FreeCiv3D server

The architecture maintains consistency:
- Same model generation and prompt systems
- Same parser framework (with FreeCiv-specific extensions)
- Same tournament utilities
- Same observation and action interfaces

## Testing Architecture

### FreeCiv Components

- **FreeCivState**: OpenSpiel-compatible game state adapter
- **FreeCivClient**: WebSocket/HTTP client for FreeCiv3D server
- **FreeCivParsers**: Rule-based and soft parsing for FreeCiv actions
- **FreeCiv Formatter**: State representation for LLM consumption
- **FreeCiv Demo**: Harness demo extending existing chess demo

### Integration Points

- **Prompt Generation**: Reuses existing prompt templates with FreeCiv-specific substitutions
- **Model Generation**: Same LLM calling infrastructure
- **Parser Chain**: Extends existing parser with FreeCiv action formats
- **Tournament Util**: Compatible with existing tournament infrastructure

## Troubleshooting

### FreeCiv3D Server Issues

**Problem**: Connection refused on port 8080
```bash
# Check if FreeCiv3D is running
docker ps | grep fciv-net

# Restart FreeCiv3D
cd ../freeciv3d
docker-compose down
docker-compose up
```

**Problem**: WebSocket connection fails
- WebSocket support is optional; the system falls back to HTTP
- Ensure port 4002 is accessible if you need WebSocket functionality

### API Key Issues

**Problem**: Missing API keys
```bash
# Check environment variables
echo $GEMINI_API_KEY
echo $OPENAI_API_KEY

# Verify .env file exists and has correct format
cat .env
```

### Game State Issues

**Problem**: FreeCiv state parsing errors
- The system includes mock game states for testing when server is unavailable
- Check server logs for JSON parsing issues
- Verify FreeCiv3D is returning valid game state data

### Parser Issues

**Problem**: Action parsing failures
- Try different parser strategies: `--parser_choice=llm_only`
- Check that LLM responses contain recognizable action formats
- Review FreeCiv action examples in parser configuration

## Configuration

### Game Configuration

Edit `config/freeciv_config.py` to adjust:
- Map size and game settings
- Model configuration
- Parser settings
- Performance optimizations

### Demo Configuration

Environment variables for quick testing:
```bash
export NUM_MOVES=5
export PARSER_CHOICE=rule_then_soft
export FREECIV_MAP_SIZE=tiny
```

## Development Workflow

### 1. Local Development

```bash
# Edit FreeCiv components
vim game_arena/harness/freeciv_state.py
vim game_arena/harness/freeciv_client.py

# Test changes
python scripts/test_freeciv_connection.py

# Run demo
python scripts/run_game.py freeciv --num_moves=3
```

### 2. Docker Development

```bash
# Code changes are live-mounted in container
docker-compose exec game-arena python scripts/run_game.py freeciv

# Rebuild if dependencies change
docker-compose build
```

### 3. Testing Changes

```bash
# Test specific functionality
python -m unittest game_arena.harness.tests.test_freeciv_state

# Comprehensive testing
python scripts/test_all_games.py --verbose
```

## Expected Outputs

### Successful FreeCiv Game

```
FreeCiv Game - Turn 1
Phase: movement
Player: Player 1 (Romans)
Score: 0 | Gold: 50
Units: 1 | Cities: 0
Legal Actions: 2

Model player 0 main response: I will move my settlers to explore and find a good location for my first city. unit_move_settlers(101)_to(2,1)

Parser output is unit_move_settlers(101)_to(2,1).
Applied action: unit_move
```

### Connection Test Success

```
✓ Server status check passed (200)
✓ Game launcher endpoint accessible (200)
✓ WebSocket connection established
✓ FreeCiv client connected
✓ Game state retrieved
✓ FreeCiv state adapter created
✓ Legal actions retrieved: 2 actions
✓ Enhanced observation generated
```

## Performance Notes

- FreeCiv games are more complex than Chess/Go and may take longer
- State observations are token-limited for LLM efficiency
- Action space is reduced to most strategic options
- Connection caching reduces server load

## Next Steps

After successful testing:
1. Run longer games to test full game flow
2. Experiment with different LLM models and strategies
3. Configure tournament settings for multi-game analysis
4. Integrate with existing Game Arena tournament infrastructure