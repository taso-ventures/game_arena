# FreeCiv LLM Agent Deployment Guide

This guide provides comprehensive instructions for deploying and configuring the FreeCiv LLM Agent in production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Deployment Options](#deployment-options)
5. [Production Considerations](#production-considerations)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS 10.15+, or Windows 10+
- **Python**: 3.9 or higher
- **Memory**: Minimum 4GB RAM, 8GB+ recommended for production
- **Storage**: 2GB available disk space
- **Network**: Reliable internet connection for model API calls

### External Dependencies

- **FreeCiv3D Server**: Running instance accessible via WebSocket
- **Model API Access**: At least one of:
  - OpenAI API key (GPT models)
  - Google API key (Gemini models)
  - Anthropic API key (Claude models)

### Optional Dependencies

- **Docker**: For containerized deployment
- **Redis**: For distributed memory/caching (future enhancement)
- **PostgreSQL**: For persistent telemetry storage

## Installation

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/your-org/game_arena.git
cd game_arena

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Optional, for development

# Install game_arena in development mode
pip install -e .
```

### Docker Installation

```bash
# Build the container
docker build -t game-arena-freeciv .

# Or use docker-compose for full stack
docker-compose -f docker-compose.e2e.yml up
```

## Configuration

### Environment Variables

Set the following environment variables:

```bash
# Required: Model API Keys (at least one)
export OPENAI_API_KEY="your-openai-key"
export GEMINI_API_KEY="your-gemini-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Required: FreeCiv3D Server Configuration
export FREECIV_SERVER_URL="ws://localhost:8443"
export FREECIV_API_URL="http://localhost:8080"

# Optional: Agent Configuration
export FREECIV_AGENT_STRATEGY="balanced"  # aggressive_expansion, economic_focus, etc.
export FREECIV_AGENT_MODEL="gpt-4"       # Model to use
export FREECIV_AGENT_MEMORY_SIZE="10"    # Number of actions to remember
export FREECIV_AGENT_TIMEOUT="30"        # Seconds for action decisions

# Optional: Performance Tuning
export FREECIV_MAX_RETRIES="3"           # Model API retry attempts
export FREECIV_ENABLE_RETHINKING="true"  # Enable illegal move recovery
export FREECIV_MAX_RETHINKS="2"          # Maximum rethinking attempts

# Optional: Monitoring
export FREECIV_ENABLE_TELEMETRY="true"   # Enable performance monitoring
export FREECIV_LOG_LEVEL="INFO"          # DEBUG, INFO, WARNING, ERROR
```

### Configuration File

Create a configuration file `config/freeciv_agent.yaml`:

```yaml
# FreeCiv LLM Agent Configuration

# Model Configuration
model:
  name: "gpt-4"                    # Model to use
  api_key_env: "OPENAI_API_KEY"    # Environment variable for API key
  options:
    max_tokens: 1000
    temperature: 0.7
    timeout: 30

# Agent Configuration
agent:
  strategy: "balanced"             # Initial strategy
  memory_size: 10                  # Actions to remember
  enable_rethinking: true          # Handle illegal moves
  max_rethinks: 2                  # Rethinking attempts
  fallback_to_random: true         # Fallback for failures

# Server Configuration
server:
  websocket_url: "ws://localhost:8443"
  api_url: "http://localhost:8080"
  connection_timeout: 10
  reconnect_attempts: 3
  reconnect_delay: 5

# Performance Configuration
performance:
  max_concurrent_requests: 5       # Limit concurrent model calls
  request_timeout: 30              # Request timeout in seconds
  retry_attempts: 3                # Retry failed requests
  backoff_factor: 2                # Exponential backoff multiplier

# Monitoring Configuration
monitoring:
  enable_telemetry: true           # Performance monitoring
  telemetry_interval: 60           # Seconds between reports
  log_level: "INFO"                # Logging level
  log_file: "logs/freeciv_agent.log"

# Security Configuration
security:
  api_key_rotation_days: 30        # Days before key rotation warning
  max_memory_usage_mb: 1024        # Memory usage limit
  enable_action_validation: true   # Validate all actions
```

### Model-Specific Configuration

#### GPT Models (OpenAI)

```yaml
model:
  name: "gpt-4"
  api_key_env: "OPENAI_API_KEY"
  options:
    max_tokens: 1000
    temperature: 0.7
    top_p: 0.9
    frequency_penalty: 0
    presence_penalty: 0
```

#### Gemini Models (Google)

```yaml
model:
  name: "gemini-2.5-flash"
  api_key_env: "GEMINI_API_KEY"
  options:
    max_tokens: 1000
    temperature: 0.7
    top_p: 0.8
    top_k: 40
```

#### Claude Models (Anthropic)

```yaml
model:
  name: "claude-opus-4"
  api_key_env: "ANTHROPIC_API_KEY"
  options:
    max_tokens: 1000
    temperature: 0.7
```

## Deployment Options

### Local Development

```bash
# Start FreeCiv3D server (mock for testing)
docker-compose -f docker-compose.e2e.yml up freeciv3d

# Run agent
python3 -m game_arena.harness.freeciv_llm_agent \
  --config config/freeciv_agent.yaml \
  --player-id 1
```

### Production Deployment

#### Option 1: Direct Python

```bash
# Create production configuration
cp config/freeciv_agent.yaml config/production.yaml

# Edit production settings
nano config/production.yaml

# Run with production config
python3 -m game_arena.harness.freeciv_llm_agent \
  --config config/production.yaml \
  --player-id 1 \
  --log-level INFO \
  --daemon
```

#### Option 2: Docker Container

```dockerfile
# Production Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

# Create non-root user
RUN useradd -m -s /bin/bash gameagent
USER gameagent

CMD ["python3", "-m", "game_arena.harness.freeciv_llm_agent", "--config", "config/production.yaml"]
```

```bash
# Build and run
docker build -t freeciv-agent .
docker run -d \
  --name freeciv-agent-1 \
  --env-file .env \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  freeciv-agent
```

#### Option 3: Kubernetes

```yaml
# k8s/freeciv-agent.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: freeciv-agent
spec:
  replicas: 2
  selector:
    matchLabels:
      app: freeciv-agent
  template:
    metadata:
      labels:
        app: freeciv-agent
    spec:
      containers:
      - name: freeciv-agent
        image: freeciv-agent:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: model-api-keys
              key: openai-key
        - name: FREECIV_SERVER_URL
          value: "ws://freeciv-server:8443"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: config
        configMap:
          name: freeciv-config
      - name: logs
        persistentVolumeClaim:
          claimName: freeciv-logs
```

### Multi-Agent Tournament

```bash
# Run multiple agents for tournament
for i in {1..4}; do
  docker run -d \
    --name freeciv-agent-$i \
    --env FREECIV_PLAYER_ID=$i \
    --env-file .env \
    freeciv-agent &
done
```

## Production Considerations

### Security

1. **API Key Management**
   - Use environment variables or secure secret management
   - Rotate API keys regularly
   - Monitor API usage and costs

2. **Network Security**
   - Use HTTPS/WSS for production
   - Implement proper firewall rules
   - Consider VPN for internal communications

3. **Input Validation**
   - Validate all observation data
   - Sanitize action strings
   - Implement rate limiting

### Performance Optimization

1. **Memory Management**
   ```yaml
   agent:
     memory_size: 20              # Increase for better context
     enable_memory_compression: true
     memory_cleanup_interval: 300
   ```

2. **Model API Optimization**
   ```yaml
   model:
     options:
       max_tokens: 500            # Reduce for faster responses
       temperature: 0.3           # Lower for more deterministic play

   performance:
     max_concurrent_requests: 3   # Limit concurrent calls
     enable_request_batching: true
     cache_responses: true
   ```

3. **Network Optimization**
   ```yaml
   server:
     connection_pooling: true
     keep_alive: true
     compression: true
   ```

### Scalability

1. **Horizontal Scaling**
   - Run multiple agent instances
   - Use load balancer for game assignments
   - Implement distributed memory (Redis)

2. **Vertical Scaling**
   - Increase memory for larger game states
   - Use faster CPUs for quicker decisions
   - Optimize model inference

### Reliability

1. **Error Handling**
   ```yaml
   agent:
     fallback_to_random: true
     max_retries: 5
     circuit_breaker_threshold: 10
     health_check_interval: 30
   ```

2. **Monitoring**
   ```yaml
   monitoring:
     enable_metrics: true
     metrics_port: 8090
     health_endpoint: "/health"
     performance_alerts: true
   ```

## Monitoring and Maintenance

### Health Checks

```bash
# HTTP health check endpoint
curl http://localhost:8090/health

# Check agent status
curl http://localhost:8090/metrics

# View logs
tail -f logs/freeciv_agent.log
```

### Performance Monitoring

The agent includes built-in telemetry:

```python
# Access telemetry data
from game_arena.harness.freeciv_telemetry import TelemetryManager

telemetry = TelemetryManager()
metrics = telemetry.get_current_metrics()

print(f"Actions per minute: {metrics['actions_per_minute']}")
print(f"Average response time: {metrics['avg_response_time']}")
print(f"Success rate: {metrics['success_rate']}")
```

### Log Analysis

Key metrics to monitor:

- **Response Times**: Should be < 30 seconds per action
- **Success Rate**: Should be > 95% for legal actions
- **Memory Usage**: Should stay within configured limits
- **API Costs**: Monitor model API usage and costs

### Maintenance Tasks

1. **Daily**
   - Check agent health status
   - Review error logs
   - Monitor API usage

2. **Weekly**
   - Analyze performance trends
   - Update model parameters if needed
   - Clean up old logs

3. **Monthly**
   - Rotate API keys
   - Update dependencies
   - Review and optimize configuration

## Troubleshooting

### Common Issues

#### Agent Won't Connect to Server

```bash
# Check server availability
nc -z localhost 8443

# Verify URL configuration
echo $FREECIV_SERVER_URL

# Check network connectivity
ping freeciv-server-host
```

#### Model API Errors

```bash
# Check API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Check rate limits
grep "rate_limit" logs/freeciv_agent.log

# Test with different model
export FREECIV_AGENT_MODEL="gpt-3.5-turbo"
```

#### Performance Issues

```bash
# Check memory usage
ps aux | grep freeciv_agent

# Monitor network latency
ping -c 10 api.openai.com

# Review configuration
cat config/production.yaml | grep -E "(timeout|memory|concurrent)"
```

#### Action Validation Errors

```bash
# Check legal actions
grep "illegal_action" logs/freeciv_agent.log

# Verify game state
curl http://localhost:8080/api/game/state

# Test action parser
python3 -c "
from game_arena.harness.freeciv_action_converter import FreeCivActionConverter
converter = FreeCivActionConverter()
# Test specific action string
"
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
export FREECIV_LOG_LEVEL="DEBUG"
python3 -m game_arena.harness.freeciv_llm_agent \
  --config config/debug.yaml \
  --debug
```

### Support and Community

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check latest docs for updates
- **Community Forum**: Ask questions and share experiences
- **Email Support**: technical-support@game-arena.com

---

## Configuration Reference

### Complete Configuration Example

See `config/freeciv_agent_complete.yaml` for a full configuration file with all available options and their descriptions.

### Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | No* | - | OpenAI API key |
| `GEMINI_API_KEY` | No* | - | Google Gemini API key |
| `ANTHROPIC_API_KEY` | No* | - | Anthropic Claude API key |
| `FREECIV_SERVER_URL` | Yes | - | FreeCiv3D WebSocket URL |
| `FREECIV_API_URL` | Yes | - | FreeCiv3D HTTP API URL |
| `FREECIV_AGENT_STRATEGY` | No | balanced | Agent strategy |
| `FREECIV_AGENT_MODEL` | No | gpt-4 | Model to use |
| `FREECIV_AGENT_MEMORY_SIZE` | No | 10 | Memory size |
| `FREECIV_ENABLE_TELEMETRY` | No | true | Enable monitoring |
| `FREECIV_LOG_LEVEL` | No | INFO | Logging level |

*At least one model API key is required.

---

For questions or support, please refer to the project documentation or open an issue on GitHub.