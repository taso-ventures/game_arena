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

"""Configuration for FreeCiv games and testing."""

import os
from typing import Dict, Any


# FreeCiv Server Connection Settings
FREECIV_SERVER_CONFIG = {
    "server_url": os.getenv("FREECIV_SERVER_URL", "http://localhost:8080"),
    "ws_url": os.getenv("FREECIV_WS_URL", "ws://localhost:4002"),
    "timeout": 30.0,
    "retry_attempts": 3,
    "retry_delay": 2.0,
}

# MVP Game Configuration for Testing
MVP_GAME_CONFIG = {
    "map_size": "small",  # Options: tiny, small, medium, large
    "map_type": "continents",  # Options: continents, archipelago, island
    "ruleset": "classic",  # Options: classic, civ2, experimental
    "turn_limit": 200,  # Maximum turns before game ends
    "victory_conditions": ["conquest", "score"],
    "difficulty": "easy",
    "barbarians": "disabled",  # Disable for simpler testing
    "fog_of_war": True,
    "simultaneous_moves": False,  # Sequential turns for clearer analysis
}

# Model Configuration for FreeCiv
MODEL_CONFIG = {
    "player_one": {
        "model_type": "gemini",
        "model_name": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "api_key": os.getenv("GEMINI_API_KEY"),
        "strategy": "balanced",
        "temperature": 0.7,
        "max_tokens": 2000,
    },
    "player_two": {
        "model_type": "openai",
        "model_name": os.getenv("OPENAI_MODEL", "gpt-4.1"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "strategy": "aggressive",
        "temperature": 0.8,
        "max_tokens": 2000,
    },
}

# Parser Configuration
PARSER_CONFIG = {
    "default_parser": "rule_then_soft",  # Options: rule_then_soft, llm_only
    "enable_soft_matching": True,
    "soft_match_threshold": 0.3,
    "max_action_attempts": 3,
    "action_timeout": 30.0,
}

# Performance and Optimization Settings
PERFORMANCE_CONFIG = {
    "state_cache_ttl": 5.0,  # Cache game state for 5 seconds
    "observation_format": "enhanced",  # Options: json, ascii, enhanced
    "max_observation_tokens": 4000,
    "enable_action_space_reduction": True,
    "max_legal_actions": 20,  # Limit actions shown to LLM
    "enable_threat_analysis": True,
    "enable_strategic_analysis": True,
}

# Test Configuration
TEST_CONFIG = {
    "max_moves_per_test": 10,
    "enable_debug_output": True,
    "log_level": "INFO",
    "save_game_history": True,
    "save_observations": True,
    "test_timeout": 300.0,  # 5 minutes per test
}

# Full configuration combining all sections
FREECIV_CONFIG = {
    "server": FREECIV_SERVER_CONFIG,
    "game": MVP_GAME_CONFIG,
    "models": MODEL_CONFIG,
    "parser": PARSER_CONFIG,
    "performance": PERFORMANCE_CONFIG,
    "test": TEST_CONFIG,
}


def get_config(section: str = None) -> Dict[str, Any]:
    """Get configuration for a specific section or all configurations.

    Args:
        section: Configuration section name (server, game, models, parser, performance, test)
                If None, returns all configurations.

    Returns:
        Dictionary containing the requested configuration
    """
    if section is None:
        return FREECIV_CONFIG

    if section not in FREECIV_CONFIG:
        raise ValueError(f"Unknown configuration section: {section}")

    return FREECIV_CONFIG[section]


def validate_config() -> bool:
    """Validate that all required configuration values are present.

    Returns:
        True if configuration is valid, False otherwise
    """
    # Check API keys
    models_config = get_config("models")

    for player, config in models_config.items():
        if not config.get("api_key"):
            print(f"Warning: Missing API key for {player}")
            return False

    # Check server configuration
    server_config = get_config("server")
    required_server_fields = ["server_url", "ws_url"]

    for field in required_server_fields:
        if not server_config.get(field):
            print(f"Error: Missing required server configuration: {field}")
            return False

    print("Configuration validation passed")
    return True


def get_test_game_config() -> Dict[str, Any]:
    """Get configuration optimized for testing and debugging.

    Returns:
        Test-optimized game configuration
    """
    test_config = get_config("game").copy()
    test_config.update({
        "map_size": "tiny",  # Smallest map for faster testing
        "turn_limit": 50,    # Shorter games for testing
        "barbarians": "disabled",
        "fog_of_war": False,  # Full visibility for debugging
        "debug_mode": True,
    })

    return test_config


def get_demo_config() -> Dict[str, Any]:
    """Get configuration optimized for demonstrations.

    Returns:
        Demo-optimized configuration
    """
    demo_config = get_config().copy()
    demo_config["game"].update({
        "map_size": "small",
        "turn_limit": 100,
        "victory_conditions": ["conquest"],
    })

    demo_config["test"].update({
        "max_moves_per_test": 20,
        "enable_debug_output": True,
    })

    return demo_config


if __name__ == "__main__":
    # Configuration validation when run as script
    print("FreeCiv Configuration:")
    print("=" * 50)

    for section_name, section_config in FREECIV_CONFIG.items():
        print(f"\n{section_name.upper()}:")
        for key, value in section_config.items():
            # Hide API keys in output
            if "api_key" in key and value:
                value = f"{value[:8]}..."
            print(f"  {key}: {value}")

    print("\n" + "=" * 50)
    is_valid = validate_config()
    print(f"Configuration valid: {is_valid}")