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

"""Configuration management for FreeCiv parsers."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FreeCivParserConfig:
  """Centralized configuration for FreeCiv parser components.

  This class consolidates all configuration parameters used by FreeCiv parsers,
  allowing for easy customization and environment-based overrides.

  Attributes:
    max_input_size: Maximum input size in bytes for security protection
    regex_timeout_seconds: Timeout for regex operations to prevent hanging
    max_cache_size: Maximum number of entries in similarity cache
    similarity_threshold: Minimum similarity score for fuzzy matching
    token_weight: Weight for token overlap similarity (0.0-1.0)
    number_weight: Weight for number/ID overlap similarity (0.0-1.0)
    action_weight: Weight for action type similarity (0.0-1.0)
    edit_weight: Weight for edit distance similarity (0.0-1.0)
    exact_substring_bonus: Bonus for exact substring matches
    partial_token_bonus: Bonus for partial token matches
    action_match_bonus: Bonus for action type exact matches
    min_token_length_for_bonus: Minimum token length to qualify for bonus
    soft_match_min_score: Minimum score threshold for soft matching
    enable_performance_logging: Whether to log performance metrics
    enable_debug_logging: Whether to log debug information
  """

  # Security and performance limits
  max_input_size: int = 10 * 1024  # 10KB
  regex_timeout_seconds: float = 5.0
  max_cache_size: int = 1000

  # Similarity calculation weights (must sum to ≤ 1.0)
  similarity_threshold: float = 0.3
  token_weight: float = 0.3
  number_weight: float = 0.3
  action_weight: float = 0.3
  edit_weight: float = 0.1

  # Bonus score constants
  exact_substring_bonus: float = 0.15
  partial_token_bonus: float = 0.05
  action_match_bonus: float = 0.1

  # Threshold constants
  min_token_length_for_bonus: int = 3
  soft_match_min_score: int = 3

  # Logging configuration
  enable_performance_logging: bool = False
  enable_debug_logging: bool = False

  def __post_init__(self) -> None:
    """Validate configuration parameters."""
    if self.max_input_size <= 0:
      raise ValueError("max_input_size must be positive")

    if self.regex_timeout_seconds <= 0:
      raise ValueError("regex_timeout_seconds must be positive")

    if self.max_cache_size <= 0:
      raise ValueError("max_cache_size must be positive")

    # Validate similarity weights
    total_weight = (
        self.token_weight + self.number_weight +
        self.action_weight + self.edit_weight
    )
    if total_weight > 1.0:
      raise ValueError(
          f"Similarity weights sum to {total_weight:.2f}, must be ≤ 1.0"
      )

    if not 0.0 <= self.similarity_threshold <= 1.0:
      raise ValueError("similarity_threshold must be between 0.0 and 1.0")

  @classmethod
  def from_environment(cls, **overrides) -> 'FreeCivParserConfig':
    """Create configuration with environment variable overrides.

    Environment variables:
      FREECIV_PARSER_MAX_INPUT_SIZE: Override max_input_size
      FREECIV_PARSER_CACHE_SIZE: Override max_cache_size
      FREECIV_PARSER_SIMILARITY_THRESHOLD: Override similarity_threshold
      FREECIV_PARSER_ENABLE_DEBUG: Override enable_debug_logging
      FREECIV_PARSER_ENABLE_PERF_LOG: Override enable_performance_logging

    Args:
      **overrides: Direct parameter overrides

    Returns:
      FreeCivParserConfig instance with environment and override values

    Examples:
      >>> config = FreeCivParserConfig.from_environment()
      >>> config = FreeCivParserConfig.from_environment(max_cache_size=500)
    """
    env_overrides = {}

    # Parse environment variables
    if max_input := os.getenv('FREECIV_PARSER_MAX_INPUT_SIZE'):
      env_overrides['max_input_size'] = int(max_input)

    if cache_size := os.getenv('FREECIV_PARSER_CACHE_SIZE'):
      env_overrides['max_cache_size'] = int(cache_size)

    if threshold := os.getenv('FREECIV_PARSER_SIMILARITY_THRESHOLD'):
      env_overrides['similarity_threshold'] = float(threshold)

    if debug := os.getenv('FREECIV_PARSER_ENABLE_DEBUG'):
      env_overrides['enable_debug_logging'] = debug.lower() in ('1', 'true', 'yes')

    if perf_log := os.getenv('FREECIV_PARSER_ENABLE_PERF_LOG'):
      env_overrides['enable_performance_logging'] = perf_log.lower() in ('1', 'true', 'yes')

    # Combine environment and direct overrides
    combined_overrides = {**env_overrides, **overrides}

    return cls(**combined_overrides)

  def create_optimized_for_testing(self) -> 'FreeCivParserConfig':
    """Create a configuration optimized for testing scenarios.

    Returns:
      FreeCivParserConfig with smaller limits and debugging enabled
    """
    return FreeCivParserConfig(
        max_input_size=1024,  # 1KB for faster tests
        max_cache_size=10,    # Small cache for predictable behavior
        regex_timeout_seconds=1.0,  # Faster timeout for tests
        enable_debug_logging=True,
        enable_performance_logging=True
    )

  def create_production_optimized(self) -> 'FreeCivParserConfig':
    """Create a configuration optimized for production environments.

    Returns:
      FreeCivParserConfig with production-tuned parameters
    """
    return FreeCivParserConfig(
        max_input_size=50 * 1024,  # 50KB for larger inputs
        max_cache_size=5000,       # Larger cache for better performance
        regex_timeout_seconds=10.0,  # More lenient timeout
        enable_performance_logging=True,
        enable_debug_logging=False  # Reduce log noise in production
    )


# Default configuration instance
DEFAULT_CONFIG = FreeCivParserConfig()