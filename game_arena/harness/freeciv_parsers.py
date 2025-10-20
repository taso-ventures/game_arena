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

"""FreeCiv-specific parsers for move extraction and validation."""

import difflib
import hashlib
import json
import re
import signal
import threading
from typing import Any, Dict, List, Optional, TypedDict

from absl import logging

from game_arena.harness import llm_parsers, parsers
from game_arena.harness.freeciv_cache import LRUCache
from game_arena.harness.freeciv_parser_config import (DEFAULT_CONFIG,
                                                      FreeCivParserConfig)
from game_arena.harness.freeciv_state import FreeCivAction
from game_arena.harness.freeciv_state_stubs import FreeCivGameStateStub
from game_arena.harness.freeciv_timeout import TimeoutProtectedRegex


# TypedDict definitions for structured data
class TargetDict(TypedDict, total=False):
    """Type definition for action target information."""

    x: int  # X coordinate
    y: int  # Y coordinate
    id: int  # Target entity ID
    value: str  # Target value/name
    name: str  # Target name
    tech: str  # Technology name


class ParametersDict(TypedDict, total=False):
    """Type definition for action parameters."""

    direction: str  # Movement direction
    damage_type: str  # Attack damage type
    priority: int  # Action priority


class ParseMetrics(TypedDict):
    """Type definition for parser performance metrics."""

    parse_time_ms: float
    method_used: str
    cache_hit: bool
    input_size: int


# Custom exceptions for better error handling
class FreeCivParserError(Exception):
    """Base exception for FreeCiv parser errors."""

    pass


class FreeCivParserTimeoutError(FreeCivParserError):
    """Raised when parser operations timeout."""

    pass


class FreeCivParserInputError(FreeCivParserError):
    """Raised when input validation fails."""

    pass


# FreeCiv instruction config for LLM-based parsing
FreeCivInstructionConfig_V0 = llm_parsers.InstructionConfig(
    name="FreeCivInstructionConfig_V0",
    instruction="""## Instructions for Extracting Final Proposed FreeCiv Action

**Objective:** Given a response containing context and the final proposed FreeCiv action, extract the final proposed FreeCiv action without excess formatting.

**Process:**

1. **Analyze the response:** From the response, identify the context preceding the final proposed FreeCiv action and the final proposed FreeCiv action itself.
   If no final proposed FreeCiv action is present, skip the steps below and then present "Clean Action: LLMEXTRACT_NO_PROPOSED_ACTION" on a new line (no additional markings or explanations are needed).

2. **Extract the final proposed FreeCiv action:** Extract the final proposed FreeCiv action. FreeCiv actions have the following format:
   - Action type (e.g., unit_move, city_production, unit_attack, etc.)
   - Actor identifier (unit ID or city ID)
   - Target information (coordinates, target ID, or parameters)

   Examples of valid FreeCiv actions:
   - unit_move_settlers(101)_to(2,3)
   - city_production_athens(301)_target(warriors)
   - unit_attack_warrior(102)_target(203)
   - unit_fortify_legions(105)
   - city_build_improvement_rome(302)_target(granary)

3. **Remove excess formatting:** From the extracted final proposed FreeCiv action, remove excess formatting while preserving the action structure:
   - Remove LaTeX formatting, HTML tags, surrounding brackets
   - Remove excess newlines, leading whitespace, trailing whitespace
   - Remove terminating periods that are not part of the action
   - Keep underscores, parentheses, and numbers that are part of the action format

4. **Present the clean final proposed FreeCiv action:** Present the clean final proposed FreeCiv action on a new line, preceded by "Clean Action: ".

**Note:** No additional markings or explanations are needed beyond "Clean Action: " and the clean final proposed FreeCiv action.""",
    final_answer_prefix="Clean Action: ",
    no_action_answer="LLMEXTRACT_NO_PROPOSED_ACTION",
)


class FreeCivLLMParser(llm_parsers.LLMParser):
    """LLM-based parser for FreeCiv actions using natural language extraction."""

    def __init__(self, model):
        """Initialize FreeCiv LLM parser.

        Args:
            model: The LLM model to use for parsing
        """
        super().__init__(model, FreeCivInstructionConfig_V0)


class FreeCivRuleBasedParser(parsers.RuleBasedMoveParser):
    """Rule-based parser for FreeCiv action strings."""

    def __init__(self, config: Optional[FreeCivParserConfig] = None):
        """Initialize FreeCiv rule-based parser.

        Args:
          config: Parser configuration, uses default if None
        """
        super().__init__()
        self._config = config or DEFAULT_CONFIG
        # Initialize timeout-protected regex
        self._protected_regex = TimeoutProtectedRegex(
            self._config.regex_timeout_seconds
        )
        # Compile regex patterns for different action types
        self._action_patterns = {
            "unit_move": re.compile(r"unit_move_([^_]+)\((\d+)\)_to\((\d+),(\d+)\)"),
            "unit_attack": re.compile(r"unit_attack_([^_]+)\((\d+)\)_target\((\d+)\)"),
            "unit_fortify": re.compile(r"unit_fortify_([^_]+)\((\d+)\)"),
            "unit_explore": re.compile(r"unit_explore_([^_]+)\((\d+)\)"),
            "unit_build_improvement": re.compile(
                r"unit_build_improvement_([^_]+)\((\d+)\)_target\(([^)]+)\)"
            ),
            "city_production": re.compile(
                r"city_production_([^_]+)\((\d+)\)_target\(([^)]+)\)"
            ),
            "city_build_improvement": re.compile(
                r"city_build_improvement_([^_]+)\((\d+)\)_target\(([^)]+)\)"
            ),
            "city_celebrate": re.compile(r"city_celebrate_([^_]+)\((\d+)\)"),
        }

    def parse(self, parser_input: parsers.TextParserInput) -> Optional[str]:
        """Parse FreeCiv action from text input with enhanced JSON support.

        Args:
            parser_input: Text input containing the action to parse

        Returns:
            Parsed action string if successful, None otherwise
        """
        import time

        start_time = time.perf_counter()
        text = parser_input.text.strip()

        # Security: Validate input size to prevent DoS attacks
        if len(text) > self._config.max_input_size:
            if self._config.enable_debug_logging:
                logging.debug(
                    "Input size %d exceeds maximum %d",
                    len(text),
                    self._config.max_input_size,
                )
            return None

        # Security: Validate individual line lengths to prevent memory exhaustion
        for i, line in enumerate(text.split("\n")):
            if len(line) > self._config.max_line_length:
                if self._config.enable_debug_logging:
                    logging.debug(
                        "Line %d length %d exceeds maximum %d",
                        i + 1,
                        len(line),
                        self._config.max_line_length,
                    )
                return None

        # Try exact canonical format FIRST (most reliable when LLM follows format)
        canonical_action = self._try_parse_canonical_format(text)
        if canonical_action:
            if self._config.enable_performance_logging:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logging.info(
                    "Parse completed: method=canonical_format, time=%.2fms, input_size=%d,"
                    " success=True",
                    elapsed_ms,
                    len(text),
                )
            return canonical_action

        # Try JSON parsing (most structured)
        json_action = self._try_parse_json(text)
        if json_action:
            if self._config.enable_performance_logging:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logging.info(
                    "Parse completed: method=json, time=%.2fms, input_size=%d,"
                    " success=True",
                    elapsed_ms,
                    len(text),
                )
            return json_action

        # Try Python repr parsing (handles LLM-generated dict formats)
        python_repr_action = self._try_parse_python_repr(text)
        if python_repr_action:
            if self._config.enable_performance_logging:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logging.info(
                    "Parse completed: method=python_repr, time=%.2fms, input_size=%d,"
                    " success=True",
                    elapsed_ms,
                    len(text),
                )
            return python_repr_action

        # Try natural language parsing using FreeCivAction model
        natural_action = self._try_parse_natural_language(text, parser_input)
        if natural_action:
            if self._config.enable_performance_logging:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logging.info(
                    "Parse completed: method=natural_language, time=%.2fms,"
                    " input_size=%d, success=True",
                    elapsed_ms,
                    len(text),
                )
            return natural_action

        # Fall back to existing regex-based parsing
        # First try matching the original text (in case it's already well-formed)
        for action_type, pattern in self._action_patterns.items():
            try:
                match = self._protected_regex.search(pattern, text)
                if match:
                    # Return the original text if it already matches perfectly
                    if self._config.enable_performance_logging:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        logging.info(
                            "Parse completed: method=regex_direct, time=%.2fms,"
                            " input_size=%d, success=True",
                            elapsed_ms,
                            len(text),
                        )
                    return text
            except Exception as e:
                if self._config.enable_debug_logging:
                    logging.debug(
                        "Regex timeout in direct match for %s: %s", action_type, e
                    )
                continue

        # If no direct match, try with cleaned text
        cleaned_text = self._clean_text(text)
        for action_type, pattern in self._action_patterns.items():
            try:
                match = self._protected_regex.search(pattern, cleaned_text)
                if match:
                    # Reconstruct the action string in canonical format
                    result = self._reconstruct_action(action_type, match.groups())
                    if self._config.enable_performance_logging:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        logging.info(
                            "Parse completed: method=regex_cleaned, time=%.2fms,"
                            " input_size=%d, success=True",
                            elapsed_ms,
                            len(text),
                        )
                    return result
            except Exception as e:
                if self._config.enable_debug_logging:
                    logging.debug(
                        "Regex timeout in cleaned match for %s: %s", action_type, e
                    )
                continue

        # If no pattern matches, try soft matching against legal moves
        result = None
        method_used = "none"
        if parser_input.legal_moves:
            result = self._soft_match(cleaned_text, parser_input.legal_moves)
            if result:
                method_used = "soft_match"

        # Log performance metrics
        if self._config.enable_performance_logging:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logging.info(
                "Parse completed: method=%s, time=%.2fms, input_size=%d, success=%s",
                method_used,
                elapsed_ms,
                len(text),
                result is not None,
            )

        return result

    def _try_parse_canonical_format(self, text: str) -> Optional[str]:
        """Try to extract exact canonical format action strings.

        This method looks for canonical action format strings that match the
        expected FreeCiv action syntax. When the LLM outputs the exact canonical
        format (as instructed in prompts), this should match and return it directly.

        Looks for patterns like:
        - tech_research_player(1)_target(Alphabet)
        - unit_move_unit(101)_to(2,3)
        - city_production_city(5)_target(Warrior)

        Args:
            text: Raw text that might contain canonical format

        Returns:
            Canonical action string if found, None otherwise
        """
        import re

        # Security: Check input size
        if len(text) > self._config.max_input_size:
            return None

        # Comprehensive canonical format patterns
        # Order matters - more specific patterns first
        canonical_patterns = [
            # Tech research: tech_research_player(1)_target(Alphabet)
            r'(tech_research_player\(\d+\)_target\([A-Za-z_\s]+\))',

            # Unit movement: unit_move_unit(101)_to(2,3)
            r'(unit_move_(?:unit|warrior|settler|worker)\(\d+\)_to\(\d+,\s*\d+\))',

            # Unit attack: unit_attack_unit(101)_target(202)
            r'(unit_attack_(?:unit|warrior|settler)\(\d+\)_target\(\d+\))',

            # City production: city_production_city(5)_target(Warrior)
            r'(city_production_(?:city|[A-Za-z]+)\(\d+\)_target\([A-Za-z_\s]+\))',

            # City improvement: city_build_improvement_city(5)_target(Barracks)
            r'(city_build_improvement_(?:city|[A-Za-z]+)\(\d+\)_target\([A-Za-z_\s]+\))',

            # Unit fortify: unit_fortify_unit(101)
            r'(unit_fortify_(?:unit|warrior)\(\d+\))',

            # Unit explore: unit_explore_unit(101)
            r'(unit_explore_(?:unit|scout)\(\d+\))',
        ]

        for pattern in canonical_patterns:
            try:
                match = self._protected_regex.search(pattern, text)
                if match:
                    canonical_string = match.group(1)
                    if self._config.enable_performance_logging:
                        logging.info("Found canonical format: %s", canonical_string)
                    return canonical_string
            except Exception as e:
                if self._config.enable_debug_logging:
                    logging.debug("Canonical pattern matching error: %s", e)
                continue

        return None

    def _try_parse_json(self, text: str) -> Optional[str]:
        """Try to parse JSON from text and convert to canonical action string.

        Args:
            text: Raw text that might contain JSON

        Returns:
            Canonical action string if JSON parsing succeeds, None otherwise
        """
        # Security: Check input size before regex processing
        if len(text) > self._config.max_input_size:
            return None

        # Look for JSON-like patterns in the text (with size limits)
        json_patterns = [
            r"\{[^}]{1,1000}\}",  # Simple JSON object (max 1000 chars)
            r"\{[^{}]{0,500}\{[^}]{0,500}\}[^{}]{0,500}\}",  # Nested JSON (limited nesting)
        ]

        for pattern in json_patterns:
            try:
                json_matches = self._protected_regex.findall(pattern, text, re.DOTALL)
                for json_match in json_matches:
                    # Security: Skip overly large JSON matches
                    if len(json_match) > self._config.max_input_size // 10:
                        if self._config.enable_debug_logging:
                            logging.debug(
                                "JSON match too large (%d chars), skipping",
                                len(json_match),
                            )
                        continue

                    try:
                        # Try to parse JSON and convert to FreeCivAction
                        action = FreeCivAction.from_json(json_match)
                        return self._action_to_canonical_string(action)
                    except json.JSONDecodeError as e:
                        if self._config.enable_debug_logging:
                            logging.debug("JSON decode error in pattern match: %s", e)
                        continue  # Try next JSON match
                    except (ValueError, TypeError) as e:
                        if self._config.enable_debug_logging:
                            logging.debug(
                                "Value/Type error in JSON action creation: %s", e
                            )
                        continue  # Try next JSON match
            except Exception as e:
                if self._config.enable_debug_logging:
                    logging.debug("Regex timeout or error in JSON parsing: %s", e)
                continue  # Try next pattern

        return None

    def _try_parse_python_repr(self, text: str) -> Optional[str]:
        """Try to parse Python dict representation from LLM output.

        Handles formats like:
        action_type='tech_research' actor_id=1 target={'value': 'Alphabet'}

        Args:
            text: Raw text that might contain Python repr format

        Returns:
            Canonical action string if parsing succeeds, None otherwise
        """
        import re
        import ast

        # Security: Check input size
        if len(text) > self._config.max_input_size:
            return None

        # Valid action types allowlist for security
        VALID_ACTION_TYPES = {
            'tech_research',
            'unit_move',
            'unit_attack',
            'unit_fortify',
            'unit_explore',
            'city_production',
            'city_build_improvement',
            'end_turn',
        }

        try:
            # Pattern to match Python-like attribute assignments
            # Matches: action_type='tech_research' actor_id=1 target={'value': 'Alphabet'}
            pattern = (
                r"action_type\s*=\s*['\"](\w+)['\"]"
                r".*?"
                r"actor_id\s*=\s*(\d+)"
                r".*?"
                r"target\s*=\s*(\{[^}]+\})"
            )

            match = re.search(pattern, text, re.DOTALL)
            if match:
                action_type = match.group(1)
                actor_id = int(match.group(2))
                target_str = match.group(3)

                # Security: Validate action type is in allowlist
                if action_type not in VALID_ACTION_TYPES:
                    logging.debug(f"Invalid action type: {action_type}")
                    return None

                # Security: Limit target string size (reduced from 500 to 200)
                if len(target_str) > 200:
                    logging.debug("Target string too large, skipping Python repr parse")
                    return None

                # Safely parse the target dict using ast.literal_eval
                try:
                    target = ast.literal_eval(target_str)
                except (ValueError, SyntaxError) as e:
                    logging.debug(f"Failed to parse target dict: {e}")
                    return None

                # Security: Validate target is a dict with expected structure
                if not isinstance(target, dict):
                    logging.debug(f"Target is not a dict: {type(target)}")
                    return None

                # Security: Limit dict size
                if len(target) > 10:
                    logging.debug(f"Target dict too large: {len(target)} keys")
                    return None

                # Security: Validate all values are simple types (str, int, float)
                for key, value in target.items():
                    if not isinstance(value, (str, int, float, type(None))):
                        logging.debug(f"Target contains invalid type: {type(value)}")
                        return None
                    # Limit string values
                    if isinstance(value, str) and len(value) > 100:
                        logging.debug(f"Target string value too long: {len(value)}")
                        return None

                # Convert to canonical format based on action type
                if action_type == "tech_research" and isinstance(target, dict):
                    tech_name = target.get('value', target.get('name', ''))
                    if tech_name and isinstance(tech_name, str):
                        return f"tech_research_player({actor_id})_target({tech_name})"

                elif action_type == "city_production" and isinstance(target, dict):
                    production = target.get('value', target.get('name', ''))
                    if production and isinstance(production, str):
                        return f"city_production_city({actor_id})_target({production})"

                elif action_type == "unit_move" and isinstance(target, dict):
                    x = target.get('x')
                    y = target.get('y')
                    if x is not None and y is not None and isinstance(x, int) and isinstance(y, int):
                        return f"unit_move_unit({actor_id})_to({x},{y})"

                elif action_type == "unit_attack" and isinstance(target, dict):
                    target_id = target.get('id', target.get('value'))
                    if target_id is not None and isinstance(target_id, int):
                        return f"unit_attack_unit({actor_id})_target({target_id})"

                # Generic fallback for other action types (with strict validation)
                if isinstance(target, dict) and 'value' in target:
                    value = target['value']
                    if isinstance(value, (str, int)):
                        return f"{action_type}_player({actor_id})_target({value})"

        except Exception as e:
            if self._config.enable_debug_logging:
                logging.debug(f"Python repr parsing failed: {e}")

        return None

    def _try_parse_natural_language(
        self, text: str, parser_input: parsers.TextParserInput
    ) -> Optional[str]:
        """Try to parse natural language using FreeCivAction model.

        Args:
            text: Natural language text
            parser_input: Parser input with context

        Returns:
            Canonical action string if parsing succeeds, None otherwise
        """
        try:
            # Create a minimal stub game state for natural language parsing
            stub_state = self._create_stub_game_state()

            # Try to parse using FreeCivAction natural language parser
            action = FreeCivAction.from_natural_language(text, stub_state)

            if action:
                return self._action_to_canonical_string(action)

            return None
        except Exception as e:
            # If natural language parsing fails, return None to try other methods
            if self._config.enable_debug_logging:
                logging.debug("Natural language parsing failed: %s", e)
            return None

    def _create_stub_game_state(self) -> FreeCivGameStateStub:
        """Create a minimal game state stub for natural language parsing.

        Returns:
            FreeCivGameStateStub with sample units and cities for parsing
        """
        return FreeCivGameStateStub()

    def _action_to_canonical_string(self, action: FreeCivAction) -> str:
        """Convert FreeCivAction to canonical action string format.

        Args:
            action: FreeCivAction object

        Returns:
            Canonical action string compatible with existing system
        """
        # Map action to the existing string format expected by the system
        if action.action_type == "unit_move" and action.target:
            return f"unit_move_{action.source}({action.actor_id})_to({action.target['x']},{action.target['y']})"

        elif action.action_type == "unit_attack" and action.target:
            return f"unit_attack_{action.source}({action.actor_id})_target({action.target['id']})"

        elif action.action_type in ["unit_fortify", "unit_explore"]:
            return f"{action.action_type}_{action.source}({action.actor_id})"

        elif action.action_type == "city_production" and action.target:
            target_value = (
                action.target.get("value")
                or action.target.get("name")
                or action.target.get("id")
            )
            return f"city_production_{action.source}({action.actor_id})_target({target_value})"

        elif action.action_type == "city_build_improvement" and action.target:
            target_value = (
                action.target.get("value")
                or action.target.get("name")
                or action.target.get("id")
            )
            return f"city_build_improvement_{action.source}({action.actor_id})_target({target_value})"

        elif action.action_type == "tech_research" and action.target:
            target_value = (
                action.target.get("value")
                or action.target.get("name")
                or action.target.get("tech")
            )
            return f"tech_research_player({action.actor_id})_target({target_value})"

        # Generic format
        target_str = ""
        if action.target:
            if "x" in action.target and "y" in action.target:
                target_str = f"_to({action.target['x']},{action.target['y']})"
            elif "id" in action.target:
                target_str = f"_target({action.target['id']})"
            elif "value" in action.target:
                target_str = f"_target({action.target['value']})"

        return f"{action.action_type}_{action.source}({action.actor_id}){target_str}"

    def _clean_text(self, text: str) -> str:
        """Clean text to extract action string.

        Args:
            text: Raw text from LLM response

        Returns:
            Cleaned text suitable for pattern matching
        """
        # Remove common prefixes and suffixes
        text = re.sub(
            r"^(I will|I choose|My action is|Action:|Move:)\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s*(\.|\!|\?)$", "", text)

        # Remove quotes and brackets
        text = re.sub(r'^["\'\[\(]|["\'\]\)]$', "", text.strip())

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Extract action-like patterns
        action_match = re.search(r"(unit_\w+|city_\w+).*?(?=\.|$)", text, re.IGNORECASE)
        if action_match:
            return action_match.group(0)

        return text

    def _reconstruct_action(self, action_type: str, groups: tuple) -> str:
        """Reconstruct canonical action string from regex groups.

        Args:
            action_type: Type of action (unit_move, city_production, etc.)
            groups: Regex capture groups

        Returns:
            Canonical action string
        """
        if action_type == "unit_move":
            unit_type, unit_id, x, y = groups
            return f"unit_move_{unit_type}({unit_id})_to({x},{y})"
        elif action_type == "unit_attack":
            unit_type, unit_id, target_id = groups
            return f"unit_attack_{unit_type}({unit_id})_target({target_id})"
        elif action_type in ["unit_fortify", "unit_explore"]:
            unit_type, unit_id = groups
            return f"{action_type}_{unit_type}({unit_id})"
        elif action_type in [
            "unit_build_improvement",
            "city_production",
            "city_build_improvement",
        ]:
            actor_type, actor_id, target = groups
            return f"{action_type}_{actor_type}({actor_id})_target({target})"
        elif action_type == "city_celebrate":
            city_type, city_id = groups
            return f"city_celebrate_{city_type}({city_id})"
        else:
            # Generic reconstruction
            return f"{action_type}_" + "_".join(str(g) for g in groups)

    def _soft_match(self, text: str, legal_moves: List[str]) -> Optional[str]:
        """Perform soft matching against legal moves.

        Args:
            text: Cleaned text to match
            legal_moves: List of legal action strings

        Returns:
            Best matching legal move or None
        """
        text_lower = text.lower()

        # Extract key components for matching
        keywords = set(re.findall(r"\w+", text_lower))
        numbers = set(re.findall(r"\d+", text))

        best_match = None
        best_score = 0

        for move in legal_moves:
            move_lower = move.lower()
            # Split on underscores and parentheses to get individual words
            move_keywords = set(re.findall(r"\w+", move_lower.replace("_", " ")))
            move_numbers = set(re.findall(r"\d+", move))

            # Calculate similarity score
            keyword_overlap = len(keywords & move_keywords)
            number_overlap = len(numbers & move_numbers)

            # Weight keywords and numbers differently
            score = keyword_overlap * 2 + number_overlap * 3

            # Bonus for exact substring match
            if text_lower in move_lower or move_lower in text_lower:
                score += 5

            if score > best_score:
                best_score = score
                best_match = move

        # Only return if we have a reasonable confidence
        if best_score >= 3:
            return best_match

        return None


class FreeCivSoftParser(parsers.SoftMoveParser):
    """Soft parser for FreeCiv actions using enhanced fuzzy matching."""

    def __init__(self, config: Optional[FreeCivParserConfig] = None):
        """Initialize FreeCiv soft parser with caching.

        Args:
          config: Parser configuration, uses default if None
        """
        super().__init__("freeciv")
        self._config = config or DEFAULT_CONFIG
        # Use configurable cache settings
        from game_arena.harness.freeciv_proxy_client import DEFAULT_MEMORY_CACHE_TTL
        self._similarity_cache = LRUCache[str, float](
            max_size=self._config.max_cache_size, ttl_seconds=DEFAULT_MEMORY_CACHE_TTL
        )
        self._legal_moves_hash = None  # Track when legal moves change
        self._cache_lock = threading.RLock()  # Thread safety for cache operations

    def parse(self, parser_input: parsers.TextParserInput) -> Optional[str]:
        """Parse FreeCiv action using soft matching.

        Args:
            parser_input: Text input containing the action to parse

        Returns:
            Best matching legal action or None
        """
        if not parser_input.legal_moves:
            return None

        # Check if legal moves changed and invalidate cache if needed (thread-safe)
        current_moves_hash = self._hash_legal_moves(parser_input.legal_moves)
        with self._cache_lock:
            if self._legal_moves_hash != current_moves_hash:
                self._similarity_cache.clear()
                self._legal_moves_hash = current_moves_hash
                if self._config.enable_debug_logging:
                    logging.debug("Cache cleared due to legal moves change")

        text = parser_input.text.strip().lower()

        # Security: Validate input size
        if len(text) > self._config.max_input_size:
            if self._config.enable_debug_logging:
                logging.debug(
                    "Input size %d exceeds maximum %d",
                    len(text),
                    self._config.max_input_size,
                )
            return None

        # Security: Validate individual line lengths to prevent memory exhaustion
        for i, line in enumerate(text.split("\n")):
            if len(line) > self._config.max_line_length:
                if self._config.enable_debug_logging:
                    logging.debug(
                        "Line %d length %d exceeds maximum %d",
                        i + 1,
                        len(line),
                        self._config.max_line_length,
                    )
                return None

        # Remove common noise words and formatting
        noise_patterns = [
            r"\b(i|will|choose|my|action|is|move|the)\b",
            r"[^\w\s\(\),]",  # Remove special chars except parentheses and commas
            r"\s+",  # Normalize whitespace
        ]

        cleaned_text = text
        for pattern in noise_patterns:
            cleaned_text = re.sub(
                pattern, " " if pattern == r"\s+" else "", cleaned_text
            )
        cleaned_text = cleaned_text.strip()

        # Score each legal move
        best_match = None
        best_score = 0

        for move in parser_input.legal_moves:
            score = self._calculate_similarity_cached(cleaned_text, move.lower())
            if score > best_score:
                best_score = score
                best_match = move

        # Return match if confidence is high enough
        if best_score > self._config.similarity_threshold:
            if self._config.enable_debug_logging:
                logging.debug(
                    "Found match '%s' with score %.3f", best_match, best_score
                )
            return best_match

        if self._config.enable_debug_logging:
            logging.debug(
                "No match found, best score %.3f below threshold %.3f",
                best_score,
                self._config.similarity_threshold,
            )
        return None

    def _calculate_similarity_cached(self, text: str, move: str) -> float:
        """Calculate similarity with caching for performance.

        Args:
            text: Cleaned input text
            move: Legal move string

        Returns:
            Cached similarity score between 0 and 1
        """
        # Generate efficient cache key using hash to reduce memory usage
        cache_key = f"sim|{hash(text)}|{hash(move)}"

        # Check cache first (thread-safe)
        with self._cache_lock:
            cached_result = self._similarity_cache.get(cache_key)
            if cached_result is not None:
                if self._config.enable_performance_logging:
                    logging.debug("Cache hit for similarity calculation")
                return cached_result

        # Calculate similarity (outside lock to avoid holding lock during computation)
        similarity = self._calculate_enhanced_similarity(text, move)

        # Cache result (thread-safe)
        with self._cache_lock:
            self._similarity_cache.set(cache_key, similarity)

        if self._config.enable_performance_logging:
            logging.debug(
                "Calculated similarity %.3f for '%s' vs '%s'", similarity, text, move
            )

        return similarity

    def _hash_legal_moves(self, legal_moves: List[str]) -> str:
        """Create a hash of legal moves for cache invalidation.

        Args:
            legal_moves: List of legal move strings

        Returns:
            SHA256 hash of the sorted legal moves
        """
        moves_str = "|".join(sorted(legal_moves))
        return hashlib.sha256(moves_str.encode()).hexdigest()[
            :32
        ]  # Use 32 chars for better collision resistance

    def invalidate_cache(self) -> None:
        """Manually invalidate the similarity cache."""
        with self._cache_lock:
            old_stats = self._similarity_cache.statistics
            self._similarity_cache.clear()
            self._legal_moves_hash = None
            if self._config.enable_debug_logging:
                logging.debug(
                    "Cache invalidated manually. Previous stats: %s", old_stats
                )

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get current cache statistics for monitoring.

        Returns:
            Dictionary with cache performance metrics
        """
        with self._cache_lock:
            return self._similarity_cache.statistics

    def _calculate_enhanced_similarity(self, text: str, move: str) -> float:
        """Enhanced similarity calculation with multiple algorithms.

        This method combines multiple similarity metrics to provide robust
        fuzzy matching between user input and legal moves. The algorithm
        uses weighted scoring across different dimensions:

        1. Token similarity: Overlap of words/tokens
        2. Number similarity: Overlap of numeric IDs
        3. Action similarity: Overlap of action keywords
        4. Edit distance: Character-level similarity

        Additional bonuses are applied for:
        - Exact substring matches
        - Partial token matches
        - Action type matches

        Args:
            text: Cleaned input text from user
            move: Legal move string from game state

        Returns:
            Similarity score between 0.0 and 1.0, where:
            - 0.0: No similarity
            - 0.3: Minimum threshold for matching (configurable)
            - 0.6+: High confidence match
            - 1.0: Perfect match

        Examples:
            >>> config = FreeCivParserConfig()
            >>> parser = FreeCivSoftParser(config)
            >>> score = parser._calculate_enhanced_similarity(
            ...     "move settlers 101",
            ...     "unit_move_settlers(101)_to(2,3)"
            ... )
            >>> score > 0.6  # High confidence due to token + number match
            True

            >>> score = parser._calculate_enhanced_similarity(
            ...     "attack warriors",
            ...     "unit_attack_warriors(102)_target(203)"
            ... )
            >>> score > 0.3  # Above threshold due to action + token match
            True

            >>> score = parser._calculate_enhanced_similarity(
            ...     "fortfy unit",  # typo
            ...     "unit_fortify_legion(105)"
            ... )
            >>> score > 0.2  # Edit distance helps with typos
            True
        """
        # Calculate individual similarity components
        token_similarity = self._calculate_token_similarity(text, move)
        number_similarity = self._calculate_number_similarity(text, move)
        action_similarity = self._calculate_action_similarity(text, move)
        edit_similarity = self._calculate_edit_similarity_optimized(text, move)

        # Build weighted similarity list
        similarities = []
        if token_similarity > 0:
            similarities.append(("token", token_similarity, self._config.token_weight))
        if number_similarity > 0:
            similarities.append(
                ("number", number_similarity, self._config.number_weight)
            )
        if action_similarity > 0:
            similarities.append(
                ("action", action_similarity, self._config.action_weight)
            )

        similarities.append(("edit", edit_similarity, self._config.edit_weight))

        # Calculate weighted average
        if similarities:
            weighted_sum = sum(score * weight for _, score, weight in similarities)
            total_weight = sum(weight for _, _, weight in similarities)
            base_similarity = weighted_sum / total_weight
        else:
            base_similarity = 0.0

        # Apply bonus adjustments
        bonus = self._calculate_bonus_adjustments(text, move)
        final_similarity = min(base_similarity + bonus, 1.0)

        return final_similarity

    def _calculate_token_similarity(self, text: str, move: str) -> float:
        """Calculate token overlap similarity using Jaccard coefficient.

        Extracts words from both strings and calculates the ratio of
        overlapping tokens to total unique tokens.

        Args:
            text: Cleaned input text (split by spaces)
            move: Legal move string (split by underscores)

        Returns:
            Token similarity score between 0 and 1

        Example:
            >>> parser._calculate_token_similarity("move settlers", "unit_move_settlers(101)")
            0.67  # 2 matches out of 3 unique tokens
        """
        text_tokens = set(text.split())
        move_tokens = set(move.lower().split("_"))

        if not text_tokens and not move_tokens:
            return 0.0

        token_overlap = len(text_tokens & move_tokens)
        total_tokens = len(text_tokens | move_tokens)
        return token_overlap / max(total_tokens, 1)

    def _calculate_number_similarity(self, text: str, move: str) -> float:
        """Calculate number/ID overlap similarity.

        Args:
            text: Cleaned input text
            move: Legal move string

        Returns:
            Number similarity score between 0 and 1
        """
        text_numbers = set(re.findall(r"\d+", text))
        move_numbers = set(re.findall(r"\d+", move))

        if not text_numbers and not move_numbers:
            return 0.0

        number_overlap = len(text_numbers & move_numbers)
        total_numbers = len(text_numbers | move_numbers)
        return number_overlap / max(total_numbers, 1)

    def _calculate_action_similarity(self, text: str, move: str) -> float:
        """Calculate action type similarity.

        Args:
            text: Cleaned input text
            move: Legal move string

        Returns:
            Action similarity score between 0 and 1
        """
        # More flexible pattern that matches action words in different contexts
        action_pattern = r"(move|attack|build|fortify|explore|research|produce)"
        text_actions = set(re.findall(action_pattern, text.lower()))
        move_actions = set(re.findall(action_pattern, move.lower()))

        if not text_actions and not move_actions:
            return 0.0

        action_overlap = len(text_actions & move_actions)
        total_actions = len(text_actions | move_actions)
        return action_overlap / max(total_actions, 1)

    def _calculate_bonus_adjustments(self, text: str, move: str) -> float:
        """Calculate bonus adjustments for similarity score.

        Args:
            text: Cleaned input text
            move: Legal move string

        Returns:
            Bonus score to add to base similarity
        """
        bonus = 0.0
        text_tokens = set(text.split())
        # More flexible pattern that matches action words in different contexts
        action_pattern = r"(move|attack|build|fortify|explore|research|produce)"
        text_actions = set(re.findall(action_pattern, text.lower()))
        move_actions = set(re.findall(action_pattern, move.lower()))

        # Exact substring match bonus
        if text in move or move in text:
            bonus += self._config.exact_substring_bonus

        # Partial substring matches for longer tokens
        for token in text_tokens:
            if len(token) > self._config.min_token_length_for_bonus and token in move:
                bonus += self._config.partial_token_bonus
                break

        # Action type exact match bonus
        if text_actions & move_actions:
            bonus += self._config.action_match_bonus

        return bonus

    def _calculate_edit_similarity_optimized(self, text: str, move: str) -> float:
        """Calculate edit distance similarity using optimized approach.

        Args:
            text: First text string
            move: Second text string

        Returns:
            Similarity score based on edit distance (0-1)
        """
        # Use difflib for better performance on longer strings
        if len(text) > 50 or len(move) > 50:
            return self._calculate_difflib_similarity(text, move)
        else:
            return self._calculate_edit_similarity(text, move)

    def _calculate_difflib_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using difflib.SequenceMatcher.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Similarity ratio from difflib
        """
        # Normalize strings for comparison
        s1 = re.sub(r"[^\w]", "", text1.lower())
        s2 = re.sub(r"[^\w]", "", text2.lower())

        if not s1 or not s2:
            return 0.0

        matcher = difflib.SequenceMatcher(None, s1, s2)
        return matcher.ratio()

    def _calculate_edit_similarity(self, text1: str, text2: str) -> float:
        """Calculate edit distance similarity using classic algorithm.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Similarity score based on edit distance (0-1)
        """
        # Normalize strings for comparison
        s1 = re.sub(r"[^\w]", "", text1.lower())
        s2 = re.sub(r"[^\w]", "", text2.lower())

        if not s1 or not s2:
            return 0.0

        # Use memoized edit distance calculation
        distance = self._edit_distance_memoized(s1, s2)
        max_len = max(len(s1), len(s2))

        # Convert to similarity (1 - normalized distance)
        similarity = 1.0 - (distance / max_len)
        return max(0.0, similarity)

    def _edit_distance_memoized(self, s1: str, s2: str) -> int:
        """Calculate edit distance with memoization for better performance.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Edit distance between the strings
        """
        # Use separate cache key prefix and hashes for efficiency
        cache_key = f"edit|{hash(s1)}|{hash(s2)}"
        with self._cache_lock:
            cached_result = self._similarity_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        # Ensure s1 is the longer string
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        if len(s2) == 0:
            result = len(s1)
        else:
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            result = previous_row[-1]

        # Cache the result using LRU cache
        with self._cache_lock:
            self._similarity_cache.set(cache_key, result)
        return result


def create_freeciv_parser_chain(
    model=None, config: Optional[FreeCivParserConfig] = None
) -> parsers.ChainedMoveParser:
    """Create a chained parser for FreeCiv actions.

    Args:
        model: Optional LLM model for natural language parsing
        config: Parser configuration, uses default if None

    Returns:
        ChainedMoveParser configured for FreeCiv with appropriate parsers

    Examples:
        >>> # Basic chain with default config
        >>> chain = create_freeciv_parser_chain()

        >>> # Chain with custom config
        >>> config = FreeCivParserConfig(max_cache_size=500)
        >>> chain = create_freeciv_parser_chain(config=config)

        >>> # Chain with LLM and custom config
        >>> chain = create_freeciv_parser_chain(model=my_model, config=config)
    """
    effective_config = config or DEFAULT_CONFIG
    parser_list = [FreeCivRuleBasedParser(effective_config)]

    # Add LLM parser if model is provided
    if model is not None:
        parser_list.append(FreeCivLLMParser(model))

    # Always add soft parser as fallback
    parser_list.append(FreeCivSoftParser(effective_config))

    return parsers.ChainedMoveParser(parser_list)
