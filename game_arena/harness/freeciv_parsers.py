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
from typing import Any, Dict, List, Optional

from game_arena.harness import llm_parsers, parsers
from game_arena.harness.freeciv_state import FreeCivAction
from game_arena.harness.freeciv_state_stubs import FreeCivGameStateStub

# Performance and security constants
MAX_INPUT_SIZE = 10 * 1024  # 10KB max input size
REGEX_TIMEOUT_SECONDS = 5

# Similarity calculation constants
SIMILARITY_THRESHOLD = 0.3
TOKEN_WEIGHT = 0.3
NUMBER_WEIGHT = 0.3
ACTION_WEIGHT = 0.3  # Increased weight for action matching
EDIT_WEIGHT = 0.1

# Bonus score constants
EXACT_SUBSTRING_BONUS = 0.15
PARTIAL_TOKEN_BONUS = 0.05
ACTION_MATCH_BONUS = 0.1

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

    def __init__(self):
        """Initialize FreeCiv rule-based parser."""
        super().__init__()
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
        text = parser_input.text.strip()

        # Security: Validate input size to prevent DoS attacks
        if len(text) > MAX_INPUT_SIZE:
            return None

        # Try JSON parsing first (most structured)
        json_action = self._try_parse_json(text)
        if json_action:
            return json_action

        # Try natural language parsing using FreeCivAction model
        natural_action = self._try_parse_natural_language(text, parser_input)
        if natural_action:
            return natural_action

        # Fall back to existing regex-based parsing
        # First try matching the original text (in case it's already well-formed)
        for action_type, pattern in self._action_patterns.items():
            match = pattern.search(text)
            if match:
                # Return the original text if it already matches perfectly
                return text

        # If no direct match, try with cleaned text
        cleaned_text = self._clean_text(text)
        for action_type, pattern in self._action_patterns.items():
            match = pattern.search(cleaned_text)
            if match:
                # Reconstruct the action string in canonical format
                return self._reconstruct_action(action_type, match.groups())

        # If no pattern matches, try soft matching against legal moves
        if parser_input.legal_moves:
            return self._soft_match(cleaned_text, parser_input.legal_moves)

        return None

    def _try_parse_json(self, text: str) -> Optional[str]:
        """Try to parse JSON from text and convert to canonical action string.

        Args:
            text: Raw text that might contain JSON

        Returns:
            Canonical action string if JSON parsing succeeds, None otherwise
        """
        # Security: Check input size before regex processing
        if len(text) > MAX_INPUT_SIZE:
            return None

        # Look for JSON-like patterns in the text (with size limits)
        json_patterns = [
            r"\{[^}]{1,1000}\}",  # Simple JSON object (max 1000 chars)
            r"\{[^{}]{0,500}\{[^}]{0,500}\}[^{}]{0,500}\}",  # Nested JSON (limited nesting)
        ]

        for pattern in json_patterns:
            json_matches = re.findall(pattern, text, re.DOTALL)
            for json_match in json_matches:
                try:
                    # Try to parse JSON and convert to FreeCivAction
                    action = FreeCivAction.from_json(json_match)
                    return self._action_to_canonical_string(action)
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue  # Try next JSON match

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
        except Exception:
            # If natural language parsing fails, return None to try other methods
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

        else:
            # Generic format
            target_str = ""
            if action.target:
                if "x" in action.target and "y" in action.target:
                    target_str = f"_to({action.target['x']},{action.target['y']})"
                elif "id" in action.target:
                    target_str = f"_target({action.target['id']})"
                elif "value" in action.target:
                    target_str = f"_target({action.target['value']})"

            return (
                f"{action.action_type}_{action.source}({action.actor_id}){target_str}"
            )

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

    def __init__(self):
        """Initialize FreeCiv soft parser with caching."""
        super().__init__("freeciv")
        self._similarity_cache = {}  # Cache for similarity calculations
        self._max_cache_size = 1000
        self._legal_moves_hash = None  # Track when legal moves change

    def parse(self, parser_input: parsers.TextParserInput) -> Optional[str]:
        """Parse FreeCiv action using soft matching.

        Args:
            parser_input: Text input containing the action to parse

        Returns:
            Best matching legal action or None
        """
        if not parser_input.legal_moves:
            return None

        # Check if legal moves changed and invalidate cache if needed
        current_moves_hash = self._hash_legal_moves(parser_input.legal_moves)
        if self._legal_moves_hash != current_moves_hash:
            self._similarity_cache.clear()
            self._legal_moves_hash = current_moves_hash

        text = parser_input.text.strip().lower()

        # Security: Validate input size
        if len(text) > MAX_INPUT_SIZE:
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
        if best_score > SIMILARITY_THRESHOLD:
            return best_match

        return None

    def _calculate_similarity_cached(self, text: str, move: str) -> float:
        """Calculate similarity with caching for performance.

        Args:
            text: Cleaned input text
            move: Legal move string

        Returns:
            Cached similarity score between 0 and 1
        """
        cache_key = (text, move)

        # Check cache first
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        # Calculate similarity
        similarity = self._calculate_enhanced_similarity(text, move)

        # Cache result (with size limit)
        if len(self._similarity_cache) >= self._max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._similarity_cache))
            del self._similarity_cache[oldest_key]

        self._similarity_cache[cache_key] = similarity
        return similarity

    def _hash_legal_moves(self, legal_moves: List[str]) -> str:
        """Create a hash of legal moves for cache invalidation.

        Args:
            legal_moves: List of legal move strings

        Returns:
            SHA256 hash of the sorted legal moves
        """
        moves_str = "|".join(sorted(legal_moves))
        return hashlib.sha256(moves_str.encode()).hexdigest()[:16]

    def invalidate_cache(self) -> None:
        """Manually invalidate the similarity cache."""
        self._similarity_cache.clear()
        self._legal_moves_hash = None

    def _calculate_enhanced_similarity(self, text: str, move: str) -> float:
        """Enhanced similarity calculation with multiple algorithms.

        This method combines multiple similarity metrics to provide robust
        fuzzy matching between user input and legal moves. The algorithm
        uses weighted scoring across different dimensions:

        1. Token similarity (40%): Overlap of words/tokens
        2. Number similarity (30%): Overlap of numeric IDs
        3. Action similarity (20%): Overlap of action keywords
        4. Edit distance (10%): Character-level similarity

        Additional bonuses are applied for:
        - Exact substring matches (+0.15)
        - Partial token matches (+0.05)
        - Action type matches (+0.10)

        Args:
            text: Cleaned input text from user
            move: Legal move string from game state

        Returns:
            Similarity score between 0.0 and 1.0, where:
            - 0.0: No similarity
            - 0.3: Minimum threshold for matching
            - 0.6+: High confidence match
            - 1.0: Perfect match

        Examples:
            >>> parser = FreeCivSoftParser()
            >>> score = parser._calculate_enhanced_similarity(
            ...     "move settlers 101",
            ...     "unit_move_settlers(101)_to(2,3)"
            ... )
            >>> score > 0.6  # High confidence
            True

            >>> score = parser._calculate_enhanced_similarity(
            ...     "attack warriors",
            ...     "unit_attack_warriors(102)_target(203)"
            ... )
            >>> score > 0.3  # Above threshold
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
            similarities.append(("token", token_similarity, TOKEN_WEIGHT))
        if number_similarity > 0:
            similarities.append(("number", number_similarity, NUMBER_WEIGHT))
        if action_similarity > 0:
            similarities.append(("action", action_similarity, ACTION_WEIGHT))

        similarities.append(("edit", edit_similarity, EDIT_WEIGHT))

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
            bonus += EXACT_SUBSTRING_BONUS

        # Partial substring matches for longer tokens
        for token in text_tokens:
            if len(token) > 3 and token in move:
                bonus += PARTIAL_TOKEN_BONUS
                break

        # Action type exact match bonus
        if text_actions & move_actions:
            bonus += ACTION_MATCH_BONUS

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
        # Use separate cache key prefix to distinguish from similarity cache
        cache_key = ("edit", s1, s2)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

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

        # Cache the result with size limit management
        if len(self._similarity_cache) >= self._max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._similarity_cache))
            del self._similarity_cache[oldest_key]

        self._similarity_cache[cache_key] = result
        return result


def create_freeciv_parser_chain(model=None) -> parsers.ChainedMoveParser:
    """Create a chained parser for FreeCiv actions.

    Args:
        model: Optional LLM model for natural language parsing

    Returns:
        ChainedMoveParser configured for FreeCiv with appropriate parsers
    """
    parser_list = [FreeCivRuleBasedParser()]

    # Add LLM parser if model is provided
    if model is not None:
        parser_list.append(FreeCivLLMParser(model))

    # Always add soft parser as fallback
    parser_list.append(FreeCivSoftParser())

    return parsers.ChainedMoveParser(parser_list)
