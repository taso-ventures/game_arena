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

import re
import json
from typing import Optional, Dict, Any, List

from game_arena.harness import llm_parsers, parsers
from game_arena.harness.freeciv_state import FreeCivAction


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


class FreeCivRuleBasedParser(parsers.MoveParser):
    """Rule-based parser for FreeCiv action strings."""

    def __init__(self):
        """Initialize FreeCiv rule-based parser."""
        super().__init__()
        # Compile regex patterns for different action types
        self._action_patterns = {
            'unit_move': re.compile(r'unit_move_([^_]+)\((\d+)\)_to\((\d+),(\d+)\)'),
            'unit_attack': re.compile(r'unit_attack_([^_]+)\((\d+)\)_target\((\d+)\)'),
            'unit_fortify': re.compile(r'unit_fortify_([^_]+)\((\d+)\)'),
            'unit_explore': re.compile(r'unit_explore_([^_]+)\((\d+)\)'),
            'unit_build_improvement': re.compile(r'unit_build_improvement_([^_]+)\((\d+)\)_target\(([^)]+)\)'),
            'city_production': re.compile(r'city_production_([^_]+)\((\d+)\)_target\(([^)]+)\)'),
            'city_build_improvement': re.compile(r'city_build_improvement_([^_]+)\((\d+)\)_target\(([^)]+)\)'),
            'city_celebrate': re.compile(r'city_celebrate_([^_]+)\((\d+)\)'),
        }

    def parse(self, parser_input: parsers.TextParserInput) -> Optional[str]:
        """Parse FreeCiv action from text input with enhanced JSON support.

        Args:
            parser_input: Text input containing the action to parse

        Returns:
            Parsed action string if successful, None otherwise
        """
        text = parser_input.text.strip()

        # Try JSON parsing first (most structured)
        json_action = self._try_parse_json(text)
        if json_action:
            return json_action

        # Try natural language parsing using FreeCivAction model
        natural_action = self._try_parse_natural_language(text, parser_input)
        if natural_action:
            return natural_action

        # Fall back to existing regex-based parsing
        cleaned_text = self._clean_text(text)

        # Check if the cleaned text matches any of our action patterns
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
        # Look for JSON-like patterns in the text
        json_patterns = [
            r'\{[^}]+\}',  # Simple JSON object
            r'\{[^{}]*\{[^}]*\}[^{}]*\}',  # Nested JSON object
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

    def _try_parse_natural_language(self, text: str, parser_input: parsers.TextParserInput) -> Optional[str]:
        """Try to parse natural language using FreeCivAction model.

        Args:
            text: Natural language text
            parser_input: Parser input with context

        Returns:
            Canonical action string if parsing succeeds, None otherwise
        """
        try:
            # Create a minimal mock game state for natural language parsing
            mock_state = self._create_mock_game_state()

            # Try to parse using FreeCivAction natural language parser
            action = FreeCivAction.from_natural_language(text, mock_state)

            if action:
                return self._action_to_canonical_string(action)

            return None
        except Exception:
            # If natural language parsing fails, return None to try other methods
            return None

    def _create_mock_game_state(self):
        """Create a minimal mock game state for natural language parsing."""
        from unittest.mock import Mock

        mock_state = Mock()

        # Mock units for unit ID resolution
        mock_state.units = {}
        for i in range(101, 110):
            unit = Mock()
            unit.unit_id = i
            unit.kind = "settlers" if i == 101 else "warrior"
            mock_state.units[i] = unit

        # Mock cities for city ID resolution
        mock_state.cities = {}
        for i in range(301, 305):
            city = Mock()
            city.city_id = i
            city.name = f"City{i}"
            mock_state.cities[i] = city

        return mock_state

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
            target_value = action.target.get('value') or action.target.get('name') or action.target.get('id')
            return f"city_production_{action.source}({action.actor_id})_target({target_value})"

        elif action.action_type == "city_build_improvement" and action.target:
            target_value = action.target.get('value') or action.target.get('name') or action.target.get('id')
            return f"city_build_improvement_{action.source}({action.actor_id})_target({target_value})"

        elif action.action_type == "tech_research" and action.target:
            target_value = action.target.get('value') or action.target.get('name') or action.target.get('tech')
            return f"tech_research_player({action.actor_id})_target({target_value})"

        else:
            # Generic format
            target_str = ""
            if action.target:
                if 'x' in action.target and 'y' in action.target:
                    target_str = f"_to({action.target['x']},{action.target['y']})"
                elif 'id' in action.target:
                    target_str = f"_target({action.target['id']})"
                elif 'value' in action.target:
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
        text = re.sub(r'^(I will|I choose|My action is|Action:|Move:)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*(\.|\!|\?)$', '', text)

        # Remove quotes and brackets
        text = re.sub(r'^["\'\[\(]|["\'\]\)]$', '', text.strip())

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Extract action-like patterns
        action_match = re.search(r'(unit_\w+|city_\w+).*?(?=\.|$)', text, re.IGNORECASE)
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
        if action_type == 'unit_move':
            unit_type, unit_id, x, y = groups
            return f"unit_move_{unit_type}({unit_id})_to({x},{y})"
        elif action_type == 'unit_attack':
            unit_type, unit_id, target_id = groups
            return f"unit_attack_{unit_type}({unit_id})_target({target_id})"
        elif action_type in ['unit_fortify', 'unit_explore']:
            unit_type, unit_id = groups
            return f"{action_type}_{unit_type}({unit_id})"
        elif action_type in ['unit_build_improvement', 'city_production', 'city_build_improvement']:
            actor_type, actor_id, target = groups
            return f"{action_type}_{actor_type}({actor_id})_target({target})"
        elif action_type == 'city_celebrate':
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
        keywords = set(re.findall(r'\w+', text_lower))
        numbers = set(re.findall(r'\d+', text))

        best_match = None
        best_score = 0

        for move in legal_moves:
            move_lower = move.lower()
            move_keywords = set(re.findall(r'\w+', move_lower))
            move_numbers = set(re.findall(r'\d+', move))

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

    def parse(self, parser_input: parsers.TextParserInput) -> Optional[str]:
        """Parse FreeCiv action using soft matching.

        Args:
            parser_input: Text input containing the action to parse

        Returns:
            Best matching legal action or None
        """
        if not parser_input.legal_moves:
            return None

        text = parser_input.text.strip().lower()

        # Remove common noise words and formatting
        noise_patterns = [
            r'\b(i|will|choose|my|action|is|move|the)\b',
            r'[^\w\s\(\),]',  # Remove special chars except parentheses and commas
            r'\s+',  # Normalize whitespace
        ]

        cleaned_text = text
        for pattern in noise_patterns:
            cleaned_text = re.sub(pattern, ' ' if pattern == r'\s+' else '', cleaned_text)
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
        if best_score > 0.3:  # Adjust threshold as needed
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

    def _calculate_enhanced_similarity(self, text: str, move: str) -> float:
        """Enhanced similarity calculation with multiple algorithms.

        Args:
            text: Cleaned input text
            move: Legal move string

        Returns:
            Similarity score between 0 and 1
        """
        # Extract tokens from both strings
        text_tokens = set(text.split())
        move_tokens = set(move.lower().split('_'))

        # Extract numbers and identifiers
        text_numbers = set(re.findall(r'\d+', text))
        move_numbers = set(re.findall(r'\d+', move))

        # Extract action type words
        text_actions = set(re.findall(r'\b(move|attack|build|fortify|explore|research|produce)\w*', text.lower()))
        move_actions = set(re.findall(r'\b(move|attack|build|fortify|explore|research|produce)\w*', move.lower()))

        # Calculate different similarity metrics
        similarities = []

        # 1. Token overlap similarity
        if text_tokens or move_tokens:
            token_overlap = len(text_tokens & move_tokens)
            total_tokens = len(text_tokens | move_tokens)
            token_similarity = token_overlap / max(total_tokens, 1)
            similarities.append(('token', token_similarity, 0.4))

        # 2. Number overlap similarity (high weight for IDs)
        if text_numbers or move_numbers:
            number_overlap = len(text_numbers & move_numbers)
            total_numbers = len(text_numbers | move_numbers)
            number_similarity = number_overlap / max(total_numbers, 1)
            similarities.append(('number', number_similarity, 0.3))

        # 3. Action type similarity (important for understanding intent)
        if text_actions or move_actions:
            action_overlap = len(text_actions & move_actions)
            total_actions = len(text_actions | move_actions)
            action_similarity = action_overlap / max(total_actions, 1)
            similarities.append(('action', action_similarity, 0.2))

        # 4. Levenshtein-like similarity for typos
        edit_similarity = self._calculate_edit_similarity(text, move)
        similarities.append(('edit', edit_similarity, 0.1))

        # Weighted average of similarities
        if similarities:
            weighted_sum = sum(score * weight for _, score, weight in similarities)
            total_weight = sum(weight for _, _, weight in similarities)
            base_similarity = weighted_sum / total_weight
        else:
            base_similarity = 0.0

        # Bonus adjustments
        bonus = 0.0

        # Exact substring match bonus
        if text in move or move in text:
            bonus += 0.15

        # Partial substring matches for longer tokens
        for token in text_tokens:
            if len(token) > 3 and token in move:
                bonus += 0.05
                break

        # Action type exact match bonus
        if text_actions & move_actions:
            bonus += 0.1

        # Final similarity with bonus
        final_similarity = min(base_similarity + bonus, 1.0)

        return final_similarity

    def _calculate_edit_similarity(self, text1: str, text2: str) -> float:
        """Calculate edit distance similarity for handling typos.

        Args:
            text1: First text string
            text2: Second text string

        Returns:
            Similarity score based on edit distance (0-1)
        """
        # Simple edit distance calculation
        def edit_distance(s1, s2):
            if len(s1) < len(s2):
                return edit_distance(s2, s1)

            if len(s2) == 0:
                return len(s1)

            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        # Normalize strings for comparison
        s1 = re.sub(r'[^\w]', '', text1.lower())
        s2 = re.sub(r'[^\w]', '', text2.lower())

        if not s1 or not s2:
            return 0.0

        max_len = max(len(s1), len(s2))
        distance = edit_distance(s1, s2)

        # Convert to similarity (1 - normalized distance)
        similarity = 1.0 - (distance / max_len)
        return max(0.0, similarity)


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