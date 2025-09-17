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
        """Parse FreeCiv action from text input.

        Args:
            parser_input: Text input containing the action to parse

        Returns:
            Parsed action string if successful, None otherwise
        """
        text = parser_input.text.strip()

        # Try to extract action using various cleaning strategies
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
    """Soft parser for FreeCiv actions using fuzzy matching."""

    def __init__(self):
        """Initialize FreeCiv soft parser."""
        super().__init__("freeciv")

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
            score = self._calculate_similarity(cleaned_text, move.lower())
            if score > best_score:
                best_score = score
                best_match = move

        # Return match if confidence is high enough
        if best_score > 0.3:  # Adjust threshold as needed
            return best_match

        return None

    def _calculate_similarity(self, text: str, move: str) -> float:
        """Calculate similarity score between text and move.

        Args:
            text: Cleaned input text
            move: Legal move string

        Returns:
            Similarity score between 0 and 1
        """
        # Extract tokens from both strings
        text_tokens = set(text.split())
        move_tokens = set(move.lower().split('_'))

        # Also extract numbers and identifiers
        text_numbers = set(re.findall(r'\d+', text))
        move_numbers = set(re.findall(r'\d+', move))

        # Calculate overlap
        token_overlap = len(text_tokens & move_tokens)
        number_overlap = len(text_numbers & move_numbers)

        # Calculate base similarity
        total_tokens = len(text_tokens | move_tokens)
        total_numbers = len(text_numbers | move_numbers)

        if total_tokens == 0 and total_numbers == 0:
            return 0.0

        token_similarity = token_overlap / max(total_tokens, 1)
        number_similarity = number_overlap / max(total_numbers, 1) if total_numbers > 0 else 0

        # Weighted combination
        similarity = 0.7 * token_similarity + 0.3 * number_similarity

        # Bonus for substring matches
        if text in move or any(token in move for token in text_tokens if len(token) > 2):
            similarity += 0.1

        return min(similarity, 1.0)


def create_freeciv_parser_chain() -> parsers.ChainedMoveParser:
    """Create a chained parser for FreeCiv actions.

    Returns:
        ChainedMoveParser configured for FreeCiv
    """
    return parsers.ChainedMoveParser([
        FreeCivRuleBasedParser(),
        FreeCivSoftParser()
    ])