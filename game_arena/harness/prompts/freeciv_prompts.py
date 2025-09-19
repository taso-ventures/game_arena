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

"""FreeCiv-specific prompt builder for LLM agents."""

import enum
from typing import Any, Dict, List

from game_arena.harness.freeciv_state import FreeCivAction

# Game phase thresholds
EARLY_GAME_TURN_LIMIT = 50
MID_GAME_TURN_LIMIT = 150

# Display limits for context compression
MAX_UNITS_DISPLAY = 20
MAX_CITIES_DISPLAY = 10
MAX_MILITARY_UNITS_DISPLAY = 15
MAX_CIVILIAN_UNITS_DISPLAY = 5

# Priority action limits
MAX_PRIORITIZED_ACTIONS = 10

# Threat detection thresholds
THREAT_DISTANCE_THRESHOLD = 3
LOW_HP_THRESHOLD = 50

# Victory condition thresholds
MILITARY_STRENGTH_THRESHOLD = 0.5
MIN_MILITARY_UNITS_FOR_DOMINATION = 3
CITY_COUNT_FOR_VICTORY = 4
ENDGAME_TURN_THRESHOLD = 100


@enum.unique
class ActionPriority(enum.IntEnum):
  """Priority levels for FreeCiv actions."""

  HIGHEST = 1  # unit_attack
  HIGH = 2  # city_production
  MEDIUM = 3  # unit_build
  LOW = 4  # unit_move
  LOWEST = 5  # city_work
  DEFAULT = 10  # unknown actions


MODEL_CONFIGS = {
    "gpt-5": {
        "max_tokens": 4000,
        "style": "structured",
        "reasoning": "chain_of_thought",
        "format": "json_with_explanation",
    },
    "claude": {
        "max_tokens": 3500,
        "style": "conversational",
        "reasoning": "step_by_step",
        "format": "natural_with_tags",
    },
    "deepseek": {
        "max_tokens": 3000,
        "style": "concise",
        "reasoning": "direct",
        "format": "structured_json",
    },
}

PROMPT_TEMPLATES = {
    "early_game": {
        "gpt-5": """STRATEGIC ANALYSIS - Turn {turn}
=============================
Victory: {victory_progress}% progress toward {victory_type}
Position: {position} (score: {score})

EARLY GAME PRIORITIES:
1. Explore and map the surrounding area
2. Find optimal city sites with resources
3. Establish basic infrastructure
4. Defend against barbarians

{strategic_summary}

AVAILABLE ACTIONS (sorted by impact):
{prioritized_actions}

Respond with JSON: {{"action": "action_type", "reasoning": "step-by-step analysis", "confidence": 0.95}}""",
        "claude": """You're playing FreeCiv as the {player_name}. It's the early game (Turn {turn}).

<game_state>
{strategic_summary}
</game_state>

<priorities>
- Exploration: Scout the map for resources and good city sites
- Settlement: Establish cities near food and strategic resources
- Defense: Protect against barbarian threats
- Infrastructure: Build basic improvements
</priorities>

<actions>
{prioritized_actions}
</actions>

Choose your next action and explain your reasoning. Focus on long-term strategic positioning.""",
        "deepseek": """Turn {turn} - Early Game

Status: {position} (Score: {score})
{strategic_summary}

Key Actions:
{prioritized_actions}

Select action with brief reasoning.""",
    },
    "mid_game": {
        "gpt-5": """STRATEGIC ANALYSIS - Turn {turn}
=============================
Victory: {victory_progress}% progress toward {victory_type}
Position: {position} (score: {score})

MID GAME PRIORITIES:
1. Expand territory and establish new cities
2. Advance technology tree strategically
3. Build military for defense/conquest
4. Develop trade and diplomacy

{strategic_summary}

AVAILABLE ACTIONS (sorted by impact):
{prioritized_actions}

Respond with JSON: {{"action": "action_type", "reasoning": "strategic analysis", "confidence": 0.90}}""",
        "claude": """You're playing FreeCiv as the {player_name}. It's the mid game (Turn {turn}).

<game_state>
{strategic_summary}
</game_state>

<priorities>
- Expansion: Claim territory before rivals
- Technology: Research key advances for military/economy
- Military: Build forces for defense or opportunity
- Diplomacy: Manage relationships with other civilizations
</priorities>

<actions>
{prioritized_actions}
</actions>

Choose your action focusing on competitive advantage and strategic positioning.""",
        "deepseek": """Turn {turn} - Mid Game

Status: {position} (Score: {score})
{strategic_summary}

Focus: Expansion, tech advancement, military buildup

Actions:
{prioritized_actions}

Select optimal action.""",
    },
    "late_game": {
        "gpt-5": """STRATEGIC ANALYSIS - Turn {turn}
=============================
Victory: {victory_progress}% progress toward {victory_type}
Position: {position} (score: {score})

LATE GAME PRIORITIES:
1. Push for victory condition completion
2. Defend against rival victory attempts
3. Optimize production for endgame
4. Execute military campaigns if needed

{strategic_summary}

AVAILABLE ACTIONS (sorted by impact):
{prioritized_actions}

Respond with JSON: {{"action": "action_type", "reasoning": "victory-focused strategy", "confidence": 0.85}}""",
        "claude": """You're playing FreeCiv as the {player_name}. It's the late game (Turn {turn}).

<game_state>
{strategic_summary}
</game_state>

<priorities>
- Victory: Focus on completing your victory condition
- Defense: Block rivals from achieving victory
- Optimization: Maximize production efficiency
- Timing: Execute decisive moves at the right moment
</priorities>

<actions>
{prioritized_actions}
</actions>

Make your move with victory in mind. Time is running out.""",
        "deepseek": """Turn {turn} - Late Game

Status: {position} (Score: {score})
{strategic_summary}

Victory Focus: {victory_type}
Progress: {victory_progress}%

Critical Actions:
{prioritized_actions}

Choose winning move.""",
    },
}


class ContextManager:
  """Manages context window optimization and information prioritization."""

  def __init__(self):
    self.max_units_display = MAX_UNITS_DISPLAY
    self.max_cities_display = MAX_CITIES_DISPLAY

  def compress_observation(
      self, obs: Dict[str, Any], _max_tokens: int
  ) -> Dict[str, Any]:
    """Intelligently compress observation while preserving decision-critical info.

    Args:
        obs: Full observation dictionary
        max_tokens: Maximum token budget for the observation

    Returns:
        Compressed observation dictionary
    """
    compressed = obs.copy()

    # Compress units by grouping similar types
    if (
        "units" in compressed
        and len(compressed["units"]) > self.max_units_display
    ):
      units = compressed["units"]
      # Keep military units and unique units, summarize workers
      military_units = [
          u
          for u in units
          if u.get("type", "").lower()
          in ["warrior", "archer", "phalanx", "legion"]
      ]
      other_units = [u for u in units if u not in military_units]

      compressed["units"] = (
          military_units[:MAX_MILITARY_UNITS_DISPLAY]
          + other_units[:MAX_CIVILIAN_UNITS_DISPLAY]
      )
      if len(units) > self.max_units_display:
        compressed["units_summary"] = (
            f"Total {len(units)} units ({len(military_units)} military,"
            f" {len(other_units)} civilian)"
        )

    # Compress cities by keeping largest and most strategic
    if (
        "cities" in compressed
        and len(compressed["cities"]) > self.max_cities_display
    ):
      cities = compressed["cities"]
      # Sort by population and keep largest
      sorted_cities = sorted(
          cities, key=lambda c: c.get("pop", 0), reverse=True
      )
      compressed["cities"] = sorted_cities[: self.max_cities_display]
      if len(cities) > self.max_cities_display:
        compressed["cities_summary"] = (
            f"Showing {self.max_cities_display} of {len(cities)} cities"
        )

    return compressed

  def prioritize_information(
      self, obs: Dict[str, Any], phase: str
  ) -> Dict[str, Any]:
    """Prioritize information based on game phase.

    Args:
        obs: Observation dictionary
        phase: Game phase ('early_game', 'mid_game', 'late_game')

    Returns:
        Prioritized observation with phase-appropriate focus
    """
    prioritized = obs.copy()

    if phase == "early_game":
      # Focus on exploration, city sites, barbarians
      prioritized["focus"] = "exploration_and_settlement"
    elif phase == "mid_game":
      # Focus on expansion, technology, military
      prioritized["focus"] = "expansion_and_technology"
    else:  # late_game
      # Focus on victory conditions, military operations
      prioritized["focus"] = "victory_and_military"

    return prioritized


class ObservationBuilder:
  """Builds formatted observations and strategic summaries."""

  def __init__(self):
    self.context_manager = ContextManager()

  def build_strategic_summary(self, obs: Dict[str, Any]) -> str:
    """Build strategic summary including victory progress and relative position.

    Args:
        obs: Observation dictionary

    Returns:
        Strategic summary string
    """
    turn = obs.get("turn", 0)
    players = obs.get("players", {})

    if not players:
      return f"Turn {turn}: Gathering intelligence..."

    # Get current player info (assuming player 1 is us for now)
    current_player = players.get(1, {})
    player_name = current_player.get("name", "Unknown")
    score = current_player.get("score", 0)
    gold = current_player.get("gold", 0)

    summary = f"Turn {turn}: Playing as {player_name}\n"
    summary += f"Score: {score}, Gold: {gold}\n"

    # Add relative position if we have other players
    if len(players) > 1:
      scores = [p.get("score", 0) for p in players.values()]
      our_rank = sorted(scores, reverse=True).index(score) + 1
      summary += (
          f"Current Ranking: {our_rank} of {len(players)} civilizations\n"
      )

    # Add unit and city counts
    units = obs.get("units", [])
    cities = obs.get("cities", [])
    summary += f"Military: {len(units)} units, Territory: {len(cities)} cities"

    return summary

  def _identify_priorities(self, _obs: Dict[str, Any], phase: str) -> str:
    """Identify dynamic priorities based on game state and phase.

    Args:
        obs: Observation dictionary
        phase: Current game phase

    Returns:
        Priority description string
    """
    if phase == "early_game":
      return (
          "Focus on exploration, finding good city sites, and basic"
          " infrastructure"
      )
    if phase == "mid_game":
      return (
          "Prioritize territorial expansion, technological advancement, and"
          " military development"
      )
    # late_game
    return (
        "Push for victory conditions, defend against rivals, and optimize"
        " production"
    )

  def _assess_threats(self, obs: Dict[str, Any]) -> str:
    """Identify immediate dangers and threats.

    Args:
        obs: Observation dictionary

    Returns:
        Threat assessment string
    """
    threats = []

    # Get our cities and units for threat analysis
    our_cities = obs.get("cities", [])
    all_units = obs.get("units", [])

    # Identify our player ID (assume player 1 for now)
    our_player_id = 1

    # Find enemy units near our cities
    enemy_units = [u for u in all_units if u.get("owner") != our_player_id]

    for city in our_cities:
      city_pos = (city.get("x", 0), city.get("y", 0))

      # Check for enemy units within 3 tiles of our cities
      nearby_enemies = []
      for unit in enemy_units:
        unit_pos = (unit.get("x", 0), unit.get("y", 0))
        distance = abs(city_pos[0] - unit_pos[0]) + abs(
            city_pos[1] - unit_pos[1]
        )

        if distance <= THREAT_DISTANCE_THRESHOLD:  # Within threatening range
          nearby_enemies.append(unit)

      if nearby_enemies:
        enemy_types = [u.get("type", "Unknown") for u in nearby_enemies]
        threats.append(
            f"{city.get('name', 'City')} threatened by {len(nearby_enemies)} "
            f"enemy units: {', '.join(set(enemy_types))}"
        )

    # Check for barbarian units (assuming negative owner means barbarian)
    barbarian_units = [u for u in all_units if u.get("owner", 0) < 0]
    if barbarian_units:
      threats.append(f"{len(barbarian_units)} barbarian units detected on map")

    # Check for units with low HP
    our_units = [u for u in all_units if u.get("owner") == our_player_id]
    low_hp_units = [u for u in our_units if u.get("hp", 100) < LOW_HP_THRESHOLD]
    if low_hp_units:
      threats.append(f"{len(low_hp_units)} of our units need healing")

    if not threats:
      return "No immediate threats detected. Maintain defensive vigilance."

    return "THREATS: " + "; ".join(threats)

  def _identify_opportunities(self, obs: Dict[str, Any]) -> str:
    """Find expansion and attack opportunities.

    Args:
        obs: Observation dictionary

    Returns:
        Opportunity description string
    """
    opportunities = []

    # Get map and unit data
    map_data = obs.get("map", {})
    tiles = map_data.get("tiles", [])
    all_units = obs.get("units", [])
    our_cities = obs.get("cities", [])

    # Identify our player ID
    our_player_id = 1

    # Look for unoccupied tiles with resources
    resource_tiles = []
    for tile in tiles:
      if tile.get("resource") and not tile.get("city"):
        # Check if tile is not too close to enemy cities
        resource_tiles.append(tile)

    if resource_tiles:
      unique_resources = set(t.get("resource") for t in resource_tiles)
      opportunities.append(
          f"Unclaimed resources available: {', '.join(unique_resources)}"
      )

    # Look for good city locations (not implemented in tile data, so estimate)
    if len(our_cities) < CITY_COUNT_FOR_VICTORY:  # Room for expansion
      opportunities.append(
          "Territory expansion possible - scout for city sites"
      )

    # Identify weak enemy units that could be attacked
    enemy_units = [u for u in all_units if u.get("owner") != our_player_id]
    our_units = [u for u in all_units if u.get("owner") == our_player_id]

    weak_enemies = [
        u for u in enemy_units if u.get("hp", 100) < LOW_HP_THRESHOLD
    ]
    if weak_enemies and our_units:
      military_units = [
          u
          for u in our_units
          if u.get("type", "").lower()
          in ["warrior", "archer", "phalanx", "legion"]
      ]
      if military_units:
        opportunities.append(
            f"Attack opportunity: {len(weak_enemies)} weakened enemy units"
        )

    # Technology opportunities (based on turn number as proxy)
    turn = obs.get("turn", 0)
    if turn < EARLY_GAME_TURN_LIMIT:
      opportunities.append("Research Bronze Working for military units")
    elif turn < ENDGAME_TURN_THRESHOLD:
      opportunities.append("Advance to Iron Working for stronger units")
    else:
      opportunities.append("Research late-game technologies for victory")

    # Trade opportunities (if we have multiple cities)
    if len(our_cities) >= 2:
      opportunities.append("Establish trade routes between cities")

    if not opportunities:
      return "Limited opportunities - focus on defense and development."

    return "OPPORTUNITIES: " + "; ".join(opportunities)

  def format_prioritized_actions(
      self, actions: List[FreeCivAction], obs: Dict[str, Any]
  ) -> str:
    """Show top actions with reasoning and impact assessment.

    Args:
        actions: List of available FreeCiv actions
        obs: Observation dictionary for context

    Returns:
        Formatted action list string
    """
    if not actions:
      return "No actions available."

    # Sort actions by priority (for now, simple heuristic)
    prioritized = self._prioritize_actions(actions)

    formatted_actions = []
    for i, action in enumerate(
        prioritized[:MAX_PRIORITIZED_ACTIONS], 1
    ):  # Top actions
      action_desc = self._format_action_description(action)
      impact = self._assess_action_impact(action, obs)
      formatted_actions.append(f"{i}. {action_desc} - {impact}")

    return "\n".join(formatted_actions)

  def _prioritize_actions(
      self, actions: List[FreeCivAction]
  ) -> List[FreeCivAction]:
    """Sort actions by strategic priority.

    Args:
        actions: List of actions to prioritize

    Returns:
        Sorted list of actions
    """
    # Map action types to priority levels using the enum
    priority_map = {
        "unit_attack": ActionPriority.HIGHEST,
        "city_production": ActionPriority.HIGH,
        "unit_build": ActionPriority.MEDIUM,
        "unit_move": ActionPriority.LOW,
        "city_work": ActionPriority.LOWEST,
    }

    return sorted(
        actions,
        key=lambda a: priority_map.get(a.action_type, ActionPriority.DEFAULT),
    )

  def _format_action_description(self, action: FreeCivAction) -> str:
    """Format a single action for display.

    Args:
        action: FreeCiv action to format

    Returns:
        Formatted action string
    """
    if action.action_type == "unit_move":
      target = action.target
      return (
          f"Move unit {action.actor_id} to ({target.get('x', '?')},"
          f" {target.get('y', '?')})"
      )
    if action.action_type == "city_production":
      item = action.target.get("value", action.target.get("item", "Unknown"))
      return f"City {action.actor_id} produces {item}"
    if action.action_type == "unit_attack":
      target_id = action.target.get("id", action.target.get("unit_id", "?"))
      return f"Unit {action.actor_id} attacks unit {target_id}"
    return f"{action.action_type} with unit/city {action.actor_id}"

  def _assess_action_impact(
      self, action: FreeCivAction, _obs: Dict[str, Any]
  ) -> str:
    """Assess the strategic impact of an action.

    Args:
        action: Action to assess
        obs: Current observation for context

    Returns:
        Impact assessment string
    """
    if action.action_type == "unit_attack":
      return "High impact: Eliminate threat or expand territory"
    if action.action_type == "city_production":
      return "Medium impact: Strengthen economy or military"
    if action.action_type == "unit_move":
      return "Low-Medium impact: Positioning for future actions"
    return "Variable impact: Situation dependent"


class FreeCivPromptBuilder:
  """Main prompt builder for FreeCiv LLM agents."""

  def __init__(self):
    """Initialize the prompt builder with templates and configurations."""
    self.templates = PROMPT_TEMPLATES
    self.model_configs = MODEL_CONFIGS
    self.observation_builder = ObservationBuilder()
    self.context_manager = ContextManager()

  def build_enhanced_prompt(
      self,
      observation: Dict[str, Any],
      legal_actions: List[FreeCivAction],
      model_name: str,
  ) -> str:
    """Generate model-specific prompt with context optimization.

    Args:
        observation: Game state observation
        legal_actions: List of legal FreeCiv actions
        model_name: Target model name ('gpt-5', 'claude', 'deepseek')

    Returns:
        Generated prompt string optimized for the specified model
    """
    # Extract state from observation
    state = observation.get("state")
    if hasattr(state, "turn"):
      obs_dict = self._state_to_dict(state)
    else:
      obs_dict = observation

    # Determine game phase
    phase = self._detect_game_phase(obs_dict)

    # Get model configuration
    model_config = self.model_configs.get(
        model_name, self.model_configs["gpt-5"]
    )

    # Compress observation for context window
    compressed_obs = self.context_manager.compress_observation(
        obs_dict, model_config["max_tokens"]
    )

    # Build strategic components
    strategic_summary = self.observation_builder.build_strategic_summary(
        compressed_obs
    )
    prioritized_actions = self.observation_builder.format_prioritized_actions(
        legal_actions, compressed_obs
    )

    # Get template for model and phase (fallback to gpt-5 for unknown models)
    phase_templates = self.templates[phase]
    template = phase_templates.get(model_name, phase_templates["gpt-5"])

    # Fill template with data
    prompt = template.format(
        turn=compressed_obs.get("turn", 0),
        player_name=self._get_player_name(compressed_obs),
        position=self._get_position_string(compressed_obs),
        score=self._get_player_score(compressed_obs),
        victory_type=self._detect_victory_type(compressed_obs),
        victory_progress=self._calculate_victory_progress(compressed_obs),
        strategic_summary=strategic_summary,
        prioritized_actions=prioritized_actions,
    )

    return prompt

  def _state_to_dict(self, state) -> Dict[str, Any]:
    """Convert FreeCivState object to dictionary.

    Args:
        state: FreeCivState object

    Returns:
        Dictionary representation of the state
    """
    # Simple conversion - in practice this would be more comprehensive
    players_raw = getattr(state, "players", {})

    # Convert players to proper dictionary format
    players_dict = {}
    if isinstance(players_raw, dict):
      for player_id, player_obj in players_raw.items():
        if hasattr(player_obj, "score"):
          players_dict[player_id] = {
              "score": getattr(player_obj, "score", 0),
              "gold": getattr(player_obj, "gold", 0),
              "name": getattr(player_obj, "name", f"Player {player_id}"),
          }
        else:
          players_dict[player_id] = player_obj

    return {
        "turn": getattr(state, "turn", 0),
        "players": players_dict,
        "units": getattr(state, "units", []),
        "cities": getattr(state, "cities", []),
    }

  def _detect_game_phase(self, obs: Dict[str, Any]) -> str:
    """Detect current game phase based on turn number and state.

    Args:
        obs: Observation dictionary

    Returns:
        Game phase string ('early_game', 'mid_game', 'late_game')
    """
    turn = obs.get("turn", 0)

    if turn <= EARLY_GAME_TURN_LIMIT:
      return "early_game"
    if turn <= MID_GAME_TURN_LIMIT:
      return "mid_game"
    return "late_game"

  def _get_player_name(self, obs: Dict[str, Any]) -> str:
    """Get current player name from observation.

    Args:
        obs: Observation dictionary

    Returns:
        Player name string
    """
    players = obs.get("players", {})
    if players and 1 in players:
      return players[1].get("name", "Romans")
    return "Romans"

  def _get_position_string(self, obs: Dict[str, Any]) -> str:
    """Get relative position string.

    Args:
        obs: Observation dictionary

    Returns:
        Position description string
    """
    players = obs.get("players", {})
    if len(players) <= 1:
      return "Solo game"

    # Simple position calculation
    our_score = players.get(1, {}).get("score", 0)
    scores = [p.get("score", 0) for p in players.values()]
    rank = sorted(scores, reverse=True).index(our_score) + 1

    if rank == 1:
      return "1st place"
    if rank == 2:
      return "2nd place"
    if rank == 3:
      return "3rd place"
    return f"{rank}th place"

  def _get_player_score(self, obs: Dict[str, Any]) -> int:
    """Get current player score.

    Args:
        obs: Observation dictionary

    Returns:
        Player score
    """
    players = obs.get("players", {})
    score = players.get(1, {}).get("score", 0)
    # Handle mock objects or other non-int types
    if hasattr(score, "return_value"):
      return getattr(score, "return_value", 0)
    try:
      return int(score)
    except (TypeError, ValueError):
      return 0

  def _calculate_victory_progress(self, obs: Dict[str, Any]) -> int:
    """Calculate progress toward victory condition.

    Args:
        obs: Observation dictionary

    Returns:
        Victory progress percentage (0-100)
    """
    # Simple heuristic based on score and turn
    turn_raw = obs.get("turn", 0)
    # Handle mock objects or other non-int types for turn
    if hasattr(turn_raw, "return_value"):
      turn = getattr(turn_raw, "return_value", 0)
    else:
      try:
        turn = int(turn_raw)
      except (TypeError, ValueError):
        turn = 0

    score = self._get_player_score(obs)

    # Rough calculation - in practice this would be more sophisticated
    progress = min(100, (score // 10) + (turn // 5))
    return max(0, progress)

  def _detect_victory_type(self, obs: Dict[str, Any]) -> str:
    """Detect the most promising victory type based on game state.

    Args:
        obs: Observation dictionary

    Returns:
        Victory type string
    """
    turn = obs.get("turn", 0)
    our_cities = obs.get("cities", [])
    all_units = obs.get("units", [])
    our_player_id = 1

    # Count our military units
    our_units = [u for u in all_units if u.get("owner") == our_player_id]
    military_units = [
        u
        for u in our_units
        if u.get("type", "").lower()
        in ["warrior", "archer", "phalanx", "legion"]
    ]

    # Early game - focus on expansion
    if turn < EARLY_GAME_TURN_LIMIT:
      return "Expansion Victory"

    # Analyze our strengths to determine best victory path
    military_strength = len(military_units) / max(len(our_units), 1)
    city_count = len(our_cities)

    # High military ratio suggests domination victory
    if (
        military_strength > MILITARY_STRENGTH_THRESHOLD
        and len(military_units) > MIN_MILITARY_UNITS_FOR_DOMINATION
    ):
      return "Domination Victory"

    # Many cities suggest economic/cultural victory
    if city_count >= CITY_COUNT_FOR_VICTORY:
      if turn > MID_GAME_TURN_LIMIT:
        return "Cultural Victory"
      return "Economic Victory"

    # Late game with few cities - likely science victory
    if turn > ENDGAME_TURN_THRESHOLD:
      return "Science Victory"

    # Default fallback
    return "Balanced Strategy"
