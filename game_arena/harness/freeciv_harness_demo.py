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

"""Demo of FreeCiv, prompt generation, model generation, and parser together."""

import os
import termcolor
from absl import app, flags

from game_arena.harness import (freeciv_state, llm_parsers,
                                model_generation_sdk, parsers,
                                prompt_generation, prompts, tournament_util)
from game_arena.harness.freeciv_client import FreeCivClient

colored = termcolor.colored

_NUM_MOVES = flags.DEFINE_integer(
    "num_moves",
    10,
    "Number of moves to play.",
)

_GEMINI_MODEL = flags.DEFINE_string(
    "gemini_model",
    "gemini-2.5-flash",
    "Gemini model to play as player one.",
)

_OPENAI_MODEL = flags.DEFINE_string(
    "openai_model",
    "gpt-4.1",
    "OpenAI model to play as player two.",
)

_PARSER_CHOICE = flags.DEFINE_enum_class(
    "parser_choice",
    tournament_util.ParserChoice.RULE_THEN_SOFT,
    tournament_util.ParserChoice,
    "Move parser to use.",
)

_FREECIV_SERVER_URL = flags.DEFINE_string(
    "freeciv_server_url",
    os.getenv("FREECIV_SERVER_URL", "http://localhost:8080"),
    "FreeCiv3D server URL.",
)

_FREECIV_WS_URL = flags.DEFINE_string(
    "freeciv_ws_url",
    os.getenv("FREECIV_WS_URL", "ws://localhost:4002"),
    "FreeCiv3D WebSocket URL.",
)


def main(_) -> None:
    # Set up FreeCiv client connection:
    freeciv_client = FreeCivClient(
        server_url=_FREECIV_SERVER_URL.value,
        ws_url=_FREECIV_WS_URL.value
    )

    print(colored("Connecting to FreeCiv3D server...", "blue"))
    try:
        freeciv_client.connect()
        print(colored("Connected successfully!", "green"))
    except Exception as e:
        print(colored(f"Failed to connect: {e}", "red"))
        return

    # Get initial game state
    raw_state = freeciv_client.get_game_state()
    freeciv_game_state = freeciv_state.FreeCivState(raw_state)

    # Set up prompt generator:
    prompt_generator = prompt_generation.PromptGeneratorText()
    prompt_template = prompts.PromptTemplate.NO_LEGAL_ACTIONS

    # Set up model generation:
    model_player_one = model_generation_sdk.AIStudioModel(
        model_name=_GEMINI_MODEL.value
    )
    model_player_two = model_generation_sdk.OpenAIChatCompletionsModel(
        model_name=_OPENAI_MODEL.value
    )

    # Set up parser:
    match _PARSER_CHOICE.value:
        case tournament_util.ParserChoice.RULE_THEN_SOFT:
            from game_arena.harness.freeciv_parsers import create_freeciv_parser_chain
            parser = create_freeciv_parser_chain()
        case tournament_util.ParserChoice.LLM_ONLY:
            parser_model = model_generation_sdk.AIStudioModel(
                model_name="gemini-2.5-flash"
            )
            parser = llm_parsers.LLMParser(
                model=parser_model,
                instruction_config=llm_parsers.FreeCivInstructionConfig_V0,
            )
        case _:
            raise ValueError(f"Unsupported parser choice: {_PARSER_CHOICE.value}")

    for move_number in range(_NUM_MOVES.value):
        print(f"\nPre-move state summary: Turn {freeciv_game_state.turn}, Phase {freeciv_game_state.phase}")
        print(f"Current player: {freeciv_game_state.current_player()}")

        if freeciv_game_state.is_terminal():
            print(colored("Game is terminal, ending move loop.", "red"))
            break

        print(colored(f"Commencing move {move_number}...", "green"))

        # 1. Generate the prompt from the game state:
        current_player_id = freeciv_game_state.current_player() + 1  # Convert to FreeCiv player ID
        observation = freeciv_game_state.to_observation(current_player_id, format="enhanced")
        legal_actions = freeciv_game_state.get_legal_actions(current_player_id)

        prompt_substitutions = {
            "readable_state_str": f"Turn {freeciv_game_state.turn}: {len(legal_actions)} actions available",
            "move_history": f"Turn {freeciv_game_state.turn} - {freeciv_game_state.phase} phase",
            "player_name": f"Player {current_player_id}",
            "move_notation": "FreeCiv actions: unit_move, city_production, etc.",
            "notation": "FreeCiv game state with units, cities, and diplomacy",
        }

        # Add strategic context from observation
        if "strategic" in observation:
            strategic_info = observation["strategic"]
            prompt_substitutions["readable_state_str"] += f"\nScore: {strategic_info.get('scoreboard', {}).get('player', 0)}"
            if strategic_info.get("economy"):
                prompt_substitutions["readable_state_str"] += f"\nGold: {strategic_info['economy']['gold']}"

        # Add tactical context
        if "tactical" in observation:
            tactical_info = observation["tactical"]
            unit_counts = tactical_info.get("unit_counts", {})
            prompt_substitutions["readable_state_str"] += f"\nUnits: {unit_counts.get('friendly', 0)} friendly, {unit_counts.get('enemy', 0)} enemy"
            prompt_substitutions["readable_state_str"] += f"\nThreats: {tactical_info.get('threats', 0)}"

        prompt = prompt_generator.generate_prompt_with_text_only(
            prompt_template=prompt_template,
            game_short_name="freeciv",
            **prompt_substitutions,
        )
        print(colored(f"Formatted prompt: {prompt.prompt_text}", "blue"))

        # 2. Call the model:
        if freeciv_game_state.current_player() == 0:
            model = model_player_one
        else:
            model = model_player_two
        response = model.generate_with_text_input(prompt)
        print(
            colored(
                f"Model player {freeciv_game_state.current_player()} main response:"
                f" {response.main_response}",
                "yellow",
            )
        )

        # 3. Parse the model response:
        # Convert legal actions to strings for parser
        legal_action_strings = [freeciv_game_state._action_to_string(action) for action in legal_actions]

        parser_input = parsers.TextParserInput(
            text=response.main_response,
            state_str=str(observation),  # Use observation as state string
            legal_moves=legal_action_strings,
            player_number=freeciv_game_state.current_player(),
        )
        parser_output = parser.parse(parser_input)
        if parser_output is None:
            print(colored("Parser output is None, ending game.", "red"))
            break
        else:
            print(colored(f"Parser output is {parser_output}.", "magenta"))

        # 4. Apply the move:
        try:
            # Find the action that matches the parser output
            selected_action = None
            for i, action_str in enumerate(legal_action_strings):
                if action_str == parser_output:
                    selected_action = legal_actions[i]
                    break

            if selected_action:
                freeciv_game_state.apply_action(selected_action)
                # Send action to FreeCiv server
                freeciv_client.submit_action(selected_action)
                print(colored(f"Applied action: {selected_action.action_type}", "cyan"))
            else:
                print(colored(f"Could not find matching action for: {parser_output}", "red"))
                break

        except Exception as e:
            print(colored(f"Error applying action: {e}", "red"))
            break

        # 5. Update game state from server:
        try:
            raw_state = freeciv_client.get_game_state()
            freeciv_game_state = freeciv_state.FreeCivState(raw_state)
        except Exception as e:
            print(colored(f"Error updating game state: {e}", "red"))
            break

    print(colored("Demo completed!", "green"))
    freeciv_client.disconnect()


if __name__ == "__main__":
    app.run(main)