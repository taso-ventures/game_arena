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

"""Agent class for submitting to Kaggle Game Arena Simulation Environments."""

import abc
import dataclasses
import json
import random
import traceback
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic, Protocol, TypeAlias, TypedDict, TypeVar

import pyspiel
from absl import logging

from game_arena.harness import (game_notation_examples, model_generation,
                                parsers, prompt_generation, prompts, rethink,
                                tournament_util)

INVALID_ACTION = pyspiel.INVALID_ACTION  # -1
ERROR_ACTION_INT = -2


class CustomJSONEncoder(json.JSONEncoder):
    """A custom JSON encoder that handles non-serializable types from various LLM libraries."""

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if hasattr(o, "to_dict") and callable(o.to_dict):
            return o.to_dict()
        try:
            return super().default(o)
        except TypeError:
            return str(o)


KaggleActionT = TypeVar("KaggleActionT")
KaggleSpielActionT = TypeVar(
    "KaggleSpielActionT", "KaggleSpielAction", "KaggleSpielActionWithExtras"
)


PromptBuilder: TypeAlias = Callable[[pyspiel.State], str]
ResponseParser: TypeAlias = Callable[
    [tournament_util.GenerateReturn, pyspiel.State], str
]


class KaggleAgent(Protocol, Generic[KaggleActionT]):
    """Kaggle agent interface."""

    def __call__(
        self,
        observation: Mapping[str, Any],
        configuration: Mapping[str, Any],
        **kwargs,
    ) -> KaggleActionT: ...


class KaggleSpielAction(TypedDict):
    """Action required by the Kaggle simulation environment Open Spiel wrapper."""

    submission: int


class KaggleSpielActionWithExtras(KaggleSpielAction):
    """Action with additional information."""

    actionString: str | None  # pylint: disable=invalid-name
    thoughts: str | None  # This goes into the "thoughts" viewer in the Kaggle UI.
    status: str | None  # pylint: disable=invalid-name
    generate_returns: Sequence[str] = dataclasses.field(default_factory=list)


class KaggleSpielAgent(
    KaggleAgent[KaggleSpielActionT], abc.ABC, Generic[KaggleSpielActionT]
):
    """Kaggle agent base class."""

    @abc.abstractmethod
    def __call__(
        self,
        observation: Mapping[str, Any],
        configuration: Mapping[str, Any],
        **kwargs,
    ) -> KaggleSpielActionT: ...


class LLMAgent(KaggleAgent[KaggleActionT], Generic[KaggleActionT]):
    """LLM agent for Kaggle Game Arena Simulation Environments."""

    model: model_generation.Model
    # TODO(google-deepmind): Align API with existing abstractions. The goal is to
    # have a generic agent __call__ function that performs three main steps:
    # 1. Map from observation to prompt.
    # 2. Call the model.
    # 3. Parse the model's response into a submittable action.
    # Users need only specify a model, and define the prompt builder and response
    # parser functions. No game specific logic should be required in the agent.
    # TODO(google-deepmind): We currently require access to the pyspiel.State, which
    # is currently not present in the agent observation. For chess, the
    # observation consists of the FEN string, which allows us to reconstruct the
    # pyspiel.State. However, this is not a general approach, and will not work
    # for other games. We can either add the serialized state to the observation
    # in the Kaggle environment, or drop the pyspiel.State dependency which will
    # be possible with the Open Spiel 2.0 updates.
    prompt_builder: PromptBuilder
    response_parser: ResponseParser


# TODO(John Schultz): Make agent fully generic across games.
class ChessLLMAgent(
    KaggleSpielAgent[KaggleSpielActionWithExtras],
    LLMAgent[KaggleSpielActionWithExtras],
):
    """LLM agent for chess.

    An agent that uses a large language model to play chess. It takes an
    observation of the game state, builds a prompt, queries the model, and parses
    the model's response to determine its action.

    Attributes:
      model: The LLM to use for generating actions.
      prompt_builder: A function that takes a `pyspiel.State` and returns a prompt
        string for the model.
      response_parser: A function that parses the model's response and returns an
        action string.
      max_model_calls: If set, the agent will start making random moves after this
        many calls to the model (used for testing).
      fallback_to_random_move: If True, the agent will take a random action if the
        action string returned by the model does not parse to a valid action.
      seed: The seed for the random number generator used for fallbacks.
      num_model_calls: The number of times the model has been called.
    """

    def __init__(
        self,
        model: model_generation.Model,
        prompt_builder: PromptBuilder,
        response_parser: ResponseParser,
        max_model_calls: int | None = None,
        fallback_to_random_move: bool = False,
        seed: int | None = None,
    ):
        super().__init__()

        self.model = model
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser
        self.max_model_calls = max_model_calls
        self.fallback_to_random_move = fallback_to_random_move
        self._rng = random.Random(seed)
        self._num_model_calls = 0

    @property
    def num_model_calls(self) -> int:
        """The number of times the model (not the agent) has been called."""
        return self._num_model_calls

    def __call__(
        self,
        observation: Mapping[str, Any],
        configuration: Mapping[str, Any],
        **kwargs,
    ) -> KaggleSpielActionWithExtras:
        """Returns an action given an observation of the current game state."""
        del configuration, kwargs
        serialized_game_and_state = observation.get("serializedGameAndState")
        if not serialized_game_and_state:
            return KaggleSpielActionWithExtras(
                submission=INVALID_ACTION,
                actionString=None,
                thoughts=None,
                status="Setup step; model not called.",
                generate_returns=[],
            )
        _, pyspiel_state = pyspiel.deserialize_game_and_state(serialized_game_and_state)

        if self.max_model_calls and self.num_model_calls >= self.max_model_calls:
            status = (
                f"MAX MODEL CALLS (N={self.num_model_calls}) REACHED;"
                " selecting random move."
            )
            logging.info(status)
            legal_moves = observation.get("legalActions") or []
            action_int = self._rng.choice(legal_moves)
            action_str = pyspiel_state.action_to_string(action_int)
            return KaggleSpielActionWithExtras(
                submission=action_int,
                actionString=action_str,
                thoughts=None,
                status=status,
                generate_returns=[],
            )

        prompt = self.prompt_builder(pyspiel_state)
        model_input = tournament_util.ModelTextInput(prompt_text=prompt)

        parsed_action_str = None
        action_int = INVALID_ACTION
        response = None
        main_response = None
        try:
            logging.info("CALLING LLM")
            self._num_model_calls += 1
            response = self.model.generate_with_text_input(model_input)
            logging.info("RESPONSE:")
            logging.info(response.main_response)
        except Exception as e:  # pylint: disable=broad-except
            logging.error("ERROR CALLING LLM")
            logging.exception(e)
            pass
        if response is None:
            logging.error("NO RESPONSE FROM LLM")
            return KaggleSpielActionWithExtras(
                submission=INVALID_ACTION,
                actionString=None,
                thoughts=None,
                status="Model non-responsive.",
                generate_returns=[],
            )

        try:
            main_response = response.main_response
            parsed_action_str = self.response_parser(response, pyspiel_state)
            action_int = pyspiel_state.string_to_action(parsed_action_str)
            logging.info("PARSED RESPONSE: %s %s", parsed_action_str, action_int)
        except Exception as e:  # pylint: disable=broad-except
            logging.error("ERROR PARSING LLM RESPONSE")
            logging.exception(e)
            pass

        legal_actions = observation.get("legalActions") or []
        if not legal_actions:
            logging.warning("NO LEGAL ACTIONS FOUND")
        if (
            self.fallback_to_random_move
            and legal_actions
            and action_int not in legal_actions
        ):
            logging.info("INVALID MOVE FROM LLM; overriding with random move.")
            action_int = self._rng.choice(legal_actions)

        logging.debug(
            "Returning: %s %s %s", action_int, parsed_action_str, main_response
        )

        return KaggleSpielActionWithExtras(
            submission=action_int,
            actionString=parsed_action_str,
            thoughts=main_response,
            status=None,
            generate_returns=[response],
        )


# TODO(John Schultz): Remove LLMAgent abstraction in favor of a generic Sampler
# agent, and in the process remove these default prompt and response parsers.
prompt_generator = prompt_generation.PromptGeneratorText()
chained_parser = parsers.ChainedMoveParser(
    [parsers.RuleBasedMoveParser(), parsers.SoftMoveParser("chess")]
)


def default_chess_prompt_builder(
    pyspiel_state: pyspiel.State,
) -> str:
    """Builds the text prompt for the LLM agent."""
    chess_notations = game_notation_examples.GAME_SPECIFIC_NOTATIONS["chess"]
    prompt_substitutions = {
        "readable_state_str": tournament_util.convert_to_readable_state(
            game_short_name="chess",
            state_str=pyspiel_state.to_string(),
            current_player=pyspiel_state.current_player(),
        ),
        "move_history": (
            tournament_util.get_action_string_history(pyspiel_state) or "None"
        ),
        "player_name": chess_notations["player_map"][pyspiel_state.current_player()],
        "move_notation": chess_notations["move_notation"],
        "notation": chess_notations["state_notation"],
    }
    prompt = prompt_generator.generate_prompt_with_text_only(
        prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS,
        game_short_name="chess",
        **prompt_substitutions,
    )
    return prompt.prompt_text


def default_response_parser(
    response: tournament_util.GenerateReturn,
    pyspiel_state: pyspiel.State,
) -> str:
    """Parses the response from the LLM."""
    parser_input = parsers.TextParserInput(
        text=response.main_response,
        state_str=pyspiel_state.to_string(),
        legal_moves=parsers.get_legal_action_strings(pyspiel_state),
        player_number=pyspiel_state.current_player(),
    )
    llm_choice_str = chained_parser.parse(parser_input)
    return llm_choice_str or ""


# TODO(John Schultz): Add a generic sampler agent. One problem is that different
# samplers have different call functions.
class ChessRethinkAgent(KaggleSpielAgent[KaggleSpielActionWithExtras]):
    """Rethink agent for chess."""

    def __init__(
        self,
        sampler: rethink.RethinkSampler,
        prompt_template: prompts.PromptTemplate,
        max_sampler_calls: int | None = None,
        random_move_fallback: bool = False,
        seed: int | None = None,
    ):
        super().__init__()
        self.sampler = sampler
        self.prompt_template = prompt_template
        self.max_sampler_calls = max_sampler_calls
        self.random_move_fallback = random_move_fallback
        self._rng = random.Random(seed)
        self._num_sampler_calls = 0

    @property
    def num_sampler_calls(self) -> int:
        """The number of times the sampler (not the model or agent) has been called."""
        return self._num_sampler_calls

    def __call__(
        self,
        observation: Mapping[str, Any],
        configuration: Mapping[str, Any],
        **kwargs,
    ) -> KaggleSpielActionWithExtras:
        """Returns an action given an observation of the current game state."""
        del configuration, kwargs
        serialized_game_and_state = observation.get("serializedGameAndState")
        if not serialized_game_and_state:
            return KaggleSpielActionWithExtras(
                submission=INVALID_ACTION,
                actionString=None,
                thoughts=None,
                status="OK; Setup step; model not called.",
                generate_returns=[],
            )
        _, pyspiel_state = pyspiel.deserialize_game_and_state(serialized_game_and_state)

        if self.max_sampler_calls and self.num_sampler_calls >= self.max_sampler_calls:
            status = (
                f"OK; MAX SAMPLER CALLS (N={self.num_sampler_calls}) REACHED;"
                " selecting random move"
            )
            logging.info(status)
            legal_moves = observation.get("legalActions") or []
            action_int = self._rng.choice(legal_moves)
            action_str = pyspiel_state.action_to_string(action_int)
            return KaggleSpielActionWithExtras(
                submission=action_int,
                actionString=action_str,
                thoughts=None,
                status=status,
                generate_returns=[],
            )

        prompt_substitutions = {
            "readable_state_str": tournament_util.convert_to_readable_state(
                game_short_name="chess",
                state_str=pyspiel_state.to_string(),
                current_player=pyspiel_state.current_player(),
            ),
            "move_history": (
                tournament_util.get_action_string_history(pyspiel_state) or "None"
            ),
            "player_name": game_notation_examples.GAME_SPECIFIC_NOTATIONS["chess"][
                "player_map"
            ][pyspiel_state.current_player()],
            "move_notation": game_notation_examples.GAME_SPECIFIC_NOTATIONS["chess"][
                "move_notation"
            ],
            "notation": game_notation_examples.GAME_SPECIFIC_NOTATIONS["chess"][
                "state_notation"
            ],
        }

        try:
            logging.info("CALLING SAMPLER")
            self._num_sampler_calls += 1
            sampler_output = self.sampler.sample_action_with_text_and_state_input(
                pyspiel_state,
                self.prompt_template,
                **prompt_substitutions,
            )
            logging.info("FIRST RESPONSE:")
            logging.info(sampler_output.generate_returns[0].main_response)
            logging.info("SAMPLED ACTION:")
            logging.info(sampler_output.action)
        except Exception as e:  # pylint: disable=broad-except
            logging.error("ERROR CALLING SAMPLER")
            logging.exception(e)
            return KaggleSpielActionWithExtras(
                submission=ERROR_ACTION_INT,
                actionString=None,
                thoughts=None,
                status=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                generate_returns=[],
            )

        main_response = ""
        for i, generate_return in enumerate(sampler_output.generate_returns):
            if i == 0:
                main_response = generate_return.main_response
            else:
                main_response += (
                    "\n\n" + "=" * 10 + f" Rethink Attempt #{i} " + "=" * 10
                )
                main_response += f"\n\n{generate_return.main_response}"
        logging.info("--ALL RESPONSES--")
        logging.info(main_response)

        generate_returns_jsons = []
        try:
            generate_returns_jsons = [
                json.dumps(generate_return.to_dict(), indent=2, cls=CustomJSONEncoder)
                for generate_return in sampler_output.generate_returns
            ]
        except Exception as e:  # pylint: disable=broad-except
            logging.error("ERROR DUMPING GENERATE RETURNS")
            logging.exception(e)

        parsed_action_str = sampler_output.action
        if sampler_output.move_type == tournament_util.MoveType.LEGAL:
            try:
                action_int = pyspiel_state.string_to_action(parsed_action_str)
                logging.info("PARSED RESPONSE: %s %s", parsed_action_str, action_int)
                return KaggleSpielActionWithExtras(
                    submission=action_int,
                    actionString=parsed_action_str,
                    thoughts=main_response,
                    status="OK",
                    generate_returns=generate_returns_jsons,
                )
            except Exception as e:  # pylint: disable=broad-except
                logging.error("ERROR SHOULD BE LEGAL BUT CONVERSION FAILED")
                logging.exception(e)
                return KaggleSpielActionWithExtras(
                    submission=INVALID_ACTION,
                    actionString=parsed_action_str,
                    thoughts=main_response,
                    status=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                    generate_returns=generate_returns_jsons,
                )
        else:
            return KaggleSpielActionWithExtras(
                submission=INVALID_ACTION,
                actionString=parsed_action_str,
                thoughts=main_response,
                status="OK; Submitting invalid action.",
                generate_returns=generate_returns_jsons,
            )


def build_default_rethink_agent(
    model: model_generation.Model,
) -> ChessRethinkAgent:
    """Builds a rethink agent with default settings for a given model."""
    sampler = rethink.RethinkSampler(
        model=model,
        strategy=tournament_util.SamplerChoice.RETHINK_WITH_ENV,
        num_max_rethinks=3,
        move_parser=parsers.RuleBasedMoveParser(),
        legality_parser=parsers.SoftMoveParser("chess"),
        game_short_name="chess",
        prompt_generator=prompt_generation.PromptGeneratorText(),
        rethink_template=None,
    )
    agent = ChessRethinkAgent(
        sampler=sampler,
        prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )
    return agent
