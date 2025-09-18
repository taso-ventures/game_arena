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

"""Tests for rethinking sampler."""

from unittest import mock

import pyspiel
from absl.testing import absltest

from game_arena.harness import (
    model_generation,
    parsers,
    prompt_generation,
    prompts,
    rethink,
    tournament_util,
)


def _create_mock_state():
  mock_state = mock.create_autospec(pyspiel.State, instance=True)
  mock_state.to_string.return_value = "fen_string"
  mock_state.current_player.return_value = 0
  return mock_state


def _fake_legality_parser_parse(parser_input):
  if parser_input.text in parser_input.legal_moves:
    return parser_input.text
  return None


class _DummyPromptGeneratorSupportsText(
    prompt_generation.PromptGeneratorSupportsText
):

  def generate_prompt_with_text_only(self, **kwargs):
    pass


class _DummyPromptGeneratorSupportsImageText(
    prompt_generation.PromptGeneratorSupportsImageText
):

  def generate_prompt_with_image_text(self, **kwargs):
    pass

  def generate_prompt_with_text_only(self, **kwargs):
    pass


@mock.patch("game_arena.harness.parsers.get_legal_action_strings")
class RethinkSamplerTest(absltest.TestCase):

  maxDiff = None  # Print out the full difference if a test fails.

  def test_sample_action_text_only_success_first_try(
      self, mock_get_legal_action_strings
  ):
    """Tests the case where the first action is legal."""
    mock_get_legal_action_strings.return_value = ["e4"]
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    mock_state = _create_mock_state()

    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=tournament_util.SamplerChoice.RETHINK_WITH_ENV,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        game_short_name="chess",
        prompt_generator=mock_prompt_generator,
        rethink_template="",
    )

    mock_generate_fn = mock.Mock()
    generate_return = tournament_util.GenerateReturn(
        main_response="e4", main_response_and_thoughts="e4"
    )
    mock_generate_fn.return_value = generate_return
    mock_prompt_generator.generate_prompt_with_text_only.return_value = "prompt"
    mock_move_parser.parse.return_value = "e4"
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    output = sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        state=mock_state,
        prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    self.assertEqual(output.action, "e4")
    self.assertEqual(output.generate_returns, [generate_return])
    self.assertDictEqual(
        output.auxiliary_outputs,
        {
            "parsed_action_attempt_0": "e4",
            "maybe_legal_action_attempt_0": "e4",
            "rethink_prompt_attempt_0": "",
        },
    )
    mock_generate_fn.assert_called_once_with("prompt")
    mock_move_parser.parse.assert_called_once()
    mock_legality_parser.parse.assert_called_once()

  def test_sample_action_text_only_single_rethink(
      self, mock_get_legal_action_strings
  ):
    """Tests the case where the first action is illegal and the second is legal."""
    mock_get_legal_action_strings.return_value = ["e4"]
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    mock_state = _create_mock_state()

    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=tournament_util.SamplerChoice.RETHINK_WITH_ENV,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        game_short_name="chess",
        prompt_generator=mock_prompt_generator,
        rethink_template="",
    )

    mock_generate_fn = mock.Mock()
    illegal_return = tournament_util.GenerateReturn(
        main_response="illegal_move", main_response_and_thoughts="illegal_move"
    )
    legal_return = tournament_util.GenerateReturn(
        main_response="e4", main_response_and_thoughts="e4"
    )
    mock_generate_fn.side_effect = [illegal_return, legal_return]
    mock_prompt_generator.generate_prompt_with_text_only.side_effect = [
        "prompt1",
        "prompt2",
    ]
    mock_move_parser.parse.side_effect = ["illegal_move", "e4"]
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    output = sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        state=mock_state,
        prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    self.assertEqual(output.action, "e4")
    self.assertEqual(output.generate_returns, [illegal_return, legal_return])
    self.assertDictEqual(
        output.auxiliary_outputs,
        {
            "parsed_action_attempt_0": "illegal_move",
            "maybe_legal_action_attempt_0": None,
            "rethink_prompt_attempt_0": "",
            "parsed_action_attempt_1": "e4",
            "maybe_legal_action_attempt_1": "e4",
            "rethink_prompt_attempt_1": (
                rethink._RETHINK_WITH_ENV_ILLEGAL_TEMPLATE.format(
                    last_move="illegal_move"
                )
            ),
        },
    )
    self.assertEqual(mock_generate_fn.call_count, 2)
    self.assertEqual(mock_move_parser.parse.call_count, 2)
    self.assertEqual(mock_legality_parser.parse.call_count, 2)

  def test_sample_action_text_only_unparsable_then_legal(
      self, mock_get_legal_action_strings
  ):
    """Tests the case where the first action is unparsable and the second is legal."""
    mock_get_legal_action_strings.return_value = ["e4"]
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    mock_state = _create_mock_state()

    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=tournament_util.SamplerChoice.RETHINK_WITH_ENV,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        game_short_name="chess",
        prompt_generator=mock_prompt_generator,
        rethink_template="",
    )

    mock_generate_fn = mock.Mock()
    unparsable_return = tournament_util.GenerateReturn(
        main_response="unparsable", main_response_and_thoughts="unparsable"
    )
    legal_return = tournament_util.GenerateReturn(
        main_response="e4", main_response_and_thoughts="e4"
    )
    mock_generate_fn.side_effect = [unparsable_return, legal_return]
    mock_prompt_generator.generate_prompt_with_text_only.side_effect = [
        "prompt1",
        "prompt2",
    ]
    mock_move_parser.parse.side_effect = [None, "e4"]
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    output = sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        state=mock_state,
        prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    self.assertEqual(output.action, "e4")
    self.assertEqual(output.generate_returns, [unparsable_return, legal_return])
    self.assertDictEqual(
        output.auxiliary_outputs,
        {
            "parsed_action_attempt_0": None,
            "maybe_legal_action_attempt_0": None,
            "rethink_prompt_attempt_0": "",
            "parsed_action_attempt_1": "e4",
            "maybe_legal_action_attempt_1": "e4",
            "rethink_prompt_attempt_1": (
                rethink._RETHINK_WITH_ENV_UNPARSABLE_TEMPLATE.format(
                    generation="unparsable"
                )
            ),
        },
    )
    self.assertEqual(mock_generate_fn.call_count, 2)
    self.assertEqual(mock_move_parser.parse.call_count, 2)
    self.assertEqual(mock_legality_parser.parse.call_count, 1)

  def test_sample_action_text_only_fail_after_max_rethinks(
      self, mock_get_legal_action_strings
  ):
    """Tests the case where all generated actions are illegal."""
    mock_get_legal_action_strings.return_value = ["d4"]
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    mock_state = _create_mock_state()

    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=tournament_util.SamplerChoice.RETHINK_WITH_ENV,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        game_short_name="chess",
        prompt_generator=mock_prompt_generator,
        rethink_template="",
    )

    mock_generate_fn = mock.Mock()
    illegal_return1 = tournament_util.GenerateReturn(
        main_response="illegal1", main_response_and_thoughts="illegal1"
    )
    illegal_return2 = tournament_util.GenerateReturn(
        main_response="illegal2", main_response_and_thoughts="illegal2"
    )
    mock_generate_fn.side_effect = [illegal_return1, illegal_return2]
    mock_prompt_generator.generate_prompt_with_text_only.side_effect = [
        "prompt1",
        "prompt2",
    ]
    mock_move_parser.parse.side_effect = ["illegal1", "illegal2"]
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    output = sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        state=mock_state,
        prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    self.assertEqual(output.action, "illegal2")
    self.assertEqual(
        output.generate_returns, [illegal_return1, illegal_return2]
    )
    self.assertDictEqual(
        output.auxiliary_outputs,
        {
            "parsed_action_attempt_0": "illegal1",
            "maybe_legal_action_attempt_0": None,
            "rethink_prompt_attempt_0": "",
            "parsed_action_attempt_1": "illegal2",
            "maybe_legal_action_attempt_1": None,
            "rethink_prompt_attempt_1": (
                rethink._RETHINK_WITH_ENV_ILLEGAL_TEMPLATE.format(
                    last_move="illegal1"
                )
            ),
        },
    )
    self.assertEqual(mock_generate_fn.call_count, 2)
    self.assertEqual(mock_move_parser.parse.call_count, 2)
    self.assertEqual(mock_legality_parser.parse.call_count, 2)

  def test_rethink_strategy(self, mock_get_legal_action_strings):
    """Tests the RETHINK strategy."""
    mock_get_legal_action_strings.return_value = ["e4"]
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    mock_state = _create_mock_state()

    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=tournament_util.SamplerChoice.RETHINK,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        game_short_name="chess",
        prompt_generator=mock_prompt_generator,
        rethink_template=None,
    )

    mock_generate_fn = mock.Mock()
    illegal_return = tournament_util.GenerateReturn(
        main_response="illegal_move", main_response_and_thoughts="illegal_move"
    )
    legal_return = tournament_util.GenerateReturn(
        main_response="e4", main_response_and_thoughts="e4"
    )
    mock_generate_fn.side_effect = [illegal_return, legal_return]
    mock_prompt_generator.generate_prompt_with_text_only.side_effect = [
        "prompt1",
        "prompt2",
    ]
    mock_move_parser.parse.side_effect = ["illegal_move", "e4"]
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        state=mock_state,
        prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
        foo="bar",
    )

    self.assertEqual(
        mock_prompt_generator.generate_prompt_with_text_only.call_args_list[0],
        mock.call(
            prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
            game_short_name="chess",
            rethink_prompt="",
            foo="bar",
        ),
    )
    self.assertEqual(
        mock_prompt_generator.generate_prompt_with_text_only.call_args_list[1],
        mock.call(
            prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
            game_short_name="chess",
            rethink_prompt="",
            foo="bar",
        ),
    )

  @mock.patch("game_arena.harness.rethink_fn.rule_explain_illegal_move")
  def test_rethink_with_env_rule_strategy(
      self, mock_rule_explain_illegal_move, mock_get_legal_action_strings
  ):
    """Tests the RETHINK_WITH_ENV_RULE strategy."""
    mock_get_legal_action_strings.return_value = ["e4"]
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    mock_state = _create_mock_state()
    mock_rule_explain_illegal_move.return_value = "some reason"

    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=tournament_util.SamplerChoice.RETHINK_WITH_ENV_RULE,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        game_short_name="chess",
        prompt_generator=mock_prompt_generator,
        rethink_template="",
    )

    mock_generate_fn = mock.Mock()
    illegal_return = tournament_util.GenerateReturn(
        main_response="illegal_move", main_response_and_thoughts="illegal_move"
    )
    legal_return = tournament_util.GenerateReturn(
        main_response="e4", main_response_and_thoughts="e4"
    )
    mock_generate_fn.side_effect = [illegal_return, legal_return]
    mock_prompt_generator.generate_prompt_with_text_only.side_effect = [
        "prompt1",
        "prompt2",
    ]
    mock_move_parser.parse.side_effect = ["illegal_move", "e4"]
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        state=mock_state,
        prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    self.assertEqual(
        mock_prompt_generator.generate_prompt_with_text_only.call_args_list[1],
        mock.call(
            prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
            game_short_name="chess",
            rethink_prompt=rethink._RETHINK_WITH_ENV_RULE_TEMPLATE.format(
                last_move="illegal_move",
                reason="some reason",
            ),
        ),
    )
    mock_rule_explain_illegal_move.assert_called_once_with(
        fen="fen_string", move_str="illegal_move"
    )

  def test_rethink_with_env_strategy(self, mock_get_legal_action_strings):
    """Tests the RETHINK_WITH_ENV strategy."""
    mock_get_legal_action_strings.return_value = ["e4"]
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    mock_state = _create_mock_state()

    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=tournament_util.SamplerChoice.RETHINK_WITH_ENV,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        game_short_name="chess",
        prompt_generator=mock_prompt_generator,
        rethink_template="",
    )

    mock_generate_fn = mock.Mock()
    illegal_return = tournament_util.GenerateReturn(
        main_response="illegal_move", main_response_and_thoughts="illegal_move"
    )
    legal_return = tournament_util.GenerateReturn(
        main_response="e4", main_response_and_thoughts="e4"
    )
    mock_generate_fn.side_effect = [illegal_return, legal_return]
    mock_prompt_generator.generate_prompt_with_text_only.side_effect = [
        "prompt1",
        "prompt2",
    ]
    mock_move_parser.parse.side_effect = ["illegal_move", "e4"]
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        state=mock_state,
        prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    self.assertEqual(
        mock_prompt_generator.generate_prompt_with_text_only.call_args_list[1],
        mock.call(
            prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
            game_short_name="chess",
            rethink_prompt=rethink._RETHINK_WITH_ENV_ILLEGAL_TEMPLATE.format(
                last_move="illegal_move"
            ),
        ),
    )

  def test_rethink_with_env_strategy_unparsable(
      self, mock_get_legal_action_strings
  ):
    """Tests the RETHINK_WITH_ENV strategy with unparsable response."""
    mock_get_legal_action_strings.return_value = ["e4"]
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    mock_state = _create_mock_state()

    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=tournament_util.SamplerChoice.RETHINK_WITH_ENV,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        game_short_name="chess",
        prompt_generator=mock_prompt_generator,
        rethink_template="",
    )

    mock_generate_fn = mock.Mock()
    unparsable_return = tournament_util.GenerateReturn(
        main_response="unparsable", main_response_and_thoughts="unparsable"
    )
    legal_return = tournament_util.GenerateReturn(
        main_response="e4", main_response_and_thoughts="e4"
    )
    mock_generate_fn.side_effect = [unparsable_return, legal_return]
    mock_prompt_generator.generate_prompt_with_text_only.side_effect = [
        "prompt1",
        "prompt2",
    ]
    mock_move_parser.parse.side_effect = [None, "e4"]
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        state=mock_state,
        prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    self.assertEqual(
        mock_prompt_generator.generate_prompt_with_text_only.call_args_list[1],
        mock.call(
            prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
            game_short_name="chess",
            rethink_prompt=rethink._RETHINK_WITH_ENV_UNPARSABLE_TEMPLATE.format(
                generation="unparsable"
            ),
        ),
    )

  @mock.patch("game_arena.harness.prompts.is_image_text")
  def test_image_based_workflow(
      self, mock_is_image_text, mock_get_legal_action_strings
  ):
    """Tests the image-based (multimodal) workflow."""
    mock_is_image_text.return_value = True
    mock_get_legal_action_strings.return_value = ["e4"]
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsImageText, instance=True
    )
    mock_state = _create_mock_state()

    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=tournament_util.SamplerChoice.RETHINK_WITH_ENV,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        game_short_name="chess",
        prompt_generator=mock_prompt_generator,
        rethink_template="",
    )

    mock_generate_fn = mock.Mock()
    generate_return = tournament_util.GenerateReturn(
        main_response="e4", main_response_and_thoughts="e4"
    )
    mock_generate_fn.return_value = generate_return
    mock_prompt_generator.generate_prompt_with_image_text.return_value = (
        "prompt"
    )
    mock_move_parser.parse.return_value = "e4"
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        state=mock_state,
        prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    mock_prompt_generator.generate_prompt_with_image_text.assert_called_once()

  def test_rethink_with_env_rule_wrong_game_error(
      self, mock_get_legal_action_strings
  ):
    """Tests ValueError for RETHINK_WITH_ENV_RULE with a non-chess game."""
    mock_get_legal_action_strings.return_value = []
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    mock_state = _create_mock_state()

    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=tournament_util.SamplerChoice.RETHINK_WITH_ENV_RULE,
        num_max_rethinks=1,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        game_short_name="go",
        prompt_generator=mock_prompt_generator,
        rethink_template="",
    )

    mock_generate_fn = mock.Mock()
    illegal_return = tournament_util.GenerateReturn(
        main_response="illegal_move", main_response_and_thoughts="illegal_move"
    )
    mock_generate_fn.return_value = illegal_return
    mock_prompt_generator.generate_prompt_with_text_only.return_value = "prompt"
    mock_move_parser.parse.return_value = "illegal_move"
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    with self.assertRaisesRegex(
        ValueError, "Only chess is supported for rule-based rethinking"
    ):
      sampler._sample_action(
          model_generate_fn=mock_generate_fn,
          state=mock_state,
          prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
      )

  def test_unsupported_strategy_error(self, mock_get_legal_action_strings):
    """Tests ValueError for an unsupported strategy."""
    mock_get_legal_action_strings.return_value = []
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )

    with self.assertRaisesRegex(ValueError, "Unsupported strategy"):
      rethink.RethinkSampler(
          model=mock_model,
          strategy="unsupported_strategy",
          num_max_rethinks=1,
          move_parser=mock_move_parser,
          legality_parser=mock_legality_parser,
          game_short_name="chess",
          prompt_generator=mock_prompt_generator,
          rethink_template="",
      )

  def test_rethink_with_env_illegal_history_strategy(
      self, mock_get_legal_action_strings
  ):
    """Tests the RETHINK_WITH_ENV_ILLEGAL_HISTORY strategy."""
    mock_get_legal_action_strings.return_value = ["e4"]
    mock_model = mock.create_autospec(model_generation.Model, instance=True)
    mock_move_parser = mock.create_autospec(parsers.TextParser, instance=True)
    mock_legality_parser = mock.create_autospec(
        parsers.TextParser, instance=True
    )
    mock_prompt_generator = mock.create_autospec(
        _DummyPromptGeneratorSupportsText, instance=True
    )
    mock_state = _create_mock_state()

    sampler = rethink.RethinkSampler(
        model=mock_model,
        strategy=tournament_util.SamplerChoice.RETHINK_WITH_ENV_ILLEGAL_HISTORY,
        num_max_rethinks=2,
        move_parser=mock_move_parser,
        legality_parser=mock_legality_parser,
        game_short_name="chess",
        prompt_generator=mock_prompt_generator,
        rethink_template="",
    )

    mock_generate_fn = mock.Mock()
    illegal_return1 = tournament_util.GenerateReturn(
        main_response="illegal1", main_response_and_thoughts="illegal1"
    )
    illegal_return2 = tournament_util.GenerateReturn(
        main_response="illegal2", main_response_and_thoughts="illegal2"
    )
    legal_return = tournament_util.GenerateReturn(
        main_response="e4", main_response_and_thoughts="e4"
    )
    mock_generate_fn.side_effect = [
        illegal_return1,
        illegal_return2,
        legal_return,
    ]
    mock_prompt_generator.generate_prompt_with_text_only.side_effect = [
        "prompt1",
        "prompt2",
        "prompt3",
    ]
    mock_move_parser.parse.side_effect = ["illegal1", "illegal2", "e4"]
    mock_legality_parser.parse.side_effect = _fake_legality_parser_parse

    output = sampler._sample_action(
        model_generate_fn=mock_generate_fn,
        state=mock_state,
        prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
    )

    self.assertEqual(output.action, "e4")
    self.assertEqual(
        output.generate_returns,
        [illegal_return1, illegal_return2, legal_return],
    )
    self.assertDictEqual(
        output.auxiliary_outputs,
        {
            "parsed_action_attempt_0": "illegal1",
            "maybe_legal_action_attempt_0": None,
            "rethink_prompt_attempt_0": "",
            "parsed_action_attempt_1": "illegal2",
            "maybe_legal_action_attempt_1": None,
            "rethink_prompt_attempt_1": (
                rethink._RETHINK_WITH_ENV_ILLEGAL_TEMPLATE.format(
                    last_move="illegal1"
                )
            ),
            "parsed_action_attempt_2": "e4",
            "maybe_legal_action_attempt_2": "e4",
            "rethink_prompt_attempt_2": (
                rethink._RETHINK_WITH_ENV_ILLEGAL_HISTORY_TEMPLATE.format(
                    illegal_history="illegal1, illegal2"
                )
            ),
        },
    )
    self.assertEqual(mock_generate_fn.call_count, 3)
    self.assertEqual(mock_move_parser.parse.call_count, 3)
    self.assertEqual(mock_legality_parser.parse.call_count, 3)

    calls = mock_prompt_generator.generate_prompt_with_text_only.call_args_list
    self.assertLen(calls, 3)
    self.assertEqual(
        calls[0],
        mock.call(
            prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
            game_short_name="chess",
            rethink_prompt="",
        ),
    )
    self.assertEqual(
        calls[1],
        mock.call(
            prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
            game_short_name="chess",
            rethink_prompt=rethink._RETHINK_WITH_ENV_ILLEGAL_TEMPLATE.format(
                last_move="illegal1"
            ),
        ),
    )
    self.assertEqual(
        calls[2],
        mock.call(
            prompt_template=prompts.PromptTemplate.NO_LEGAL_ACTIONS_RETHINK_APPENDED,
            game_short_name="chess",
            rethink_prompt=rethink._RETHINK_WITH_ENV_ILLEGAL_HISTORY_TEMPLATE.format(
                illegal_history="illegal1, illegal2"
            ),
        ),
    )


if __name__ == "__main__":
  absltest.main()
