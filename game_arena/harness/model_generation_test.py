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

"""Tests for model generation base implementations."""

import tenacity
from absl.testing import absltest

from game_arena.harness import model_generation, tournament_util


# This class will be decorated by the metaclass logic in model_generation.Model
# when it is defined, applying the default retry logic. The tests will then
# modify the retry behavior on an instance of this class.
class MockModelForRetryTest(model_generation.Model):
    """A mock model for testing the retry mechanism."""

    def __init__(self):
        super().__init__(model_name="mock_model")
        # We build mock-like functionality into the method itself because the
        # retry decorator wraps the real method, not a mock object.
        self._generate_call_count = 0
        self._side_effect_queue = []

    def set_side_effects(self, effects):
        """Sets a queue of side effects for the generate method to produce."""
        self._side_effect_queue = list(effects)

    @property
    def generate_call_count(self):
        """Returns the number of times the generate method was called."""
        return self._generate_call_count

    def generate_with_text_input(
        self, model_input: tournament_util.ModelTextInput
    ) -> tournament_util.GenerateReturn:
        """Generates a response, cycling through pre-configured side effects."""
        self._generate_call_count += 1
        if self._side_effect_queue:
            effect = self._side_effect_queue.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return effect
        # Default success response if no side effects are configured.
        return tournament_util.GenerateReturn(
            main_response="success", main_response_and_thoughts="success"
        )


class RetryTest(absltest.TestCase):

    def test_retry_on_failure_succeeds(self):
        """Tests that the model call is retried on failure and eventually succeeds."""
        model = MockModelForRetryTest()

        # Instead of patching the decorator, we modify the retry behavior of the
        # already-decorated method on the instance. This is a cleaner approach.
        model.generate_with_text_input.retry.stop = tenacity.stop_after_attempt(3)
        model.generate_with_text_input.retry.wait = tenacity.wait_none()

        model.set_side_effects(
            [
                Exception("transient error"),
                Exception("another transient error"),
                tournament_util.GenerateReturn(
                    main_response="final success",
                    main_response_and_thoughts="final success",
                ),
            ]
        )

        response = model.generate_with_text_input(
            tournament_util.ModelTextInput(prompt_text="test")
        )

        self.assertEqual(response.main_response, "final success")
        self.assertEqual(model.generate_call_count, 3)

    def test_retry_gives_up_on_persistent_error(self):
        """Tests that the retry mechanism gives up after a set number of attempts."""
        model = MockModelForRetryTest()

        model.generate_with_text_input.retry.stop = tenacity.stop_after_attempt(3)
        model.generate_with_text_input.retry.wait = tenacity.wait_none()

        model.set_side_effects(
            [
                Exception("persistent error"),
                Exception("persistent error"),
                Exception("persistent error"),
            ]
        )

        with self.assertRaisesRegex(Exception, "persistent error"):
            model.generate_with_text_input(
                tournament_util.ModelTextInput(prompt_text="test")
            )

        self.assertEqual(model.generate_call_count, 3)


if __name__ == "__main__":
    absltest.main()
