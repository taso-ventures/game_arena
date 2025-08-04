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

"""Tests for LLM-based parsers."""

from absl.testing import absltest
from absl.testing import parameterized
from game_arena.harness import llm_parsers


class LlmParsersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'simple_case',
          'Clean Move: e4',
          'Clean Move: ',
          'e4',
      ),
      (
          'trailing_whitespace',
          'Clean Move: e4  ',
          'Clean Move: ',
          'e4',
      ),
      (
          'leading_whitespace',
          '  Clean Move: e4',
          'Clean Move: ',
          'e4',
      ),
      (
          'multiline',
          'Clean Move: e4\nSome other text',
          'Clean Move: ',
          'e4',
      ),
      (
          'special_characters',
          'Clean Move: a+b#c',
          'Clean Move: ',
          'a+b#c',
      ),
  )
  def test_parse_extractor_response(
      self, response, final_answer_prefix, expected
  ):
    self.assertEqual(
        llm_parsers._parse_extractor_response(
            response=response, final_answer_prefix=final_answer_prefix
        ),
        expected,
    )


if __name__ == '__main__':
  absltest.main()
