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

"""Go-specific prompt builder for LLM agents.

This module provides Go-specific prompt generation with strategic analysis,
territory evaluation, and entertaining strategic guidance focused on
the beauty and depth of Go gameplay.
"""

from typing import Any, Dict, List
from game_arena.harness.prompts.base import BasePromptBuilder


class GoPromptBuilder(BasePromptBuilder):
    """Go prompt builder - uses original prompt system.

    Go continues to use the standard prompt templates from the original
    system, not the new FreeCiv-specific enhancement system.
    """

    def __init__(self):
        """Initialize the Go prompt builder."""
        super().__init__("go")

    def build_enhanced_prompt(
        self,
        observation: Dict[str, Any],
        legal_actions: List[Any],  # Go moves
        model_name: str,
        **kwargs
    ) -> str:
        """Generate go prompt using original template system.

        Go uses the same prompts as before - this new builder system
        was only intended for FreeCiv enhancements.

        Raises:
            NotImplementedError: Go should use original PromptGeneratorText
        """
        raise NotImplementedError(
            "Go should continue using the original PromptGeneratorText class "
            "from prompt_generation.py, not this new system. This new system "
            "was only intended for FreeCiv enhancements."
        )