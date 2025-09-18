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

"""Library for generating model responses."""

import datetime
import traceback
from typing import Any, Generic, Mapping, Protocol, runtime_checkable

import tenacity
from absl import logging

from game_arena.harness import tournament_util


def _log_retry_warning(retry_state: tenacity.RetryCallState):
  assert retry_state.outcome is not None
  exception = retry_state.outcome.exception()
  traceback_str = "".join(traceback.format_exception(exception))
  logging.warning(
      "Attempting retry # %d. Traceback: %s. Retry state: %s",
      retry_state.attempt_number,
      traceback_str,
      retry_state,
  )


class DoNotRetryError(Exception):
  """An exception that should not be retried."""

  def __init__(self, *args, info: Any = None, **kwargs):
    super().__init__(*args, **kwargs)
    self.info = info

  def __str__(self):
    base_str = super().__str__()
    if self.info:
      return f"{base_str}\nInfo: {self.info}"
    return base_str


class UnsupportedCapabilityError(DoNotRetryError):
  """An exception for when a model does not support a capability."""


# TODO(google-deepmind): This catches all exceptions and retries them. It should be
# more selective about what it retries. Some error cases may be always fatal.
# With SDKs, the possible cases are sometimes not fully documented.
# Furthermore, SDKs may error without raising an exception. The error is only
# evident in the fields of the response.
# With HTTP APIs, we can rely on the status code to determine if it should
# be retried.
def _custom_retry_condition(retry_state: tenacity.RetryCallState) -> bool:
  """Return True if we should retry."""
  if retry_state.outcome is None:
    return False  # Should not happen, but to be safe.
  if not retry_state.outcome.failed:
    return False  # Do not retry on success.
  exception = retry_state.outcome.exception()
  return not isinstance(exception, DoNotRetryError)


_retry_decorator = tenacity.retry(
    retry=_custom_retry_condition,
    wait=tenacity.wait_random_exponential(min=1, max=60),
    stop=tenacity.stop_after_delay(datetime.timedelta(minutes=60)),
    before_sleep=_log_retry_warning,
    reraise=True,
)


@runtime_checkable
class Model(Generic[tournament_util.ModelTextInputT], Protocol):
  """Large language model with text input support."""

  def __init__(
      self,
      model_name: str,
      *,
      model_options: Mapping[str, Any] | None = None,
      api_options: Mapping[str, Any] | None = None,
  ):
    """Initializes the model.

    Args:
      model_name: Name of the model.
      model_options: Model options such as temperature, max output tokens,
        thinking budget/reasoning level.
      api_options: API options such as returning thinking text, streaming,
        timeouts.
    """
    self._model_name = model_name
    self._model_options = model_options
    self._api_options = api_options

  @property
  def model_name(self) -> str:
    return self._model_name

  def generate_with_text_input(
      self,
      model_input: tournament_util.ModelTextInputT,
  ) -> tournament_util.GenerateReturn:
    ...

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if hasattr(cls, "generate_with_text_input"):
      setattr(
          cls,
          "generate_with_text_input",
          _retry_decorator(cls.generate_with_text_input),
      )


@runtime_checkable
class MultimodalModel(
    Model, Generic[tournament_util.ModelImageTextInputT], Protocol
):
  """Large language model with text and image input support."""

  def generate_with_image_text_input(
      self,
      model_input: tournament_util.ModelImageTextInputT,
  ) -> tournament_util.GenerateReturn:
    ...

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if hasattr(cls, "generate_with_image_text_input"):
      setattr(
          cls,
          "generate_with_image_text_input",
          _retry_decorator(cls.generate_with_image_text_input),
      )
