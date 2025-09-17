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

"""Timeout protection utilities for FreeCiv parsers."""

import functools
import signal
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar

from absl import logging

F = TypeVar('F', bound=Callable[..., Any])


class TimeoutError(Exception):
  """Raised when an operation times out."""
  pass


@contextmanager
def timeout_protection(timeout_seconds: float) -> Generator[None, None, None]:
  """Context manager for protecting operations with timeout.

  This uses threading.Timer for cross-platform compatibility.
  For Unix systems, signal.alarm() could be more efficient but
  doesn't work in threads.

  Args:
    timeout_seconds: Maximum time to allow for the operation

  Raises:
    TimeoutError: If operation exceeds timeout

  Example:
    with timeout_protection(5.0):
        result = expensive_regex_operation(text)
  """
  if timeout_seconds <= 0:
    # No timeout protection
    yield
    return

  timed_out = threading.Event()
  timer = None

  def timeout_handler() -> None:
    timed_out.set()

  try:
    timer = threading.Timer(timeout_seconds, timeout_handler)
    timer.start()

    start_time = time.time()
    yield

    if timed_out.is_set():
      elapsed = time.time() - start_time
      raise TimeoutError(
          f"Operation timed out after {elapsed:.2f} seconds "
          f"(limit: {timeout_seconds:.2f} seconds)"
      )

  finally:
    if timer:
      timer.cancel()


def with_timeout(timeout_seconds: float) -> Callable[[F], F]:
  """Decorator to add timeout protection to functions.

  Args:
    timeout_seconds: Maximum time to allow for function execution

  Returns:
    Decorated function with timeout protection

  Example:
    @with_timeout(5.0)
    def regex_search(pattern, text):
        return pattern.search(text)
  """
  def decorator(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      with timeout_protection(timeout_seconds):
        return func(*args, **kwargs)
    return wrapper
  return decorator


class TimeoutProtectedRegex:
  """Wrapper for regex operations with timeout protection."""

  def __init__(self, timeout_seconds: float = 5.0):
    """Initialize with timeout configuration.

    Args:
      timeout_seconds: Default timeout for regex operations
    """
    self.timeout_seconds = timeout_seconds

  def search(self, pattern, text: str, flags: int = 0) -> Any:
    """Protected regex search with timeout.

    Args:
      pattern: Compiled regex pattern or string pattern
      text: Text to search
      flags: Regex flags

    Returns:
      Match object or None

    Raises:
      TimeoutError: If search exceeds timeout
    """
    try:
      with timeout_protection(self.timeout_seconds):
        if hasattr(pattern, 'search'):
          # Already compiled pattern
          return pattern.search(text)
        else:
          # String pattern, compile and search
          import re
          compiled_pattern = re.compile(pattern, flags)
          return compiled_pattern.search(text)
    except TimeoutError as e:
      logging.warning("Regex search timed out: %s", e)
      raise

  def findall(self, pattern, text: str, flags: int = 0) -> list[str]:
    """Protected regex findall with timeout.

    Args:
      pattern: Compiled regex pattern or string pattern
      text: Text to search
      flags: Regex flags

    Returns:
      List of matches

    Raises:
      TimeoutError: If search exceeds timeout
    """
    try:
      with timeout_protection(self.timeout_seconds):
        if hasattr(pattern, 'findall'):
          # Already compiled pattern
          return pattern.findall(text)
        else:
          # String pattern, compile and search
          import re
          compiled_pattern = re.compile(pattern, flags)
          return compiled_pattern.findall(text)
    except TimeoutError as e:
      logging.warning("Regex findall timed out: %s", e)
      raise

  def sub(self, pattern, repl: str, text: str, count: int = 0, flags: int = 0) -> str:
    """Protected regex substitution with timeout.

    Args:
      pattern: Compiled regex pattern or string pattern
      repl: Replacement string
      text: Text to process
      count: Maximum number of substitutions
      flags: Regex flags

    Returns:
      Text with substitutions applied

    Raises:
      TimeoutError: If substitution exceeds timeout
    """
    try:
      with timeout_protection(self.timeout_seconds):
        if hasattr(pattern, 'sub'):
          # Already compiled pattern
          return pattern.sub(repl, text, count)
        else:
          # String pattern, compile and substitute
          import re
          compiled_pattern = re.compile(pattern, flags)
          return compiled_pattern.sub(repl, text, count)
    except TimeoutError as e:
      logging.warning("Regex substitution timed out: %s", e)
      raise