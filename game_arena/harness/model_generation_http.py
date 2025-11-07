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

"""Model generation implementations using HTTP POST APIs."""

import asyncio
import base64
import dataclasses
import datetime
import json
import multiprocessing
import os
import re
import threading
import time
from typing import Any, Mapping, Sequence

import aiohttp
import requests
from absl import logging

from game_arena.harness import model_generation, tournament_util

DEEPSEEK_THOUGHT_TAG_START = "<think>"


def _sanitize_url_for_logging(url: str) -> str:
    """Sanitize URL by removing potential API keys or passwords for safe logging.

    Args:
        url: The URL to sanitize

    Returns:
        Sanitized URL with sensitive parameters redacted
    """
    # Remove common API key parameters
    sanitized = re.sub(
        r"([?&])(api_key|key|token|password|secret)=[^&]*",
        r"\1\2=***",
        url,
        flags=re.IGNORECASE,
    )

    # Remove API keys in path segments (e.g., /api/v1/API_KEY_HERE/...)
    sanitized = re.sub(r"/[A-Za-z0-9_-]{20,}/", "/***/", sanitized)

    return sanitized


def _sanitize_request_for_logging(request: Mapping[str, Any]) -> dict[str, Any]:
    """Sanitize request object by removing sensitive information for safe logging.

    Args:
        request: The request object to sanitize

    Returns:
        Sanitized request with sensitive data redacted
    """
    sanitized = dict(request)
    # Remove or redact sensitive headers if present
    if "headers" in sanitized:
        headers = dict(sanitized["headers"])
        for header_name in headers:
            if header_name.lower() in ["authorization", "x-api-key", "api-key"]:
                headers[header_name] = "***"
        sanitized["headers"] = headers
    return sanitized


DEEPSEEK_THOUGHT_TAG_END = "</think>"


def _deepseek_separate_main_response_and_thoughts(
    response: str,
) -> tuple[str, str] | None:
    """Separates the main response and thoughts from a Deepseek response.

    Args:
      response: The response from the Deepseek model.

    Returns:
      A tuple of the main response and thoughts, or None if the tags are not
      present.
    """
    if DEEPSEEK_THOUGHT_TAG_START not in response:
        return None
    else:
        thoughts_and_response = response.split(DEEPSEEK_THOUGHT_TAG_START)[1]
        if DEEPSEEK_THOUGHT_TAG_END in thoughts_and_response:
            thoughts = thoughts_and_response.split(DEEPSEEK_THOUGHT_TAG_END)[0]
            main_response = thoughts_and_response.split(DEEPSEEK_THOUGHT_TAG_END)[1]
            return main_response, thoughts


def _create_image_text_content(
    model_input: tournament_util.ModelImageTextInput,
):
    """Creates content payload with text followed by image."""
    content = [{"type": "text", "text": model_input.prompt_text}]
    base64_image = base64.b64encode(model_input.prompt_image_bytes).decode("utf-8")
    content.append(
        {
            "type": "image_url",
            "image_url": {
                "url": (
                    f"data:{model_input.prompt_image_mime_type};base64,{base64_image}"
                )
            },
        }
    )
    return content


class TogetherAIModel(model_generation.MultimodalModel):
    """Wrapper for access to models served by Together.AI."""

    def __init__(
        self,
        model_name: str,
        *,
        model_options: Mapping[str, Any] | None = None,
        api_options: Mapping[str, Any] | None = None,
        api_key: str | None = None,
    ):
        super().__init__(
            model_name, model_options=model_options, api_options=api_options
        )
        # If API key is None, defaults to TOGETHER_API_KEY in environment.
        if api_key is None:
            try:
                api_key = os.environ["TOGETHER_API_KEY"]
            except KeyError as e:
                logging.error(
                    "TOGETHER_API_KEY environment variable not set. Please set it to"
                    " use %s.",
                    self._model_name,
                )
                raise e
        self._headers = {"Authorization": f"Bearer {api_key}"}

    # TODO(google-deepmind): Add error handling.
    def _generate(
        self,
        content: Sequence[Mapping[str, Any]],
        system_instruction: str | None = None,
    ) -> tournament_util.GenerateReturn:
        messages = []
        if system_instruction is not None:
            messages.append({"role": "system", "content": system_instruction})
        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

        if self._model_options is None:
            self._model_options = {}
        if self._api_options is None:
            self._api_options = {}

        request_options = {}

        if "max_tokens" in self._model_options:
            max_sampling_length = self._model_options["max_tokens"]
        else:
            match self._model_name:
                case "deepseek-ai/DeepSeek-R1-0528-tput":
                    max_context_length = 163839
                case "deepseek-ai/DeepSeek-R1":
                    max_context_length = 163839
                case "deepseek-ai/DeepSeek-V3":
                    max_context_length = 163839
                case "moonshotai/Kimi-K2-Instruct":
                    max_context_length = 128000
                case "Qwen/Qwen3-235B-A22B-Thinking-2507":
                    max_context_length = 262144
                case "Qwen/Qwen3-235B-A22B-Instruct-2507-tput":
                    max_context_length = 262144
                case "Qwen/Qwen3-235B-A22B-fp8-tput":
                    max_context_length = 40960
                case _:
                    logging.warning(
                        "max_context_length unknown for model %s. Defaulting to 32768.",
                        self._model_name,
                    )
                    max_context_length = 32768
            max_sampling_length = max_context_length - self._model_options.get(
                "max_prompt_length", 16384
            )

        request_options["max_tokens"] = max_sampling_length

        for supported_model_option in [
            "temperature",
            "top_k",
            "top_p",
        ]:
            if supported_model_option in self._model_options:
                request_options[supported_model_option] = self._model_options[
                    supported_model_option
                ]
        request = {
            "model": self._model_name,
            "messages": messages,
        } | request_options

        parallel_attempts = self._api_options.get("parallel_attempts", 1)
        if parallel_attempts > 1:
            payload_details = {
                "url": "https://api.together.xyz/v1/chat/completions",
                "payload": request,
                "headers": self._headers,
                "timeout": datetime.timedelta(
                    seconds=self._api_options.get("timeout", 3600)
                ),
            }
            name_to_payload_details = {
                f"{self._model_name}_attempt_{i}": payload_details
                for i in range(parallel_attempts)
            }
            maybe_successful_result, error_infos = asyncio.run(
                _post_requests_async_return_first_success(name_to_payload_details)
            )
            if maybe_successful_result is None:
                raise RuntimeError(
                    f"All {parallel_attempts} attempts failed: {error_infos}"
                )
            completion = maybe_successful_result.json_response
        else:
            # Splitting the request into a thread to be able to inspect the status of
            # the request and print the progress.
            q = multiprocessing.Queue()

            def do_request():
                try:
                    assert self._api_options is not None
                    response = requests.post(
                        "https://api.together.xyz/v1/chat/completions",
                        json=request,
                        headers=self._headers,
                        timeout=self._api_options.get("timeout", 3600),
                    )
                    response.raise_for_status()
                    q.put(
                        {"status_code": response.status_code, "json": response.json()}
                    )
                except requests.exceptions.RequestException as e:
                    logging.exception("Request to Together.AI failed: %s", e)
                    q.put(e)

            thread = threading.Thread(target=do_request)
            thread.start()

            last_wait_message_time = time.monotonic()
            while thread.is_alive():
                thread.join(timeout=1.0)
                current_time = time.monotonic()
                if current_time - last_wait_message_time >= 120:  # 2 minutes
                    print(f"Still waiting on {self._model_name}")
                    last_wait_message_time = current_time

            result = q.get()
            if isinstance(result, Exception):
                raise result

            assert result["status_code"] == 200

            completion = result["json"]

        content = completion["choices"][0]["message"]["content"]
        if content is None:
            logging.warning(
                "Together.AI Completion return content is None. Returning empty"
                " string. Request: %s",
                request,
            )
            content = ""

        main_response = content
        main_response_and_thoughts = ""

        if "deepseek" in self._model_name.lower():
            maybe_separated = _deepseek_separate_main_response_and_thoughts(content)
            if maybe_separated is not None:
                main_response = maybe_separated[0]
                main_response_and_thoughts = content

        generation_tokens = None
        prompt_tokens = None
        if "usage" in completion:
            usage = completion["usage"]
            generation_tokens = usage["completion_tokens"]
            prompt_tokens = usage["prompt_tokens"]

        return tournament_util.GenerateReturn(
            main_response=main_response,
            main_response_and_thoughts=main_response_and_thoughts,
            request_for_logging=_sanitize_request_for_logging(request),
            response_for_logging=completion,
            generation_tokens=generation_tokens,
            prompt_tokens=prompt_tokens,
        )

    def generate_with_text_input(
        self, model_input: tournament_util.ModelTextInput
    ) -> tournament_util.GenerateReturn:
        content = [{"type": "text", "text": model_input.prompt_text}]
        return self._generate(content, model_input.system_instruction)

    def generate_with_image_text_input(
        self, model_input: tournament_util.ModelImageTextInput
    ) -> tournament_util.GenerateReturn:
        content = _create_image_text_content(model_input)
        return self._generate(content, model_input.system_instruction)


class XAIModel(model_generation.MultimodalModel):
    """Wrapper to call the xAI API. Supports both streaming and non-streaming."""

    def __init__(
        self,
        model_name: str,
        *,
        model_options: Mapping[str, Any] | None = None,
        api_options: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        debug: bool = False,
    ):
        super().__init__(
            model_name, model_options=model_options, api_options=api_options
        )
        if self._model_options is None:
            self._model_options = {}
        if self._api_options is None:
            self._api_options = {}

        if api_key is None:
            api_key = os.environ["XAI_API_KEY"]

        if api_key is None:
            raise ValueError("XAI API key not found.")

        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._debug = debug
        self._stream = self._api_options.get("stream", True)

    # TODO(google-deepmind): Add error handling.
    def _post_request(self, request: Mapping[str, Any], stream: bool):
        """Sends a POST request to the xAI API and handles errors."""
        timeout = self._api_options.get("timeout", 20 * 60)
        try:
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                json=request,
                headers=self._headers,
                stream=stream,
                timeout=timeout,
            )
            response.raise_for_status()

            return response
        except requests.exceptions.HTTPError as e:
            if e.response is not None:
                print("Error during XAI request: ", flush=True)
                print(f"Status code: {e.response.status_code}", flush=True)
                print(f"Reason: {e.response.reason}", flush=True)
                print(f"Headers: {e.response.headers}", flush=True)
                print(f"Response text: {e.response.text}", flush=True)
                print(request)
                if e.response.status_code == 400:
                    raise model_generation.DoNotRetryError(
                        str(e),
                        info={"request": request, "response": e.response},
                    ) from e
            raise

    def _generate(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> tournament_util.GenerateReturn:
        request = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
            "stream_options": {"include_usage": True},
        }
        for option in ["temperature", "top_p", "top_k"]:
            if option in self._model_options:
                request[option] = self._model_options[option]

        response = self._post_request(request, stream=False)
        completion = response.json()
        full_content = completion["choices"][0]["message"]["content"]

        if full_content is None:
            logging.warning(
                "xAI Completion return content is None. Returning empty string."
                " Request: %s",
                request,
            )
            full_content = ""

        full_reasoning_content = ""
        if "reasoning_content" in completion["choices"][0]["message"]:
            full_reasoning_content = completion["choices"][0]["message"][
                "reasoning_content"
            ]
        total_generation_tokens = None
        total_prompt_tokens = None
        total_reasoning_tokens = None
        if "usage" in completion:
            usage = completion["usage"]
            if "completion_tokens" in usage:
                total_generation_tokens = usage["completion_tokens"]
            if "prompt_tokens" in usage:
                total_prompt_tokens = usage["prompt_tokens"]
            if (
                "completion_tokens_details" in usage
                and "reasoning_tokens" in usage["completion_tokens_details"]
            ):
                total_reasoning_tokens = usage["completion_tokens_details"][
                    "reasoning_tokens"
                ]
        main_response_and_thoughts = (
            full_content + ("\n\n" + full_reasoning_content)
            if full_reasoning_content
            else ""
        )
        return tournament_util.GenerateReturn(
            main_response=full_content,
            main_response_and_thoughts=main_response_and_thoughts,
            request_for_logging=_sanitize_request_for_logging(request),
            response_for_logging=completion,
            generation_tokens=total_generation_tokens,
            prompt_tokens=total_prompt_tokens,
            reasoning_tokens=total_reasoning_tokens,
        )

    def _print_debug(self, message: str):
        if self._debug:
            print(message)

    def _log_chunk_received(
        self,
        start_time: float,
        elapsed_time: float,
        full_content: str,
        delta_content: str,
        is_thinking_chunk: bool,
        last_full_content_print_time: float,
        current_time: float,
    ):
        """Logs information about received chunks."""
        time_since_last_full_print = current_time - last_full_content_print_time
        if (
            last_full_content_print_time == start_time
            or time_since_last_full_print >= 120
        ):
            if is_thinking_chunk:
                self._print_debug(
                    f"[{self._model_name}] Received thinking chunk, elapsed:"
                    f" {elapsed_time:.2f}s, content_len:"
                    f" {len(full_content)}, full_content: {full_content}"
                )
            else:
                self._print_debug(
                    f"[{self._model_name}] Received chunk, elapsed:"
                    f" {elapsed_time:.2f}s, content_len:"
                    f" {len(full_content)}, full_content: {full_content}"
                )
            return True
        else:
            if is_thinking_chunk:
                self._print_debug(
                    f"[{self._model_name}] Received thinking chunk, elapsed:"
                    f" {elapsed_time:.2f}s, chunk: {delta_content}"
                )
            else:
                self._print_debug(
                    f"[{self._model_name}] Received chunk, elapsed:"
                    f" {elapsed_time:.2f}s, chunk: {delta_content}"
                )
            return False

    def _generate_streaming(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> tournament_util.GenerateReturn:
        request = {
            "model": self._model_name,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        full_content = ""
        full_reasoning_content = ""
        total_generation_tokens = 0
        total_prompt_tokens = 0
        total_reasoning_tokens = 0
        response_for_logging = []
        start_time = time.monotonic()
        last_full_content_print_time = start_time
        self._print_debug(f"[{self._model_name}] Starting streaming completion.")

        response = self._post_request(request, stream=True)
        try:
            for chunk in response.iter_lines():
                if chunk:
                    current_time = time.monotonic()
                    elapsed_time = current_time - start_time
                    if chunk.startswith(b"data: "):
                        chunk_data = chunk[len(b"data: ") :].decode("utf-8")
                        if chunk_data == "[DONE]":
                            self._print_debug(
                                f"[{self._model_name}] Streaming finished, elapsed:"
                                f" {elapsed_time:.2f}s"
                            )
                            break
                        try:
                            json_data = json.loads(chunk_data)
                            if "choices" in json_data and json_data["choices"]:
                                delta = json_data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    delta_content = delta["content"]
                                    full_content += delta_content
                                    is_full_content_print = self._log_chunk_received(
                                        start_time=start_time,
                                        elapsed_time=elapsed_time,
                                        full_content=full_content,
                                        delta_content=delta_content,
                                        is_thinking_chunk=False,
                                        last_full_content_print_time=last_full_content_print_time,
                                        current_time=current_time,
                                    )
                                    if is_full_content_print:
                                        # Log a chunk only every 2 minutes.
                                        response_for_logging.append(
                                            {
                                                "chunk": chunk_data,
                                                "elapsed_time": elapsed_time,
                                                "full_content": full_content,
                                            }
                                        )
                                        last_full_content_print_time = current_time
                                if "reasoning_content" in delta:
                                    delta_reasoning_content = delta["reasoning_content"]
                                    full_reasoning_content += delta_reasoning_content
                                    is_full_content_print = self._log_chunk_received(
                                        start_time=start_time,
                                        elapsed_time=elapsed_time,
                                        full_content=full_reasoning_content,
                                        delta_content=delta_reasoning_content,
                                        is_thinking_chunk=True,
                                        last_full_content_print_time=last_full_content_print_time,
                                        current_time=current_time,
                                    )
                                    if is_full_content_print:
                                        response_for_logging.append(
                                            {
                                                "chunk": chunk_data,
                                                "elapsed_time": elapsed_time,
                                                "full_content": full_reasoning_content,
                                            }
                                        )
                                        last_full_content_print_time = current_time

                            if "usage" in json_data:
                                usage = json_data["usage"]
                                response_for_logging.append({"usage_chunk": json_data})
                                if "completion_tokens" in usage:
                                    total_generation_tokens += usage[
                                        "completion_tokens"
                                    ]
                                if "prompt_tokens" in usage:
                                    total_prompt_tokens += usage["prompt_tokens"]
                                if (
                                    "completion_tokens_details" in usage
                                    and "reasoning_tokens"
                                    in usage["completion_tokens_details"]
                                ):
                                    total_reasoning_tokens += usage[
                                        "completion_tokens_details"
                                    ]["reasoning_tokens"]

                        except json.JSONDecodeError as e:
                            logging.warning(
                                "[%s] JSON decode error for chunk: %s, elapsed: %.2fs",
                                self._model_name,
                                chunk_data,
                                elapsed_time,
                            )
                            raise e
        finally:
            response.close()

        main_response_and_thoughts = (
            full_content + ("\n\n" + full_reasoning_content)
            if full_reasoning_content
            else ""
        )
        return tournament_util.GenerateReturn(
            main_response=full_content,
            main_response_and_thoughts=main_response_and_thoughts,
            request_for_logging=_sanitize_request_for_logging(request),
            response_for_logging={"filtered_chunks_list": response_for_logging},
            generation_tokens=total_generation_tokens,
            prompt_tokens=total_prompt_tokens,
            reasoning_tokens=total_reasoning_tokens,
        )

    def generate_with_text_input(
        self, model_input: tournament_util.ModelTextInput
    ) -> tournament_util.GenerateReturn:
        messages = []
        if model_input.system_instruction is not None:
            messages.append(
                {"role": "system", "content": model_input.system_instruction}
            )

        messages.append(
            {
                "role": "user",
                "content": model_input.prompt_text,
            }
        )

        if self._stream:
            return self._generate_streaming(messages)
        else:
            return self._generate(messages)

    def generate_with_image_text_input(
        self, model_input: tournament_util.ModelImageTextInput
    ) -> tournament_util.GenerateReturn:
        if "grok-3" in self._model_name:
            raise model_generation.UnsupportedCapabilityError(
                f"Model {self._model_name} does not support image input."
            )
        messages = []
        if model_input.system_instruction is not None:
            messages.append(
                {"role": "system", "content": model_input.system_instruction}
            )
        content = _create_image_text_content(model_input)
        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )
        if self._stream:
            return self._generate_streaming(messages)
        else:
            return self._generate(messages)


@dataclasses.dataclass(frozen=True, kw_only=True)
class PostRequestResult:
    name: str
    status: int
    json_response: Mapping[str, Any]


async def _post_request_async(
    *,
    session: aiohttp.ClientSession,
    name: str,
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
    timeout: datetime.timedelta,
) -> PostRequestResult:
    """Posts a request asynchronously and returns the result."""
    sanitized_url = _sanitize_url_for_logging(url)
    logging.debug("Starting POST for '%s' to %s", name, sanitized_url)
    try:
        async with session.post(
            url, json=payload, headers=headers, timeout=timeout.total_seconds()
        ) as response:
            logging.info(
                "Finished POST for '%s' with status: %d", name, response.status
            )
            # Raise an exception for bad status codes (4xx or 5xx)
            response.raise_for_status()

            json_response = await response.json()
            return PostRequestResult(
                name=name, status=response.status, json_response=json_response
            )

    except asyncio.CancelledError:
        logging.warning("POST for '%s' was cancelled.", name)
        raise
    except Exception as e:
        logging.error("POST for '%s' failed: %s", name, type(e).__name__)
        # Re-raise the exception so the main loop knows it failed
        raise


async def _post_requests_async_return_first_success(
    name_to_payload_details: Mapping[str, Mapping[str, Any]],
) -> tuple[PostRequestResult | None, Sequence[Mapping[str, Any]]]:
    """Posts requests asynchronously and returns the first successful result."""
    error_infos = []
    successful_result = None

    async def _wrapper(
        *,
        session: aiohttp.ClientSession,
        name: str,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
        timeout: datetime.timedelta,
    ) -> tuple[str, PostRequestResult | None, Exception | None]:
        """Wraps the async post request to return name and exception."""
        try:
            result = await _post_request_async(
                session=session,
                name=name,
                url=url,
                payload=payload,
                headers=headers,
                timeout=timeout,
            )
            return name, result, None
        except Exception as e:  # pylint: disable=broad-exception-caught
            return name, None, e

    async with aiohttp.ClientSession() as session:
        if not name_to_payload_details:
            return None, []

        tasks = [
            asyncio.create_task(_wrapper(session=session, name=name, **payload_details))
            for name, payload_details in name_to_payload_details.items()
        ]

        for future in asyncio.as_completed(tasks):
            name, result, exc = await future
            if exc is None:
                successful_result = result
                print(f"\n--- First SUCCESSFUL POST received from '{name}' ---")
                break  # We have a winner
            else:
                print(f"Ignoring failed task '{name}'. Waiting for the next one...")
                error_infos.append(
                    {
                        "request_name": name,
                        "error": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        else:
            print("\n--- All POST requests failed. No successful result found. ---\n")

        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        return successful_result, error_infos


class OllamaModel(model_generation.MultimodalModel):
    """Wrapper for local Ollama API access.
    
    Ollama provides local LLM inference with models like Llama, Mistral, etc.
    By default connects to http://localhost:11434, but can be configured via
    OLLAMA_BASE_URL environment variable or base_url parameter.
    
    Example usage:
        model = OllamaModel(model_name="llama3.2")
        model = OllamaModel(model_name="llama3.2:70b", base_url="http://192.168.1.100:11434")
    """

    def __init__(
        self,
        model_name: str,
        *,
        model_options: Mapping[str, Any] | None = None,
        api_options: Mapping[str, Any] | None = None,
        base_url: str | None = None,
    ):
        super().__init__(
            model_name, model_options=model_options, api_options=api_options
        )
        # If base_url is None, use OLLAMA_BASE_URL env var or default to localhost
        if base_url is None:
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self._base_url = base_url.rstrip("/")
        self._api_url = f"{self._base_url}/api/generate"
        self._chat_url = f"{self._base_url}/api/chat"

    def _generate(
        self,
        content: Sequence[Mapping[str, Any]],
        system_instruction: str | None = None,
    ) -> tournament_util.GenerateReturn:
        """Generate using Ollama chat endpoint."""
        messages = []
        if system_instruction is not None:
            messages.append({"role": "system", "content": system_instruction})
        
        # Convert content to messages format
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    messages.append({"role": "user", "content": item.get("text", "")})
                elif item.get("type") == "image_url":
                    # Ollama supports images in messages
                    messages.append({
                        "role": "user",
                        "content": item.get("text", ""),
                        "images": [item["image_url"]["url"]]
                    })
            elif isinstance(item, str):
                messages.append({"role": "user", "content": item})

        if self._model_options is None:
            self._model_options = {}
        if self._api_options is None:
            self._api_options = {}

        # Build Ollama request
        request = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
        }

        # Add optional parameters
        options = {}
        if "temperature" in self._model_options:
            options["temperature"] = self._model_options["temperature"]
        if "top_k" in self._model_options:
            options["top_k"] = self._model_options["top_k"]
        if "top_p" in self._model_options:
            options["top_p"] = self._model_options["top_p"]
        if "max_tokens" in self._model_options:
            options["num_predict"] = self._model_options["max_tokens"]
        
        if options:
            request["options"] = options

        try:
            timeout = self._api_options.get("timeout", 300)
            response = requests.post(
                self._chat_url,
                json=request,
                timeout=timeout,
            )
            response.raise_for_status()
            completion = response.json()

            # Extract content from response
            response_content = completion.get("message", {}).get("content", "")
            if response_content is None:
                logging.warning(
                    "Ollama completion returned None content. Returning empty string."
                    " Request: %s",
                    request,
                )
                response_content = ""

            # Extract token counts if available
            generation_tokens = None
            prompt_tokens = None
            if "eval_count" in completion:
                generation_tokens = completion["eval_count"]
            if "prompt_eval_count" in completion:
                prompt_tokens = completion["prompt_eval_count"]

            return tournament_util.GenerateReturn(
                main_response=response_content,
                main_response_and_thoughts="",
                request_for_logging=request,
                response_for_logging=completion,
                generation_tokens=generation_tokens,
                prompt_tokens=prompt_tokens,
            )

        except requests.exceptions.RequestException as e:
            logging.exception("Request to Ollama failed: %s", e)
            raise

    def generate_with_text_input(
        self, model_input: tournament_util.ModelTextInput
    ) -> tournament_util.GenerateReturn:
        content = [{"type": "text", "text": model_input.prompt_text}]
        return self._generate(content, model_input.system_instruction)

    def generate_with_image_text_input(
        self, model_input: tournament_util.ModelImageTextInput
    ) -> tournament_util.GenerateReturn:
        content = _create_image_text_content(model_input)
        return self._generate(content, model_input.system_instruction)


class HuggingFaceModel(model_generation.MultimodalModel):
    """Wrapper for HuggingFace Inference API.
    
    Supports both HuggingFace Inference API (serverless) and Inference Endpoints.
    Requires HF_TOKEN environment variable or api_key parameter.
    
    For local inference with transformers library, consider using Ollama or vLLM instead.
    
    Example usage:
        model = HuggingFaceModel(model_name="meta-llama/Llama-3.2-3B-Instruct")
        model = HuggingFaceModel(
            model_name="meta-llama/Llama-3.2-70B-Instruct",
            api_key="hf_...",
            use_inference_endpoint=True,
            endpoint_url="https://your-endpoint.endpoints.huggingface.cloud"
        )
    """

    def __init__(
        self,
        model_name: str,
        *,
        model_options: Mapping[str, Any] | None = None,
        api_options: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        use_inference_endpoint: bool = False,
        endpoint_url: str | None = None,
    ):
        super().__init__(
            model_name, model_options=model_options, api_options=api_options
        )
        
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        if api_key is None:
            raise ValueError(
                "HuggingFace API key not found. Set HF_TOKEN environment variable "
                "or pass api_key parameter."
            )
        
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._use_inference_endpoint = use_inference_endpoint
        
        # Determine API URL
        if use_inference_endpoint and endpoint_url:
            self._api_url = endpoint_url.rstrip("/")
        elif use_inference_endpoint:
            raise ValueError(
                "endpoint_url must be provided when use_inference_endpoint=True"
            )
        else:
            # Use serverless Inference API
            self._api_url = (
                f"https://api-inference.huggingface.co/models/{model_name}"
            )

    def _generate(
        self,
        content: Sequence[Mapping[str, Any]],
        system_instruction: str | None = None,
    ) -> tournament_util.GenerateReturn:
        """Generate using HuggingFace Inference API."""
        messages = []
        if system_instruction is not None:
            messages.append({"role": "system", "content": system_instruction})
        
        # Convert content to text
        user_content = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                user_content += item.get("text", "")
            elif isinstance(item, str):
                user_content += item
        
        messages.append({"role": "user", "content": user_content})

        if self._model_options is None:
            self._model_options = {}
        if self._api_options is None:
            self._api_options = {}

        # Build request parameters
        parameters = {}
        if "max_tokens" in self._model_options:
            parameters["max_new_tokens"] = self._model_options["max_tokens"]
        if "temperature" in self._model_options:
            parameters["temperature"] = self._model_options["temperature"]
        if "top_k" in self._model_options:
            parameters["top_k"] = self._model_options["top_k"]
        if "top_p" in self._model_options:
            parameters["top_p"] = self._model_options["top_p"]
        
        parameters["return_full_text"] = False

        request = {
            "inputs": user_content,
            "parameters": parameters,
        }

        try:
            timeout = self._api_options.get("timeout", 300)
            response = requests.post(
                self._api_url,
                json=request,
                headers=self._headers,
                timeout=timeout,
            )
            response.raise_for_status()
            completion = response.json()

            # Handle different response formats
            response_text = ""
            if isinstance(completion, list) and len(completion) > 0:
                result = completion[0]
                response_text = result.get("generated_text", "")
            elif isinstance(completion, dict):
                response_text = completion.get("generated_text", "")

            if not response_text:
                logging.warning(
                    "HuggingFace completion returned empty content. Request: %s",
                    request,
                )
                response_text = ""

            # Normalize completion to dict for logging
            completion_dict = completion if isinstance(completion, dict) else {"results": completion}

            return tournament_util.GenerateReturn(
                main_response=response_text,
                main_response_and_thoughts="",
                request_for_logging=_sanitize_request_for_logging(request),
                response_for_logging=completion_dict,
                generation_tokens=None,  # HF API doesn't always return token counts
                prompt_tokens=None,
            )

        except requests.exceptions.RequestException as e:
            logging.exception("Request to HuggingFace failed: %s", e)
            raise

    def generate_with_text_input(
        self, model_input: tournament_util.ModelTextInput
    ) -> tournament_util.GenerateReturn:
        content = [{"type": "text", "text": model_input.prompt_text}]
        return self._generate(content, model_input.system_instruction)

    def generate_with_image_text_input(
        self, model_input: tournament_util.ModelImageTextInput
    ) -> tournament_util.GenerateReturn:
        # HuggingFace Inference API has limited multimodal support
        # Fall back to text-only for now
        logging.warning(
            "HuggingFace model %s may not support image input via Inference API. "
            "Using text-only.",
            self._model_name,
        )
        content = [{"type": "text", "text": model_input.prompt_text}]
        return self._generate(content, model_input.system_instruction)
