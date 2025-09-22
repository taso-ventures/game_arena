#!/usr/bin/env python3
"""Demonstration of improved token counting accuracy."""

from game_arena.harness.freeciv_memory import TokenManager


def demonstrate_token_counting():
  """Demonstrate the improved token counting features."""
  print("=== Token Counting Accuracy Demonstration ===\n")

  # Test with different models
  models = ["gpt-4", "claude-opus-4", "gemini-2.5", "deepseek"]

  sample_texts = [
    "Hello world!",
    "FreeCiv is a turn-based strategy game.",
    "Move unit_warrior(101) from position (10,14) to position (11,14).",
    "The FreeCiv LLM agent should analyze the current game state, consider strategic options, and select the most appropriate action based on the chosen strategy and available legal moves.",
  ]

  for model in models:
    print(f"Model: {model}")
    token_manager = TokenManager(model)
    info = token_manager.get_tokenizer_info()

    print(f"  Token limit: {info['token_limit']}")
    print(f"  Has tokenizer: {info['has_tokenizer']}")
    print(f"  Approximation ratio: {info['approximation_ratio']:.2f} chars/token")
    print()

    for i, text in enumerate(sample_texts, 1):
      tokens = token_manager.count_tokens(text)
      basic_estimate = len(text) // 4
      ratio = len(text) / tokens if tokens > 0 else 0

      print(f"  Text {i}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
      print(f"    Length: {len(text)} chars")
      print(f"    Tokens (improved): {tokens}")
      print(f"    Tokens (basic): {basic_estimate}")
      print(f"    Actual ratio: {ratio:.2f} chars/token")
      print()

    print("-" * 60)

  # Demonstrate truncation
  print("\n=== Truncation Demonstration ===\n")

  long_text = """
  In FreeCiv, strategic decision-making involves multiple considerations:
  1. Economic development through city growth and infrastructure
  2. Military expansion and territorial control
  3. Technological advancement through research
  4. Diplomatic relations with other civilizations
  5. Resource management and trade optimization

  The LLM agent must balance these competing priorities while adapting
  to the current game state, opponent actions, and long-term strategic goals.
  Each decision should maximize the civilization's chances of victory
  while minimizing risks and maintaining flexibility for future turns.
  """

  token_manager = TokenManager("gpt-4")
  original_tokens = token_manager.count_tokens(long_text)

  print(f"Original text: {len(long_text)} chars, {original_tokens} tokens")
  print()

  # Test different truncation limits
  for limit in [50, 25, 10]:
    # Temporarily set limit for demonstration
    original_limit = token_manager.limit
    token_manager.limit = limit + 20  # Add some buffer

    truncated = token_manager.truncate_to_limit(long_text, reserve=10)
    truncated_tokens = token_manager.count_tokens(truncated)

    print(f"Truncated to ~{limit} tokens:")
    print(f"  Result: {len(truncated)} chars, {truncated_tokens} tokens")
    print(f"  Preview: '{truncated[:100]}...'")
    print()

    # Restore limit
    token_manager.limit = original_limit

  print("=== Summary ===")
  print("✓ Model-specific approximation ratios")
  print("✓ Improved accuracy over basic character division")
  print("✓ Support for exact tokenizers when available")
  print("✓ Precise truncation with binary search")
  print("✓ Fallback graceful degradation")


if __name__ == "__main__":
  demonstrate_token_counting()