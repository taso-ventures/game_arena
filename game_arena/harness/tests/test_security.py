"""Security tests for FreeCiv proxy client.

This module contains tests that verify the security fixes for the FreeCiv
proxy client, including protection against JSON injection, oversized messages,
cache poisoning, and other attack vectors.
"""

import json
import unittest

from game_arena.harness.freeciv_proxy_client import (
  create_secure_cache_key,
  safe_json_loads,
  MAX_JSON_SIZE,
  MAX_JSON_DEPTH,
)


class TestJSONSecurity(unittest.TestCase):
  """Test JSON parsing security."""

  def test_safe_json_loads_valid_input(self):
    """Test that valid JSON is parsed correctly."""
    valid_json = '{"type": "test", "data": {"value": 42}}'
    result = safe_json_loads(valid_json)
    self.assertEqual(result["type"], "test")
    self.assertEqual(result["data"]["value"], 42)

  def test_safe_json_loads_oversized_input(self):
    """Test that oversized JSON is rejected."""
    # Create JSON larger than MAX_JSON_SIZE
    large_data = "x" * (MAX_JSON_SIZE + 1000)
    large_json = json.dumps({"data": large_data})

    with self.assertRaises(ValueError) as cm:
      safe_json_loads(large_json)
    self.assertIn("exceeds maximum", str(cm.exception))

  def test_safe_json_loads_deeply_nested_input(self):
    """Test that deeply nested JSON is rejected."""
    # Create JSON with depth > MAX_JSON_DEPTH
    nested_dict = {}
    current = nested_dict
    for i in range(MAX_JSON_DEPTH + 10):
      current["nested"] = {}
      current = current["nested"]

    deeply_nested_json = json.dumps(nested_dict)

    with self.assertRaises(ValueError) as cm:
      safe_json_loads(deeply_nested_json)
    self.assertIn("depth exceeds maximum", str(cm.exception))

  def test_safe_json_loads_invalid_json_syntax(self):
    """Test that invalid JSON syntax is rejected."""
    invalid_json = '{"type": "test", "data": invalid}'

    with self.assertRaises(json.JSONDecodeError):
      safe_json_loads(invalid_json)

  def test_safe_json_loads_non_object_root(self):
    """Test that non-object root is rejected."""
    array_json = '["array", "not", "object"]'

    with self.assertRaises(ValueError) as cm:
      safe_json_loads(array_json)
    self.assertIn("must be an object at root level", str(cm.exception))

  def test_safe_json_loads_non_string_input(self):
    """Test that non-string input is rejected."""
    with self.assertRaises(ValueError) as cm:
      safe_json_loads({"not": "string"})
    self.assertIn("Input must be a string", str(cm.exception))


class TestCacheKeySecurity(unittest.TestCase):
  """Test cache key security."""

  def test_create_secure_cache_key_valid_input(self):
    """Test that valid inputs create secure cache keys."""
    key = create_secure_cache_key("state", "llm_optimized")
    self.assertIsInstance(key, str)
    self.assertIn("state", key)
    self.assertIn("llm_optimized", key)
    # Should include hash for security
    self.assertTrue(len(key.split("_")) >= 3)

  def test_create_secure_cache_key_sanitization(self):
    """Test that invalid characters are sanitized."""
    key = create_secure_cache_key("state", "format@with#invalid!chars")
    self.assertIsInstance(key, str)
    # Should not contain invalid characters
    self.assertNotIn("@", key)
    self.assertNotIn("#", key)
    self.assertNotIn("!", key)

  def test_create_secure_cache_key_injection_attempt(self):
    """Test that injection attempts are neutralized."""
    malicious_input = "../../../etc/passwd"
    key = create_secure_cache_key("state", malicious_input)
    self.assertIsInstance(key, str)
    # Should not contain path traversal components
    self.assertNotIn("../", key)
    self.assertNotIn("/etc/", key)

  def test_create_secure_cache_key_empty_input(self):
    """Test that empty inputs are rejected."""
    with self.assertRaises(ValueError):
      create_secure_cache_key("", "valid")

    with self.assertRaises(ValueError):
      create_secure_cache_key("valid", "")

  def test_create_secure_cache_key_non_string_input(self):
    """Test that non-string inputs are rejected."""
    with self.assertRaises(ValueError):
      create_secure_cache_key(123, "valid")

    with self.assertRaises(ValueError):
      create_secure_cache_key("valid", None)

  def test_create_secure_cache_key_collision_resistance(self):
    """Test that different inputs produce different keys."""
    key1 = create_secure_cache_key("state", "format1")
    key2 = create_secure_cache_key("state", "format2")
    self.assertNotEqual(key1, key2)

    # Even with similar inputs
    key3 = create_secure_cache_key("state", "format")
    key4 = create_secure_cache_key("state", "format_")
    self.assertNotEqual(key3, key4)


class TestInputValidation(unittest.TestCase):
  """Test input validation security."""

  def test_sql_injection_prevention(self):
    """Test that SQL injection attempts are neutralized."""
    sql_injection = "'; DROP TABLE cache; --"

    # Should be sanitized in cache key creation
    key = create_secure_cache_key("state", sql_injection)
    # Should not contain dangerous SQL characters
    self.assertNotIn("'", key)
    self.assertNotIn(";", key)
    self.assertNotIn("--", key)
    self.assertNotIn(" ", key)  # Spaces should be removed

  def test_xss_prevention(self):
    """Test that XSS attempts are neutralized."""
    xss_attempt = "<script>alert('xss')</script>"

    # Should be sanitized in cache key creation
    key = create_secure_cache_key("state", xss_attempt)
    self.assertNotIn("<", key)
    self.assertNotIn(">", key)
    self.assertNotIn("(", key)
    self.assertNotIn(")", key)
    self.assertNotIn("'", key)

  def test_path_traversal_prevention(self):
    """Test that path traversal attempts are neutralized."""
    path_traversal = "../../../sensitive/file"

    # Should be sanitized in cache key creation
    key = create_secure_cache_key("state", path_traversal)
    self.assertNotIn(".", key)
    self.assertNotIn("/", key)


if __name__ == "__main__":
  unittest.main()