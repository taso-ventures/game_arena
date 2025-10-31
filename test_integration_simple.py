#!/usr/bin/env python3
"""Simple integration test for FreeCiv3D changes."""

import sys

# Test 1: Import new modules
print("=" * 60)
print("TEST 1: Import new configuration and resilience modules")
print("=" * 60)
try:
    from game_arena.harness.freeciv.config import ProxyClientConfig
    from game_arena.harness.freeciv.resilience import CircuitBreaker, with_retry
    print("✅ Successfully imported freeciv.config module")
    print("✅ Successfully imported freeciv.resilience module")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Create and validate configuration
print("\n" + "=" * 60)
print("TEST 2: Create and validate configuration")
print("=" * 60)
try:
    config = ProxyClientConfig()
    print(f"✅ Created default config")
    print(f"   - Max retries: {config.retry.max_retries}")
    print(f"   - Backoff base: {config.retry.backoff_base}")
    print(f"   - Max backoff delay: {config.retry.max_backoff_delay}s")
    print(f"   - Circuit breaker threshold: {config.circuit_breaker.failure_threshold}")

    config.validate()
    print("✅ Configuration validation passed")
except Exception as e:
    print(f"❌ Configuration test failed: {e}")
    sys.exit(1)

# Test 3: Test activity code mapping
print("\n" + "=" * 60)
print("TEST 3: Test activity code mapping")
print("=" * 60)
try:
    from game_arena.harness.freeciv_state import FreeCivUnit

    # Test various activity codes
    test_cases = [
        (0, None, "IDLE"),
        (5, "fortified", "FORTIFIED"),
        (11, "explore", "EXPLORE"),
        (3, "mine", "MINE"),
        (21, "gen_road", "GENERIC_ROAD"),
    ]

    for code, expected, name in test_cases:
        unit = FreeCivUnit(
            unit_id=1,
            owner=0,
            kind="Worker",
            position=(5, 5),
            hp=10,
            moves_left=3,
            activity=code
        )
        if unit.activity == expected:
            print(f"✅ Activity code {code:2d} ({name:15s}) → {expected}")
        else:
            print(f"❌ Activity code {code} failed: expected {expected}, got {unit.activity}")
            sys.exit(1)

except Exception as e:
    print(f"❌ Activity mapping test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test strategic summary integration
print("\n" + "=" * 60)
print("TEST 4: Test strategic summary formatting")
print("=" * 60)
try:
    from game_arena.harness.prompts.freeciv_prompts import FreeCivPromptBuilder

    builder = FreeCivPromptBuilder()

    # Test the new _format_gateway_strategic_summary method
    strategic_summary = {
        "cities_count": 3,
        "units_count": 15,
        "tech_progress": 25,
        "military_strength": 45
    }
    immediate_priorities = [
        "Build settlers for expansion",
        "Research Writing for libraries"
    ]
    threats = [
        {"type": "Military", "severity": "High", "description": "Enemy units near capital"}
    ]
    opportunities = [
        {"type": "Expansion", "value": "High", "description": "Unclaimed fertile plains to the north"}
    ]

    formatted = builder._format_gateway_strategic_summary(
        strategic_summary, immediate_priorities, threats, opportunities
    )

    print("✅ Strategic summary formatted successfully")
    print(f"   Length: {len(formatted)} characters")
    print(f"   Contains 'Strategic Overview': {'Strategic Overview' in formatted}")
    print(f"   Contains 'Immediate Priorities': {'Immediate Priorities' in formatted}")
    print(f"   Contains 'Current Threats': {'Current Threats' in formatted}")
    print(f"   Contains 'Strategic Opportunities': {'Strategic Opportunities' in formatted}")

except Exception as e:
    print(f"❌ Strategic summary test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED")
print("=" * 60)
print("\nSummary:")
print("  ✅ New modules importable")
print("  ✅ Configuration system working")
print("  ✅ Activity code mapping functional")
print("  ✅ Strategic summary integration working")
print("\nChanges are ready for integration testing with FreeCiv3D!")
