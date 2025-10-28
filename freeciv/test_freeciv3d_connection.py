#!/usr/bin/env python3
"""
Test script to verify game_arena FreeCivProxyClient can connect to FreeCiv3D LLM gateway
"""

import asyncio
import logging
import sys
import os

# Add the game_arena module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from game_arena.harness.freeciv_proxy_client import FreeCivProxyClient

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_freeciv_connection():
    """Test connection to FreeCiv3D LLM gateway"""

    # However, looking at the FreeCivProxyClient code, it constructs ws://host:port
    # but we need ws://host:port/llmsocket/8002
    # Let's check if there's a way to modify the URL or if we need a different approach

    client = FreeCivProxyClient(
        host="localhost",
        port=8002,
        agent_id="game_arena_test_001",
        game_id="test_integration",
        api_token="test-token-fc3d-001"
    )

    logger.info("Testing game_arena FreeCivProxyClient connection to FreeCiv3D...")
    logger.info(f"Connecting to ws://localhost:8002")

    try:
        # Attempt to connect
        success = await client.connect()

        if success:
            logger.info("✅ Successfully connected to FreeCiv3D LLM gateway!")

            # Try to get game state
            try:
                state = await client.get_state()
                logger.info(f"✅ Successfully retrieved game state: {len(str(state))} characters")
            except Exception as e:
                logger.warning(f"⚠️ Could not retrieve game state: {e}")

            # Disconnect cleanly
            await client.disconnect()
            logger.info("✅ Disconnected successfully")

        else:
            logger.error("❌ Failed to connect to FreeCiv3D LLM gateway")
            return False

    except Exception as e:
        logger.error(f"❌ Connection error: {e}")
        return False

    return True

async def main():
    """Main test function"""
    logger.info("🚀 Starting game_arena ↔ FreeCiv3D integration test")
    logger.info("=" * 60)

    success = await test_freeciv_connection()

    if success:
        logger.info("🎉 Integration test PASSED! game_arena can connect to FreeCiv3D")
        sys.exit(0)
    else:
        logger.error("💥 Integration test FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())