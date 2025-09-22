#!/bin/bash
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

set -e

# Script to run end-to-end tests with Docker containers

echo "🚀 Starting End-to-End FreeCiv Integration Tests"

# Function to cleanup containers
cleanup() {
  echo "🧹 Cleaning up containers..."
  docker-compose -f docker-compose.e2e.yml down --volumes --remove-orphans
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
  echo "❌ docker-compose is required but not installed"
  exit 1
fi

# Build and start containers
echo "🏗️  Building Docker containers..."
docker-compose -f docker-compose.e2e.yml build

echo "🏃 Starting FreeCiv3D server..."
docker-compose -f docker-compose.e2e.yml up -d freeciv3d

# Wait for server to be healthy
echo "⏳ Waiting for FreeCiv3D server to be ready..."
timeout 60 bash -c 'until docker-compose -f docker-compose.e2e.yml exec freeciv3d nc -z localhost 8080; do sleep 2; done'

if [ $? -ne 0 ]; then
  echo "❌ FreeCiv3D server failed to start within 60 seconds"
  docker-compose -f docker-compose.e2e.yml logs freeciv3d
  exit 1
fi

echo "✅ FreeCiv3D server is ready"

# Run tests
echo "🧪 Running end-to-end tests..."
if docker-compose -f docker-compose.e2e.yml run --rm game-arena-test; then
  echo "✅ All end-to-end tests passed!"
  exit 0
else
  echo "❌ Some end-to-end tests failed"
  echo "📋 FreeCiv3D Server Logs:"
  docker-compose -f docker-compose.e2e.yml logs freeciv3d
  echo "📋 Test Container Logs:"
  docker-compose -f docker-compose.e2e.yml logs game-arena-test
  exit 1
fi