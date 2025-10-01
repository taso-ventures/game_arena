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

FROM ubuntu:22.04

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies for FreeCiv3D server
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    nodejs \
    npm \
    supervisor \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /opt/freeciv3d

# Clone FreeCiv3D repository (using mock for testing)
# In production, this would clone the actual FreeCiv3D repo
RUN mkdir -p freeciv-web freeciv-proxy publite2

# Create mock FreeCiv3D server structure
COPY docker/freeciv3d/mock-server.js freeciv-web/
COPY docker/freeciv3d/mock-proxy.py freeciv-proxy/
COPY docker/freeciv3d/package.json freeciv-web/

# Install Node.js dependencies
WORKDIR /opt/freeciv3d/freeciv-web
RUN npm install

# Install Python dependencies for proxy
WORKDIR /opt/freeciv3d/freeciv-proxy
RUN pip3 install websockets asyncio json logging

# Create supervisor configuration
COPY docker/freeciv3d/supervisord.conf /etc/supervisor/conf.d/freeciv3d.conf

# Expose ports
EXPOSE 8080 8443 7000

# Set up health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD nc -z localhost 8080 && nc -z localhost 7000 || exit 1

# Create startup script
COPY docker/freeciv3d/start.sh /opt/freeciv3d/start.sh
RUN chmod +x /opt/freeciv3d/start.sh

WORKDIR /opt/freeciv3d

CMD ["/opt/freeciv3d/start.sh"]