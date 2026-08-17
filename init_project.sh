#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Script: init_project.sh
# Description: Generates directory structure and base scaffolding for
#              Chatbot-Engine-Gateway (AI Agent Gateway microservice).
# ==============================================================================

echo "🚀 Initializing AI Agent Gateway microservice structure..."

# 1. Create directories
mkdir -p app/core
mkdir -p app/api/v1
mkdir -p app/agents
mkdir -p app/services
mkdir -p app/schemas

# 2. Create Python package markers (__init__.py)
touch app/__init__.py
touch app/core/__init__.py
touch app/api/__init__.py
touch app/api/v1/__init__.py
touch app/agents/__init__.py
touch app/services/__init__.py
touch app/schemas/__init__.py

# 3. Create module placeholder files if they don't already exist
touch app/core/config.py
touch app/core/security.py
touch app/api/v1/chat.py
touch app/agents/base.py
touch app/agents/dispatcher.py
touch app/services/llm_client.py
touch app/services/memory.py
touch app/services/django_api.py
touch app/schemas/payload.py
touch app/main.py

# 4. Create base environment and dependency files
touch requirements.txt
touch .env.example
touch .gitignore

echo "✅ Project scaffolding successfully created!"
