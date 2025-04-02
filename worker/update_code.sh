# #!/bin/bash

# echo "🔄 Pulling latest code from GitHub..."

# # Load GitHub token
# GITHUB_TOKEN=${GH_TOKEN}

# if [[ -z "$GITHUB_TOKEN" ]]; then
#     echo "❌ ERROR: GH_TOKEN is not set."
#     exit 1
# fi

# # Use GitHub token for authentication
# REPO_URL="https://${GITHUB_TOKEN}@github.com/michal-glos-brq/LunarPitsResearchToolbox.git"

# # Fetch latest changes without modifying files yet
# git -C /app fetch "$REPO_URL"

# # Check if there are updates in src/
# CHANGES=$(git -C /app diff --name-only origin/main -- src/)

# if [[ -z "$CHANGES" ]]; then
#     echo "✅ No updates in /app/src/. Skipping update."
#     exit 0
# fi

# echo "🔄 Updates found in /app/src/, applying changes..."

# # Pull only the src/ folder updates
# git -C /app checkout origin/main -- src/

# if [[ $? -ne 0 ]]; then
#     echo "❌ Git pull failed!"
#     exit 1
# fi

# echo "✅ /app/src/ updated."

# # Stop any running Celery workers
# echo "🛑 Stopping Celery worker: $WORKER_ID"
# pkill -f "celery worker --hostname=$WORKER_ID"

# # Start a new Celery worker with the same ID
# echo "🚀 Starting new Celery worker: $WORKER_ID"
# bash /app/run_celery.sh
