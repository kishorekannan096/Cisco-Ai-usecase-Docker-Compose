#!/bin/bash
set -e

# Base Directories
VSS_BASE="/home/administrator/vss/video-search-and-summarization"


echo "=============================================="
echo "Initializing Manufacturing QA Infrastructure"
echo "=============================================="

# 1. Create Networks
echo "[1/4] Creating Networks..."
docker network create vss-shared-network 2>/dev/null || echo "vss-shared-network already exists"


# 2. Deploy VSS Event Reviewer
echo "[2/4] Deploying VSS Event Reviewer..."
cd "$VSS_BASE/deploy/docker/event_reviewer"
# Note: Ensure .env exists or variables are set. Assuming defaults or existing .env
docker compose up -d
echo "VSS Event Reviewer Deployed."

# 3. Deploy CV Event Detector
echo "[3/4] Deploying CV Event Detector..."
cd "$VSS_BASE/examples/cv-event-detector"
docker compose up -d
echo "CV Event Detector Deployed."



echo "=============================================="
echo "Infrastructure Deployment Complete!"
echo "Please wait a few minutes for services to become healthy."
echo "=============================================="
