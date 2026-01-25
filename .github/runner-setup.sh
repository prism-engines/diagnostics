#!/bin/bash
# PRISM Self-Hosted Runner Setup
#
# Run this once to set up the GitHub Actions runner on your machine.
#
# Prerequisites:
#   - GitHub Personal Access Token with repo scope
#   - macOS or Linux
#
# Usage:
#   chmod +x .github/runner-setup.sh
#   ./.github/runner-setup.sh

set -e

RUNNER_DIR="$HOME/actions-runner"
REPO="prism-engines/prism"

echo "=== PRISM Self-Hosted Runner Setup ==="
echo ""

# Check for token
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: GITHUB_TOKEN environment variable not set"
    echo ""
    echo "1. Go to: https://github.com/settings/tokens"
    echo "2. Generate a new token with 'repo' scope"
    echo "3. Run: export GITHUB_TOKEN=your_token_here"
    echo "4. Re-run this script"
    exit 1
fi

# Create runner directory
mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="osx"
    ARCH="arm64"  # For Apple Silicon, use "x64" for Intel
else
    OS="linux"
    ARCH="x64"
fi

# Download runner
RUNNER_VERSION="2.311.0"
RUNNER_URL="https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-${OS}-${ARCH}-${RUNNER_VERSION}.tar.gz"

echo "Downloading runner..."
curl -o actions-runner.tar.gz -L "$RUNNER_URL"
tar xzf actions-runner.tar.gz
rm actions-runner.tar.gz

# Get registration token
echo "Getting registration token..."
REG_TOKEN=$(curl -s -X POST \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    "https://api.github.com/repos/$REPO/actions/runners/registration-token" \
    | grep -o '"token": "[^"]*"' | cut -d'"' -f4)

if [ -z "$REG_TOKEN" ]; then
    echo "Error: Could not get registration token"
    echo "Make sure GITHUB_TOKEN has 'repo' scope"
    exit 1
fi

# Configure runner
echo "Configuring runner..."
./config.sh --url "https://github.com/$REPO" \
    --token "$REG_TOKEN" \
    --name "prism-local" \
    --labels "self-hosted,prism,macos" \
    --work "_work" \
    --unattended

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the runner:"
echo "  cd $RUNNER_DIR"
echo "  ./run.sh"
echo ""
echo "To install as a service (runs on boot):"
echo "  cd $RUNNER_DIR"
echo "  sudo ./svc.sh install"
echo "  sudo ./svc.sh start"
echo ""
echo "Runner will pick up jobs from: https://github.com/$REPO/actions"
