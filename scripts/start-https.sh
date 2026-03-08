#!/bin/bash
# Start HTTPS proxy for ModelStag
#
# Usage: ./start-https.sh [port] [backend_port]
# Defaults: HTTPS on 8443, backend on 8000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CERTS_DIR="$PROJECT_DIR/certs"

HTTPS_PORT="${1:-8443}"
BACKEND_PORT="${2:-8000}"
BACKEND_URL="http://localhost:$BACKEND_PORT"

# Check if certificates exist
if [ ! -f "$CERTS_DIR/cert.pem" ] || [ ! -f "$CERTS_DIR/key.pem" ]; then
    echo "Certificates not found. Generating..."
    "$SCRIPT_DIR/generate-certs.sh"
    echo ""
fi

# Check if backend is running
if ! curl -s "http://localhost:$BACKEND_PORT/health" > /dev/null 2>&1; then
    echo "WARNING: Backend server not responding at $BACKEND_URL"
    echo "Make sure to start the ModelStag server first:"
    echo "  poetry run modelstag serve"
    echo ""
fi

echo "Starting HTTPS proxy..."
python3 "$SCRIPT_DIR/https_proxy.py" --port "$HTTPS_PORT" --backend "$BACKEND_URL"
