#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "========================================="
echo "==> 1. Upgrading pip & installing backend"
echo "========================================="
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

echo "========================================="
echo "==> 2. Building React Vite frontend"
echo "========================================="
if [ -d "webapp" ]; then
    cd webapp
    npm install
    npm run build
    cd ..
    echo "==> Frontend build completed successfully!"
else
    echo "==> 'webapp' directory not found, skipping frontend build."
fi

echo "========================================="
echo "==> Build process completed successfully!"
echo "========================================="
