#!/bin/bash
# Imprint Setup Script for macOS
# This script creates a fresh virtual environment and installs all dependencies.

echo "🎨 Setting up Imprint environment..."

# 1. Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 could not be found. Please install Python first."
    exit 1
fi

# 2. Remove old virtual environment if it exists
if [ -d "venv" ]; then
    echo "🗑️ Removing old virtual environment..."
    rm -rf venv
fi

# 3. Create a fresh virtual environment
echo "🌱 Creating new virtual environment..."
python3 -m venv venv

# 4. Activate it
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# 5. Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# 6. Install requirements
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To run the app, simply execute:"
echo "    source venv/bin/activate"
echo "    python frontend/app_gradio.py"
