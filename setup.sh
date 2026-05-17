#!/bin/bash

echo "🚀 Setting up SpaceX Landing Prediction Project..."

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create required directories
echo "Creating directories..."
mkdir -p data/raw data/processed data/external
mkdir -p models
mkdir -p logs
mkdir -p output

# Verify installation
echo "Verifying installation..."
python -c "import pandas, numpy, sklearn, xgboost, streamlit, flask; print('✓ All packages installed!')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the web app:"
echo "  streamlit run deployment/streamlit/app.py"
echo ""
echo "To run the API:"
echo "  python deployment/flask/app.py"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""