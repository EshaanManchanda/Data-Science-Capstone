#!/bin/bash
# Quick Start Script - Run the complete project

echo "🚀 SpaceX Falcon 9 Landing Prediction"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "✅ Setup complete!"
echo ""
echo "Choose an option:"
echo ""
echo "  1. Run Demo (quick test)"
echo "  2. Run Web App (Streamlit)"
echo "  3. Run API (Flask)"
echo "  4. Run Notebooks (Jupyter)"
echo "  5. Run Tests"
echo "  6. Train Model"
echo "  7. Exit"
echo ""

read -p "Enter choice (1-7): " choice

case $choice in
    1)
        echo "🎮 Running demo..."
        python demo.py
        ;;
    2)
        echo "🌐 Starting Streamlit web app..."
        streamlit run deployment/streamlit/app.py
        ;;
    3)
        echo "🌍 Starting Flask API..."
        python deployment/flask/app.py
        ;;
    4)
        echo "📓 Starting Jupyter..."
        jupyter lab notebooks/
        ;;
    5)
        echo "🧪 Running tests..."
        pytest tests/ -v
        ;;
    6)
        echo "🤖 Training model..."
        python src/main.py
        ;;
    7)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac