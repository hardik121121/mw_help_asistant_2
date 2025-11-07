#!/bin/bash

echo "🍉 Starting Watermelon Documentation Assistant..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit streamlit-chat
fi

# Launch Streamlit app
echo "✅ Launching Streamlit application..."
echo "📍 Application will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
