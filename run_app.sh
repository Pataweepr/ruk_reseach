#!/bin/bash

# Pose Detection Application Launcher
# This script sets up and runs the pose detection application

echo "🎯 Pose Detection Application Launcher"
echo "======================================"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✅ Conda found"
    
    # Check if ruk_pos environment exists
    if conda env list | grep -q "ruk_pos"; then
        echo "✅ ruk_pos environment found"
        echo "🔄 Activating ruk_pos environment..."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate ruk_pos
    else
        echo "❌ ruk_pos environment not found"
        echo "🔧 Creating ruk_pos environment..."
        conda create -n ruk_pos python=3.11 -y
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate ruk_pos
        echo "📦 Installing dependencies..."
        pip install -r requirements.txt
    fi
else
    echo "⚠️  Conda not found, using system Python"
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "📁 Creating models directory..."
    mkdir -p models
    echo "⚠️  Note: OpenPose models not found. The app will use fallback detection."
    echo "   To install OpenPose models, see setup_environment.md"
fi

# Check if templates directory exists
if [ ! -d "templates" ]; then
    echo "❌ Templates directory not found!"
    echo "   Please ensure templates/pose_estimation.html exists"
    exit 1
fi

# Check if main application file exists
if [ ! -f "pose_estimation_app.py" ]; then
    echo "❌ Main application file not found!"
    echo "   Please ensure pose_estimation_app.py exists"
    exit 1
fi

echo ""
echo "🚀 Starting Pose Estimation Application..."
echo "   Access the application at: http://localhost:5001"
echo "   Press Ctrl+C to stop the application"
echo ""

# Run the application
python pose_estimation_app.py
