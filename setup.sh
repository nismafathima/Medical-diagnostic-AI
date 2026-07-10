#!/bin/bash
# MedAI Setup Script - CPU-Optimized Edition
# Usage: bash setup.sh
# Windows users: Use setup.bat or run commands manually

set -e  # Exit on error

echo "============================================"
echo "🏥 MedAI Assistant - CPU Optimization Setup"
echo "============================================"
echo ""

# Check Python version
echo "✓ Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

# Create virtual environment
echo ""
echo "✓ Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "  Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "✓ Activating virtual environment..."
source venv/bin/activate
echo "  Virtual environment activated"

# Upgrade pip
echo ""
echo "✓ Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "  pip upgraded"

# Install requirements
echo ""
echo "✓ Installing dependencies (CPU-optimized)..."
pip install -r requirements.txt
echo "  Dependencies installed"

# Create cache directory
echo ""
echo "✓ Creating cache directories..."
mkdir -p ~/.cache/huggingface
mkdir -p .streamlit
echo "  Cache directories created"

# Create Streamlit config
echo ""
echo "✓ Creating Streamlit config..."
if [ ! -f ".streamlit/config.toml" ]; then
    cat > ".streamlit/config.toml" << 'EOF'
[client]
showErrorDetails = true
logger.level = "info"

[server]
maxUploadSize = 200
maxMessageSize = 200
runOnSave = false

[browser]
gatherUsageStats = false

[logger]
level = "info"
EOF
    echo "  Streamlit config created"
else
    echo "  Streamlit config already exists. Skipping..."
fi

# Create environment file
echo ""
echo "✓ Creating environment file..."
if [ ! -f ".env" ]; then
    cat > ".env" << 'EOF'
# MedAI Environment Configuration
# CPU Optimization Settings

# Set number of PyTorch threads (adjust based on your CPU cores)
# Recommended: 4-8 for most systems
TORCH_NUM_THREADS=4

# Optional: OpenAI API Key (leave empty to use local model)
# OPENAI_API_KEY=sk-...

# Cache directory
HF_HOME=~/.cache/huggingface
EOF
    echo "  Environment file created"
else
    echo "  Environment file already exists. Skipping..."
fi

echo ""
echo "============================================"
echo "✅ Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment (if not already):"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the application:"
echo "   python app.py"
echo ""
echo "3. Open browser to:"
echo "   http://localhost:7860"
echo ""
echo "📝 Note: First run will download models (~1.5GB, 5-15 minutes)"
echo ""
echo "For more information, see:"
echo "  - README_OPTIMIZED.md (Getting started)"
echo "  - OPTIMIZATION_GUIDE.md (Performance details)"
echo ""
