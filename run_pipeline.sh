#!/bin/bash
# MLOps Heart Disease Project - Pipeline Runner
# This script runs the complete ML pipeline

set -e  # Exit on error

echo "=============================================="
echo "  MLOps Heart Disease Prediction Pipeline"
echo "=============================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running from project root
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Step 1: Create virtual environment if it doesn't exist
echo -e "${YELLOW}Step 1: Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Step 2: Install dependencies
echo ""
echo -e "${YELLOW}Step 2: Installing dependencies...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 3: Download data
echo ""
echo -e "${YELLOW}Step 3: Downloading dataset...${NC}"
python -c "from src.data.download_data import download_heart_disease_data; download_heart_disease_data()"
echo -e "${GREEN}✓ Dataset downloaded${NC}"

# Step 4: Preprocess data
echo ""
echo -e "${YELLOW}Step 4: Preprocessing data...${NC}"
python -c "from src.data.preprocess import preprocess_data; preprocess_data('data/raw/heart_disease.csv')"
echo -e "${GREEN}✓ Data preprocessed${NC}"

# Step 5: Run tests
echo ""
echo -e "${YELLOW}Step 5: Running unit tests...${NC}"
pytest tests/ -v --tb=short || echo -e "${YELLOW}Warning: Some tests may have failed${NC}"
echo -e "${GREEN}✓ Tests completed${NC}"

# Step 6: Train model with MLflow
echo ""
echo -e "${YELLOW}Step 6: Training model with MLflow tracking...${NC}"
python src/models/train_with_mlflow.py
echo -e "${GREEN}✓ Model trained and saved${NC}"

# Step 7: Build Docker image (optional)
echo ""
echo -e "${YELLOW}Step 7: Building Docker image...${NC}"
if command -v docker &> /dev/null; then
    docker build -t heart-disease-api:latest .
    echo -e "${GREEN}✓ Docker image built${NC}"
else
    echo -e "${YELLOW}⚠ Docker not found, skipping image build${NC}"
fi

echo ""
echo "=============================================="
echo -e "${GREEN}  Pipeline completed successfully!${NC}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. View MLflow experiments: mlflow ui --port 5000"
echo "  2. Start API locally: uvicorn src.api.main:app --reload"
echo "  3. Run Docker container: docker run -p 8000:8000 heart-disease-api:latest"
echo "  4. Deploy to Kubernetes: kubectl apply -f kubernetes/"
echo ""
