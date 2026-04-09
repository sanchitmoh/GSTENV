#!/bin/bash
# GST Agent Environment - Deployment Script for Linux/Mac
# Run this script to deploy locally or prepare for HuggingFace Space

set -e

MODE="${1:-local}"
HF_USERNAME="${2:-}"
SPACE_NAME="${3:-gstagent-env}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;37m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  GST Agent Environment Deployment${NC}"
echo -e "${CYAN}  Mode: $MODE${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Docker
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version)
        echo -e "${GREEN}✓ Docker found: $DOCKER_VERSION${NC}"
    else
        echo -e "${RED}✗ Docker not found. Please install Docker.${NC}"
        exit 1
    fi
    
    # Check if Docker is running
    if docker info &> /dev/null; then
        echo -e "${GREEN}✓ Docker is running${NC}"
    else
        echo -e "${RED}✗ Docker is not running. Please start Docker.${NC}"
        exit 1
    fi
    
    # Check .env file
    if [ -f ".env" ]; then
        echo -e "${GREEN}✓ .env file found${NC}"
    else
        echo -e "${YELLOW}⚠ .env file not found. Creating from .env.example...${NC}"
        if [ -f ".env.example" ]; then
            cp .env.example .env
            echo -e "${YELLOW}✓ Created .env file. Please edit it with your API keys!${NC}"
            echo -e "${YELLOW}  Required: OPENAI_API_KEY${NC}"
            
            # Open .env in default editor
            ${EDITOR:-nano} .env
            
            read -p "Press Enter after editing .env file, or 'q' to quit: " continue
            if [ "$continue" = "q" ]; then
                exit 0
            fi
        else
            echo -e "${RED}✗ .env.example not found!${NC}"
            exit 1
        fi
    fi
    
    echo ""
}

# Build Docker image
build_docker_image() {
    echo -e "${YELLOW}Building Docker image...${NC}"
    echo -e "${GRAY}This may take 5-10 minutes on first build...${NC}"
    echo ""
    
    if docker build -t gstagent-env:latest .; then
        echo ""
        echo -e "${GREEN}✓ Docker image built successfully${NC}"
        
        # Show image details
        IMAGE_SIZE=$(docker images gstagent-env:latest --format "{{.Size}}")
        echo -e "${GRAY}  Image size: $IMAGE_SIZE${NC}"
    else
        echo -e "${RED}✗ Docker build failed!${NC}"
        exit 1
    fi
    
    echo ""
}

# Run Docker container locally
start_local_container() {
    echo -e "${YELLOW}Starting Docker container...${NC}"
    
    # Stop existing container if running
    if docker ps -a --filter "name=gstagent-server" --format "{{.Names}}" | grep -q "gstagent-server"; then
        echo -e "${GRAY}Stopping existing container...${NC}"
        docker stop gstagent-server > /dev/null 2>&1 || true
        docker rm gstagent-server > /dev/null 2>&1 || true
    fi
    
    if docker run -d \
        --name gstagent-server \
        -p 7860:7860 \
        --env-file .env \
        gstagent-env:latest; then
        
        echo -e "${GREEN}✓ Container started successfully${NC}"
        echo ""
        
        # Wait for container to be ready
        echo -e "${YELLOW}Waiting for server to start...${NC}"
        sleep 5
        
        # Test health endpoint
        if curl -s http://localhost:7860/health | grep -q "ok"; then
            echo -e "${GREEN}✓ Server is healthy and responding${NC}"
        else
            echo -e "${YELLOW}⚠ Server may still be starting up...${NC}"
        fi
        
        echo ""
        echo -e "${CYAN}========================================${NC}"
        echo -e "${GREEN}  Server is running!${NC}"
        echo -e "${CYAN}========================================${NC}"
        echo -e "  URL: http://localhost:7860"
        echo -e "  Health: http://localhost:7860/health"
        echo ""
        echo -e "${YELLOW}Useful commands:${NC}"
        echo -e "${GRAY}  View logs:    docker logs -f gstagent-server${NC}"
        echo -e "${GRAY}  Stop server:  docker stop gstagent-server${NC}"
        echo -e "${GRAY}  Remove:       docker rm gstagent-server${NC}"
        echo ""
        
    else
        echo -e "${RED}✗ Failed to start container!${NC}"
        echo ""
        echo -e "${YELLOW}Container logs:${NC}"
        docker logs gstagent-server
        exit 1
    fi
}

# Test the deployment
test_deployment() {
    echo -e "${YELLOW}Testing deployment...${NC}"
    echo ""
    
    # Test 1: Health check
    echo -e "${GRAY}Test 1: Health check...${NC}"
    if curl -s http://localhost:7860/health | grep -q "ok"; then
        echo -e "${GREEN}  ✓ Health check passed${NC}"
    else
        echo -e "${RED}  ✗ Health check failed${NC}"
    fi
    
    # Test 2: Reset endpoint
    echo -e "${GRAY}Test 2: Reset endpoint...${NC}"
    RESET_RESPONSE=$(curl -s -X POST http://localhost:7860/reset \
        -H "Content-Type: application/json" \
        -d '{"task_id": "invoice_match"}')
    
    if echo "$RESET_RESPONSE" | grep -q "session_id"; then
        SESSION_ID=$(echo "$RESET_RESPONSE" | grep -o '"session_id":"[^"]*"' | cut -d'"' -f4)
        echo -e "${GREEN}  ✓ Reset endpoint passed (session: ${SESSION_ID:0:8}...)${NC}"
        
        # Test 3: Step endpoint
        echo -e "${GRAY}Test 3: Step endpoint...${NC}"
        STEP_RESPONSE=$(curl -s -X POST http://localhost:7860/step \
            -H "Content-Type: application/json" \
            -d "{\"session_id\": \"$SESSION_ID\", \"action\": {\"action_type\": \"match_invoice\", \"invoice_id\": \"INV-001\"}}")
        
        if echo "$STEP_RESPONSE" | grep -q "step_number"; then
            echo -e "${GREEN}  ✓ Step endpoint passed${NC}"
        else
            echo -e "${RED}  ✗ Step endpoint failed${NC}"
        fi
    else
        echo -e "${RED}  ✗ Reset endpoint failed${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${GREEN}  Testing complete!${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

# Prepare for HuggingFace Space deployment
prepare_hf_space() {
    echo -e "${YELLOW}Preparing for HuggingFace Space deployment...${NC}"
    echo ""
    
    if [ -z "$HF_USERNAME" ]; then
        read -p "Enter your HuggingFace username: " HF_USERNAME
    fi
    
    # Create README.md with HF frontmatter
    echo -e "${GRAY}Creating HuggingFace Space README...${NC}"
    
    cat > README.md << EOF
---
title: GST Agent Environment
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
app_port: 7860
---

# GST Agent Environment

OpenEnv-compatible RL environment for Indian GST reconciliation.

## Features

- Invoice matching and reconciliation
- ITC (Input Tax Credit) audit
- Multi-agent orchestration with RAG
- Deterministic grading system

## API Endpoints

- \`GET /health\` - Health check
- \`POST /reset\` - Reset environment
- \`POST /step\` - Execute action
- \`GET /state/{session_id}\` - Get current state
- \`GET /leaderboard\` - View leaderboard

## Usage

\`\`\`python
import requests

# Reset environment
response = requests.post(
    "https://$HF_USERNAME-$SPACE_NAME.hf.space/reset",
    json={"task_id": "invoice_match"}
)
session_id = response.json()["session_id"]

# Execute action
response = requests.post(
    "https://$HF_USERNAME-$SPACE_NAME.hf.space/step",
    json={
        "session_id": session_id,
        "action": {
            "action_type": "match_invoice",
            "invoice_id": "INV-001"
        }
    }
)
\`\`\`

## Documentation

See [implementation_plan.md](implementation_plan.md) for architecture details.
EOF
    
    echo -e "${GREEN}✓ Created README.md with HuggingFace frontmatter${NC}"
    
    # Check if git is initialized
    if [ ! -d ".git" ]; then
        echo ""
        echo -e "${GRAY}Initializing git repository...${NC}"
        git init
        echo -e "${GREEN}✓ Git repository initialized${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${GREEN}  HuggingFace Space Preparation Complete!${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "1. Create a new Space at: https://huggingface.co/new-space"
    echo -e "${GRAY}   - Name: $SPACE_NAME${NC}"
    echo -e "${GRAY}   - SDK: Docker${NC}"
    echo ""
    echo -e "2. Add your Space as a git remote:"
    echo -e "${GRAY}   git remote add hf https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME${NC}"
    echo ""
    echo -e "3. Configure secrets in Space settings:"
    echo -e "${GRAY}   - OPENAI_API_KEY: Your OpenAI API key${NC}"
    echo -e "${GRAY}   - GST_API_KEY: (optional) Your API auth key${NC}"
    echo ""
    echo -e "4. Push to HuggingFace:"
    echo -e "${GRAY}   git add .${NC}"
    echo -e "${GRAY}   git commit -m 'Deploy to HuggingFace Space'${NC}"
    echo -e "${GRAY}   git push hf main${NC}"
    echo ""
}

# Main execution
case "$MODE" in
    local)
        check_prerequisites
        build_docker_image
        start_local_container
        
        read -p "Run tests? (y/n): " run_tests
        if [ "$run_tests" = "y" ]; then
            test_deployment
        fi
        ;;
    
    hf-prepare)
        check_prerequisites
        build_docker_image
        prepare_hf_space
        ;;
    
    test)
        test_deployment
        ;;
    
    *)
        echo "Usage: $0 {local|hf-prepare|test} [hf_username] [space_name]"
        echo ""
        echo "Modes:"
        echo "  local       - Build and run locally"
        echo "  hf-prepare  - Prepare for HuggingFace Space deployment"
        echo "  test        - Test existing deployment"
        exit 1
        ;;
esac

echo -e "${GREEN}Done!${NC}"
