# GST Agent Environment - Deployment Script for Windows
# Run this script to deploy locally or prepare for HuggingFace Space

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('local', 'hf-prepare', 'test')]
    [string]$Mode = 'local',
    
    [Parameter(Mandatory=$false)]
    [string]$HFUsername = '',
    
    [Parameter(Mandatory=$false)]
    [string]$SpaceName = 'gstagent-env'
)

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GST Agent Environment Deployment" -ForegroundColor Cyan
Write-Host "  Mode: $Mode" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
function Test-Prerequisites {
    Write-Host "Checking prerequisites..." -ForegroundColor Yellow
    
    # Check Docker
    try {
        $dockerVersion = docker --version
        Write-Host "✓ Docker found: $dockerVersion" -ForegroundColor Green
    } catch {
        Write-Host "✗ Docker not found. Please install Docker Desktop." -ForegroundColor Red
        exit 1
    }
    
    # Check if Docker is running
    try {
        docker info | Out-Null
        Write-Host "✓ Docker is running" -ForegroundColor Green
    } catch {
        Write-Host "✗ Docker is not running. Starting Docker Desktop..." -ForegroundColor Yellow
        Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        Write-Host "Waiting for Docker to start (30 seconds)..." -ForegroundColor Yellow
        Start-Sleep -Seconds 30
        
        try {
            docker info | Out-Null
            Write-Host "✓ Docker started successfully" -ForegroundColor Green
        } catch {
            Write-Host "✗ Failed to start Docker. Please start it manually." -ForegroundColor Red
            exit 1
        }
    }
    
    # Check .env file
    if (Test-Path ".env") {
        Write-Host "✓ .env file found" -ForegroundColor Green
    } else {
        Write-Host "⚠ .env file not found. Creating from .env.example..." -ForegroundColor Yellow
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" ".env"
            Write-Host "✓ Created .env file. Please edit it with your API keys!" -ForegroundColor Yellow
            Write-Host "  Required: OPENAI_API_KEY" -ForegroundColor Yellow
            
            # Open .env in default editor
            Start-Process ".env"
            
            $continue = Read-Host "Press Enter after editing .env file, or 'q' to quit"
            if ($continue -eq 'q') {
                exit 0
            }
        } else {
            Write-Host "✗ .env.example not found!" -ForegroundColor Red
            exit 1
        }
    }
    
    Write-Host ""
}

# Build Docker image
function Build-DockerImage {
    Write-Host "Building Docker image..." -ForegroundColor Yellow
    Write-Host "This may take 5-10 minutes on first build..." -ForegroundColor Gray
    Write-Host ""
    
    try {
        docker build -t gstagent-env:latest .
        Write-Host ""
        Write-Host "✓ Docker image built successfully" -ForegroundColor Green
        
        # Show image details
        $imageInfo = docker images gstagent-env:latest --format "{{.Size}}"
        Write-Host "  Image size: $imageInfo" -ForegroundColor Gray
    } catch {
        Write-Host "✗ Docker build failed!" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
}

# Run Docker container locally
function Start-LocalContainer {
    Write-Host "Starting Docker container..." -ForegroundColor Yellow
    
    # Stop existing container if running
    $existing = docker ps -a --filter "name=gstagent-server" --format "{{.Names}}"
    if ($existing -eq "gstagent-server") {
        Write-Host "Stopping existing container..." -ForegroundColor Gray
        docker stop gstagent-server | Out-Null
        docker rm gstagent-server | Out-Null
    }
    
    try {
        docker run -d `
            --name gstagent-server `
            -p 7860:7860 `
            --env-file .env `
            gstagent-env:latest
        
        Write-Host "✓ Container started successfully" -ForegroundColor Green
        Write-Host ""
        
        # Wait for container to be ready
        Write-Host "Waiting for server to start..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
        
        # Test health endpoint
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:7860/health" -Method Get -TimeoutSec 10
            if ($response.status -eq "ok") {
                Write-Host "✓ Server is healthy and responding" -ForegroundColor Green
            }
        } catch {
            Write-Host "⚠ Server may still be starting up..." -ForegroundColor Yellow
        }
        
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  Server is running!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  URL: http://localhost:7860" -ForegroundColor White
        Write-Host "  Health: http://localhost:7860/health" -ForegroundColor White
        Write-Host ""
        Write-Host "Useful commands:" -ForegroundColor Yellow
        Write-Host "  View logs:    docker logs -f gstagent-server" -ForegroundColor Gray
        Write-Host "  Stop server:  docker stop gstagent-server" -ForegroundColor Gray
        Write-Host "  Remove:       docker rm gstagent-server" -ForegroundColor Gray
        Write-Host ""
        
    } catch {
        Write-Host "✗ Failed to start container!" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        
        # Show logs
        Write-Host ""
        Write-Host "Container logs:" -ForegroundColor Yellow
        docker logs gstagent-server
        exit 1
    }
}

# Test the deployment
function Test-Deployment {
    Write-Host "Testing deployment..." -ForegroundColor Yellow
    Write-Host ""
    
    # Test 1: Health check
    Write-Host "Test 1: Health check..." -ForegroundColor Gray
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:7860/health" -Method Get
        if ($response.status -eq "ok") {
            Write-Host "  ✓ Health check passed" -ForegroundColor Green
        } else {
            Write-Host "  ✗ Health check failed" -ForegroundColor Red
        }
    } catch {
        Write-Host "  ✗ Health check failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    # Test 2: Reset endpoint
    Write-Host "Test 2: Reset endpoint..." -ForegroundColor Gray
    try {
        $body = @{ task_id = "invoice_match" } | ConvertTo-Json
        $response = Invoke-RestMethod -Uri "http://localhost:7860/reset" -Method Post -Body $body -ContentType "application/json"
        if ($response.session_id) {
            Write-Host "  ✓ Reset endpoint passed (session: $($response.session_id.Substring(0,8))...)" -ForegroundColor Green
            $sessionId = $response.session_id
            
            # Test 3: Step endpoint
            Write-Host "Test 3: Step endpoint..." -ForegroundColor Gray
            try {
                $stepBody = @{
                    session_id = $sessionId
                    action = @{
                        action_type = "match_invoice"
                        invoice_id = "INV-001"
                    }
                } | ConvertTo-Json -Depth 3
                
                $stepResponse = Invoke-RestMethod -Uri "http://localhost:7860/step" -Method Post -Body $stepBody -ContentType "application/json"
                Write-Host "  ✓ Step endpoint passed" -ForegroundColor Green
            } catch {
                Write-Host "  ✗ Step endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
            }
        } else {
            Write-Host "  ✗ Reset endpoint failed" -ForegroundColor Red
        }
    } catch {
        Write-Host "  ✗ Reset endpoint failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Testing complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
}

# Prepare for HuggingFace Space deployment
function Prepare-HFSpace {
    Write-Host "Preparing for HuggingFace Space deployment..." -ForegroundColor Yellow
    Write-Host ""
    
    if (-not $HFUsername) {
        $HFUsername = Read-Host "Enter your HuggingFace username"
    }
    
    # Create README.md with HF frontmatter
    Write-Host "Creating HuggingFace Space README..." -ForegroundColor Gray
    
    $readmeContent = @"
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

- ``GET /health`` - Health check
- ``POST /reset`` - Reset environment
- ``POST /step`` - Execute action
- ``GET /state/{session_id}`` - Get current state
- ``GET /leaderboard`` - View leaderboard

## Usage

````python
import requests

# Reset environment
response = requests.post(
    "https://$HFUsername-$SpaceName.hf.space/reset",
    json={"task_id": "invoice_match"}
)
session_id = response.json()["session_id"]

# Execute action
response = requests.post(
    "https://$HFUsername-$SpaceName.hf.space/step",
    json={
        "session_id": session_id,
        "action": {
            "action_type": "match_invoice",
            "invoice_id": "INV-001"
        }
    }
)
````

## Documentation

See [implementation_plan.md](implementation_plan.md) for architecture details.
"@
    
    $readmeContent | Out-File -FilePath "README.md" -Encoding UTF8
    Write-Host "✓ Created README.md with HuggingFace frontmatter" -ForegroundColor Green
    
    # Check if git is initialized
    if (-not (Test-Path ".git")) {
        Write-Host ""
        Write-Host "Initializing git repository..." -ForegroundColor Gray
        git init
        Write-Host "✓ Git repository initialized" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  HuggingFace Space Preparation Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Create a new Space at: https://huggingface.co/new-space" -ForegroundColor White
    Write-Host "   - Name: $SpaceName" -ForegroundColor Gray
    Write-Host "   - SDK: Docker" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Add your Space as a git remote:" -ForegroundColor White
    Write-Host "   git remote add hf https://huggingface.co/spaces/$HFUsername/$SpaceName" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. Configure secrets in Space settings:" -ForegroundColor White
    Write-Host "   - OPENAI_API_KEY: Your OpenAI API key" -ForegroundColor Gray
    Write-Host "   - GST_API_KEY: (optional) Your API auth key" -ForegroundColor Gray
    Write-Host ""
    Write-Host "4. Push to HuggingFace:" -ForegroundColor White
    Write-Host "   git add ." -ForegroundColor Gray
    Write-Host "   git commit -m 'Deploy to HuggingFace Space'" -ForegroundColor Gray
    Write-Host "   git push hf main" -ForegroundColor Gray
    Write-Host ""
}

# Main execution
switch ($Mode) {
    'local' {
        Test-Prerequisites
        Build-DockerImage
        Start-LocalContainer
        
        $runTests = Read-Host "Run tests? (y/n)"
        if ($runTests -eq 'y') {
            Test-Deployment
        }
    }
    
    'hf-prepare' {
        Test-Prerequisites
        Build-DockerImage
        Prepare-HFSpace
    }
    
    'test' {
        Test-Deployment
    }
}

Write-Host "Done!" -ForegroundColor Green
