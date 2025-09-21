#!/bin/bash

## Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONTAINER_NAME="nodule_seg_pipeline"

# Load environment variables if .env file exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    print_status "Loading environment variables from .env file"
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
elif [ -f "$PROJECT_ROOT/.env.template" ]; then
    print_warning "No .env file found. Copy .env.template to .env and customize paths"
    print_warning "Using default paths for now"
fi

# Set default environment variables if not set
export PUID=${PUID:-$(id -u)}
export PGID=${PGID:-$(id -g)}
export DATA_PATH=${DATA_PATH:-"$PROJECT_ROOT/demofolder/data"}
export OUTPUT_PATH=${OUTPUT_PATH:-"$PROJECT_ROOT/output"}ical Imaging Nodule Segmentation Pipeline - Run Script
# Easy execution of the containerized pipeline

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONTAINER_NAME="nodule_seg_pipeline"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE="docker-compose"
    elif docker compose version &> /dev/null; then
        DOCKER_COMPOSE="docker compose"
    else
        print_error "Docker Compose is not available"
        exit 1
    fi
}

# Function to show help
show_help() {
    echo "Medical Imaging Nodule Segmentation Pipeline - Run Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start           Start the container"
    echo "  stop            Stop the container"
    echo "  restart         Restart the container"
    echo "  run             Run the segmentation pipeline"
    echo "  shell           Open interactive shell in container"
    echo "  jupyter         Start Jupyter notebook service"
    echo "  logs            Show container logs"
    echo "  status          Show container status"
    echo "  clean           Clean up containers and volumes"
    echo "  custom          Run custom command in container"
    echo ""
    echo "Options:"
    echo "  -h, --help      Show this help message"
    echo "  -f, --follow    Follow logs (for logs command)"
    echo "  -v, --verbose   Verbose output"
    echo ""
    echo "Examples:"
    echo "  $0 start                    # Start the container"
    echo "  $0 run                      # Run the segmentation pipeline"
    echo "  $0 shell                    # Open interactive shell"
    echo "  $0 logs -f                  # Follow logs"
    echo "  $0 custom 'python3 --version'  # Run custom command"
    echo ""
}

# Function to check container status
check_status() {
    if $DOCKER_COMPOSE ps | grep -q "$CONTAINER_NAME.*Up"; then
        return 0  # Container is running
    else
        return 1  # Container is not running
    fi
}

# Function to start the container
start_container() {
    print_status "Starting the nodule segmentation container..."
    
    cd "$PROJECT_ROOT"
    
    if check_status; then
        print_warning "Container is already running"
        return 0
    fi
    
    if $DOCKER_COMPOSE up -d nodule-segmentation; then
        print_success "Container started successfully!"
        
        # Wait for container to be ready
        print_status "Waiting for container to be ready..."
        sleep 5
        
        # Check health
        if $DOCKER_COMPOSE exec nodule-segmentation python3 -c "import SimpleITK; print('Container is ready!')" 2>/dev/null; then
            print_success "Container is healthy and ready!"
        else
            print_warning "Container started but health check failed"
        fi
    else
        print_error "Failed to start container"
        exit 1
    fi
}

# Function to stop the container
stop_container() {
    print_status "Stopping the nodule segmentation container..."
    
    cd "$PROJECT_ROOT"
    
    if ! check_status; then
        print_warning "Container is not running"
        return 0
    fi
    
    if $DOCKER_COMPOSE stop nodule-segmentation; then
        print_success "Container stopped successfully!"
    else
        print_error "Failed to stop container"
        exit 1
    fi
}

# Function to restart the container
restart_container() {
    print_status "Restarting the nodule segmentation container..."
    stop_container
    sleep 2
    start_container
}

# Function to run the segmentation pipeline
run_pipeline() {
    print_status "Running the nodule segmentation pipeline..."
    
    cd "$PROJECT_ROOT"
    
    if ! check_status; then
        print_status "Container is not running. Starting it first..."
        start_container
    fi
    
    # Check if the segmentation script exists
    if $DOCKER_COMPOSE exec nodule-segmentation test -f "/app/scripts/DLCS24_KNN_2mm_Extend_Seg.sh"; then
        print_status "Executing segmentation pipeline..."
        if $DOCKER_COMPOSE exec nodule-segmentation bash /app/scripts/DLCS24_KNN_2mm_Extend_Seg.sh; then
            print_success "Pipeline executed successfully!"
        else
            print_error "Pipeline execution failed"
            exit 1
        fi
    else
        print_error "Segmentation script not found in container"
        print_status "Available scripts:"
        $DOCKER_COMPOSE exec nodule-segmentation ls -la /app/scripts/ || true
        exit 1
    fi
}

# Function to open interactive shell
open_shell() {
    print_status "Opening interactive shell in container..."
    
    cd "$PROJECT_ROOT"
    
    if ! check_status; then
        print_status "Container is not running. Starting it first..."
        start_container
    fi
    
    print_status "Entering container shell..."
    print_status "Type 'exit' to leave the container"
    $DOCKER_COMPOSE exec nodule-segmentation bash
}

# Function to start Jupyter
start_jupyter() {
    print_status "Starting Jupyter notebook service..."
    
    cd "$PROJECT_ROOT"
    
    if $DOCKER_COMPOSE --profile jupyter up -d; then
        print_success "Jupyter started successfully!"
        echo ""
        echo "Access Jupyter at: http://localhost:8888"
        echo "Token: medical_imaging_2024"
        echo ""
        echo "To stop Jupyter:"
        echo "  $DOCKER_COMPOSE --profile jupyter down"
    else
        print_error "Failed to start Jupyter"
        exit 1
    fi
}

# Function to show logs
show_logs() {
    cd "$PROJECT_ROOT"
    
    if [ "$1" = "-f" ] || [ "$1" = "--follow" ]; then
        print_status "Following container logs (Ctrl+C to stop)..."
        $DOCKER_COMPOSE logs -f nodule-segmentation
    else
        print_status "Showing container logs..."
        $DOCKER_COMPOSE logs nodule-segmentation
    fi
}

# Function to show status
show_status() {
    print_status "Container status:"
    
    cd "$PROJECT_ROOT"
    
    echo ""
    echo "=== Docker Compose Services ==="
    $DOCKER_COMPOSE ps
    
    echo ""
    echo "=== Container Health ==="
    if check_status; then
        print_success "Container is running"
        
        # Show resource usage
        if command -v docker &> /dev/null; then
            echo ""
            echo "=== Resource Usage ==="
            docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | grep "$CONTAINER_NAME" || true
        fi
        
        # Test dependencies
        echo ""
        echo "=== Dependency Check ==="
        if $DOCKER_COMPOSE exec nodule-segmentation python3 -c "
import SimpleITK
import radiomics
import sklearn
import skimage
import scipy
import pandas
import numpy
print('✓ All dependencies available')
" 2>/dev/null; then
            print_success "All dependencies are working"
        else
            print_warning "Some dependencies may not be working"
        fi
    else
        print_warning "Container is not running"
    fi
}

# Function to clean up
cleanup() {
    print_status "Cleaning up containers and volumes..."
    
    cd "$PROJECT_ROOT"
    
    # Stop all services
    $DOCKER_COMPOSE down --remove-orphans
    
    # Remove volumes (optional)
    read -p "Remove data volumes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        $DOCKER_COMPOSE down -v
        print_status "Volumes removed"
    fi
    
    print_success "Cleanup completed!"
}

# Function to run custom command
run_custom() {
    local command="$1"
    
    if [ -z "$command" ]; then
        print_error "No command provided"
        echo "Usage: $0 custom 'your command here'"
        exit 1
    fi
    
    print_status "Running custom command: $command"
    
    cd "$PROJECT_ROOT"
    
    if ! check_status; then
        print_status "Container is not running. Starting it first..."
        start_container
    fi
    
    $DOCKER_COMPOSE exec nodule-segmentation bash -c "$command"
}

# Main function
main() {
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Check Docker Compose availability
    check_docker_compose
    
    local command="$1"
    shift || true
    
    case "$command" in
        "start")
            start_container
            ;;
        "stop")
            stop_container
            ;;
        "restart")
            restart_container
            ;;
        "run")
            run_pipeline
            ;;
        "shell")
            open_shell
            ;;
        "jupyter")
            start_jupyter
            ;;
        "logs")
            show_logs "$@"
            ;;
        "status")
            show_status
            ;;
        "clean")
            cleanup
            ;;
        "custom")
            run_custom "$1"
            ;;
        "-h"|"--help"|"help"|"")
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"