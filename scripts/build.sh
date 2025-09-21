#!/bin/bash

# Medical Imaging Nodule Segmentation Pipeline - Build Script
# Automated Docker container build and setup

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
IMAGE_NAME="medical-imaging/nodule-segmentation"
IMAGE_TAG="latest"
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

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null; then
        print_warning "docker-compose not found. Checking for 'docker compose'..."
        if ! docker compose version &> /dev/null; then
            print_error "Docker Compose is not available. Please install Docker Compose."
            exit 1
        else
            DOCKER_COMPOSE="docker compose"
        fi
    else
        DOCKER_COMPOSE="docker-compose"
    fi
    
    print_success "Prerequisites check passed!"
}

# Function to create necessary directories
setup_directories() {
    print_status "Setting up directory structure..."
    
    # Create directories if they don't exist
    directories=(
        "docker/logs"
        "docker/notebooks" 
        "src"
        "scripts"
        "params"
        "data"
        "output"
        "logs"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$PROJECT_ROOT/$dir" ]; then
            mkdir -p "$PROJECT_ROOT/$dir"
            print_status "Created directory: $dir"
        fi
    done
    
    print_success "Directory structure ready!"
}

# Function to copy source files
setup_source_files() {
    print_status "Setting up source files..."
    
    # Copy source files to appropriate directories
    if [ -f "$PROJECT_ROOT/scr/candidateSeg_pipiline.py" ]; then
        cp "$PROJECT_ROOT/scr/candidateSeg_pipiline.py" "$PROJECT_ROOT/src/"
        print_status "Copied main pipeline script"
    fi
    
    if [ -f "$PROJECT_ROOT/scr/cvseg_utils.py" ]; then
        cp "$PROJECT_ROOT/scr/cvseg_utils.py" "$PROJECT_ROOT/src/"
        print_status "Copied utility scripts"
    fi
    
    if [ -f "$PROJECT_ROOT/DLCS24_KNN_2mm_Extend_Seg.sh" ]; then
        cp "$PROJECT_ROOT/DLCS24_KNN_2mm_Extend_Seg.sh" "$PROJECT_ROOT/scripts/"
        chmod +x "$PROJECT_ROOT/scripts/DLCS24_KNN_2mm_Extend_Seg.sh"
        print_status "Copied execution scripts"
    fi
    
    if [ -f "$PROJECT_ROOT/scr/Pyradiomics_feature_extarctor_pram.json" ]; then
        cp "$PROJECT_ROOT/scr/Pyradiomics_feature_extarctor_pram.json" "$PROJECT_ROOT/params/"
        print_status "Copied parameter files"
    fi
    
    print_success "Source files ready!"
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image: $IMAGE_NAME:$IMAGE_TAG"
    
    cd "$PROJECT_ROOT"
    
    # Build with docker-compose
    if $DOCKER_COMPOSE build --no-cache; then
        print_success "Docker image built successfully!"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to verify the build
verify_build() {
    print_status "Verifying Docker image..."
    
    # Check if image exists
    if docker images | grep -q "$IMAGE_NAME"; then
        print_success "Docker image verified!"
        
        # Show image details
        print_status "Image details:"
        docker images | grep "$IMAGE_NAME" | head -1
        
        # Test basic functionality
        print_status "Testing basic functionality..."
        if docker run --rm "$IMAGE_NAME:$IMAGE_TAG" python3 -c "import SimpleITK, radiomics, sklearn, skimage, scipy, pandas, numpy; print('All dependencies available!')"; then
            print_success "All dependencies are working correctly!"
        else
            print_warning "Some dependencies may not be working correctly"
        fi
    else
        print_error "Docker image not found after build"
        exit 1
    fi
}

# Function to show usage instructions
show_usage() {
    print_status "Build complete! Here's how to use the container:"
    echo ""
    echo "1. Start the container:"
    echo "   $DOCKER_COMPOSE up -d nodule-segmentation"
    echo ""
    echo "2. Run the segmentation pipeline:"
    echo "   $DOCKER_COMPOSE exec nodule-segmentation bash /app/scripts/DLCS24_KNN_2mm_Extend_Seg.sh"
    echo ""
    echo "3. Run interactively:"
    echo "   $DOCKER_COMPOSE exec nodule-segmentation bash"
    echo ""
    echo "4. Start Jupyter (optional):"
    echo "   $DOCKER_COMPOSE --profile jupyter up -d"
    echo "   # Access at http://localhost:8888 (token: medical_imaging_2024)"
    echo ""
    echo "5. View logs:"
    echo "   $DOCKER_COMPOSE logs -f nodule-segmentation"
    echo ""
    echo "6. Stop the container:"
    echo "   $DOCKER_COMPOSE down"
    echo ""
}

# Function to clean up previous builds
cleanup() {
    print_status "Cleaning up previous builds..."
    
    # Stop and remove containers
    $DOCKER_COMPOSE down --remove-orphans 2>/dev/null || true
    
    # Remove previous images (optional)
    if [ "$1" = "--clean" ]; then
        docker rmi "$IMAGE_NAME:$IMAGE_TAG" 2>/dev/null || true
        print_status "Removed previous image"
    fi
}

# Main build process
main() {
    echo "========================================"
    echo "Medical Imaging Pipeline - Build Script"
    echo "========================================"
    echo ""
    
    # Parse command line arguments
    CLEAN_BUILD=false
    if [ "$1" = "--clean" ]; then
        CLEAN_BUILD=true
        print_status "Clean build requested"
    fi
    
    # Execute build steps
    check_prerequisites
    
    if [ "$CLEAN_BUILD" = true ]; then
        cleanup --clean
    fi
    
    setup_directories
    setup_source_files
    build_image
    verify_build
    show_usage
    
    print_success "Build completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Review the README.md for detailed usage instructions"
    echo "2. Prepare your input data in the expected format"
    echo "3. Start the container and run your analysis"
    echo ""
}

# Run main function with all arguments
main "$@"