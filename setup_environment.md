# Pose Detection Application Setup Guide

## Environment Setup for "ruk_pos"

This guide will help you set up the pose detection application in the "ruk_pos" environment.

### Prerequisites

- Python 3.11
- Conda or virtual environment manager
- Webcam or camera device
- At least 4GB RAM (8GB recommended for OpenPose)

### Step 1: Create and Activate Environment

```bash
# Create conda environment with Python 3.11
conda create -n ruk_pos python=3.11 -y

# Activate the environment
conda activate ruk_pos
```

### Step 2: Install Dependencies

```bash
# Navigate to project directory
cd /Users/pataweeprakrankamanant/Desktop/ruk_reseach

# Install requirements
pip install -r requirements.txt
```

### Step 3: OpenPose Installation (Optional but Recommended)

The application includes fallback detection, but for best results, install OpenPose:

#### Option A: Using Pre-built Binaries (Recommended)
```bash
# Download OpenPose models (required for pose detection)
mkdir -p models
cd models

# Download COCO model
wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
wget https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt

# Download MPI model (alternative)
wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel
wget https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt
```

#### Option B: Build from Source (Advanced)
```bash
# Clone OpenPose repository
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
cd openpose

# Follow the build instructions in the repository
# This requires CUDA, cuDNN, and other dependencies
```

### Step 4: Test Installation

```bash
# Test Python imports
python -c "import cv2, numpy, flask; print('Basic dependencies OK')"

# Test OpenPose (if installed)
python -c "import pyopenpose; print('OpenPose OK')" || echo "OpenPose not available - using fallback"
```

### Step 5: Run the Application

```bash
# Make sure you're in the ruk_pos environment
conda activate ruk_pos

# Run the application
python pose_detection_app.py
```

The application will start on `http://localhost:5000`

### Step 6: Access the Web Interface

1. Open your web browser
2. Navigate to `http://localhost:5000`
3. Click "Start Stream" to begin pose detection
4. Use "Toggle Detection" to enable/disable pose analysis

## Troubleshooting

### Common Issues

1. **Camera not found**
   - Ensure your webcam is connected and not used by other applications
   - Try changing the camera index in the code (0, 1, 2, etc.)

2. **OpenPose not working**
   - The application will automatically fall back to OpenCV-based detection
   - Check that model files are in the `models/` directory

3. **Performance issues**
   - Reduce video resolution in the code
   - Close other applications using the camera
   - Use a more powerful GPU if available

4. **Import errors**
   - Make sure you're in the correct environment: `conda activate ruk_pos`
   - Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

### Environment Variables

You can set these environment variables to customize the application:

```bash
# Set OpenPose model path
export OPENPOSE_MODEL_PATH="/path/to/openpose/models"

# Set camera device
export CAMERA_DEVICE=0

# Set video resolution
export VIDEO_WIDTH=640
export VIDEO_HEIGHT=480
```

### Performance Optimization

For better performance:

1. **Use GPU acceleration** (if available):
   ```bash
   pip install opencv-python-gpu
   ```

2. **Reduce model complexity**:
   - Edit `pose_detection_app.py`
   - Change `net_resolution` to a smaller value like "160x120"

3. **Limit frame rate**:
   - Modify the FPS setting in the camera configuration

## File Structure

```
ruk_reseach/
├── pose_detection_app.py      # Main application
├── requirements.txt           # Python dependencies
├── setup_environment.md      # This setup guide
├── templates/
│   └── pose_detection.html   # Web interface
├── models/                   # OpenPose model files (created during setup)
└── CPR_application.html      # Reference HTML file
```

## API Endpoints

The application provides these API endpoints:

- `GET /` - Main web interface
- `GET /video_feed` - Video stream with pose detection
- `GET /api/pose_data` - Current pose analysis data
- `POST /api/toggle_detection` - Enable/disable pose detection
- `GET /api/status` - Application status

## Next Steps

1. Test the application with different poses
2. Customize the pose analysis logic for your specific use case
3. Add additional features like pose recording or analysis
4. Integrate with other applications or databases

For more information, refer to the OpenPose documentation: https://github.com/CMU-Perceptual-Computing-Lab/openpose
