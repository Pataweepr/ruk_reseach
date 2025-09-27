# 🎯 Pose Detection Application

A real-time pose detection application built with Python 3.11, Flask, and OpenPose technology. This application provides a web-based interface for real-time human pose detection and analysis.

## 🌟 Features

- **Real-time Pose Detection**: Uses OpenPose for accurate human pose estimation
- **Web-based Interface**: Modern, responsive HTML interface
- **Fallback Detection**: Works even without OpenPose using OpenCV-based detection
- **Pose Analysis**: Provides detailed pose metrics and body part tracking
- **Live Video Stream**: Real-time video feed with pose overlay
- **API Endpoints**: RESTful API for integration with other applications

## 🚀 Quick Start

### Prerequisites

- Python 3.11
- Webcam or camera device
- Conda (recommended) or pip

### Installation

1. **Clone or download the project files**

2. **Set up the environment**:
   ```bash
   # Create conda environment
   conda create -n ruk_pos python=3.11 -y
   conda activate ruk_pos
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   # Option 1: Use the launcher script
   ./run_app.sh
   
   # Option 2: Run directly
   python pose_detection_app.py
   ```

4. **Access the web interface**:
   - Open your browser
   - Navigate to `http://localhost:5001`
   - Click "Start Stream" to begin pose detection

## 📁 Project Structure

```
ruk_reseach/
├── pose_detection_app.py      # Main Flask application
├── requirements.txt           # Python dependencies
├── run_app.sh                # Application launcher script
├── setup_environment.md      # Detailed setup instructions
├── README.md                 # This file
├── templates/
│   └── pose_detection.html   # Web interface
├── models/                   # OpenPose model files (created during setup)
└── CPR_application.html      # Reference HTML file
```

## 🔧 Configuration

### Environment Variables

You can customize the application using these environment variables:

```bash
export CAMERA_DEVICE=0          # Camera device index
export VIDEO_WIDTH=640          # Video width
export VIDEO_HEIGHT=480         # Video height
export OPENPOSE_MODEL_PATH="models/"  # Path to OpenPose models
```

### OpenPose Models

For best performance, download OpenPose models:

```bash
mkdir -p models
cd models

# Download COCO model
wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
wget https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt
```

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/video_feed` | GET | Video stream with pose detection |
| `/api/pose_data` | GET | Current pose analysis data |
| `/api/toggle_detection` | POST | Enable/disable pose detection |
| `/api/status` | GET | Application status |

### Example API Usage

```python
import requests

# Get current pose data
response = requests.get('http://localhost:5001/api/pose_data')
pose_data = response.json()

# Toggle detection
response = requests.post('http://localhost:5001/api/toggle_detection')
result = response.json()
```

## 🎮 Usage

### Web Interface

1. **Start Stream**: Begin video capture and pose detection
2. **Toggle Detection**: Enable/disable pose analysis
3. **Stop Stream**: Stop video capture
4. **View Analysis**: Monitor real-time pose data and metrics

### Pose Analysis Features

- **Body Part Tracking**: 18 key body points (COCO format)
- **Confidence Scores**: Detection confidence for each body part
- **Arm Angle Calculation**: Real-time arm angle measurements
- **Pose Status**: Overall pose detection status

## 🔍 Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Ensure webcam is connected and not used by other apps
   - Try different camera indices (0, 1, 2, etc.)

2. **OpenPose not working**:
   - Application automatically falls back to OpenCV detection
   - Check model files are in the `models/` directory

3. **Performance issues**:
   - Reduce video resolution
   - Close other applications
   - Use GPU acceleration if available

4. **Import errors**:
   - Ensure you're in the correct environment: `conda activate ruk_pos`
   - Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### Performance Optimization

- **GPU Acceleration**: Install `opencv-python-gpu` for better performance
- **Model Complexity**: Reduce `net_resolution` in the code for faster processing
- **Frame Rate**: Adjust FPS settings in camera configuration

## 🛠️ Development

### Adding New Features

1. **Custom Pose Analysis**: Modify the `get_pose_analysis()` method
2. **New API Endpoints**: Add routes to the Flask application
3. **UI Enhancements**: Update the HTML template and CSS
4. **Additional Models**: Integrate other pose detection models

### Testing

```bash
# Run basic tests
python -c "import pose_detection_app; print('Import successful')"

# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

## 📚 Dependencies

### Core Dependencies
- `opencv-python==4.8.1.78` - Computer vision library
- `numpy==1.24.3` - Numerical computing
- `flask==2.3.3` - Web framework
- `flask-cors==4.0.0` - CORS support

### Optional Dependencies
- `pyopenpose==1.7.0` - OpenPose Python bindings
- `pillow==10.0.1` - Image processing
- `matplotlib==3.7.2` - Plotting and visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is based on the OpenPose library by CMU-Perceptual-Computing-Lab. Please refer to their license terms for OpenPose usage.

## 🙏 Acknowledgments

- [CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) - Core pose detection technology
- OpenCV community for computer vision tools
- Flask community for web framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the setup guide (`setup_environment.md`)
3. Check system requirements and dependencies
4. Ensure proper environment activation

---

**Happy Pose Detecting! 🎯**
