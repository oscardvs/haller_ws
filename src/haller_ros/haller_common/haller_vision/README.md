# Haller Vision

Vision pipeline for the Haller robot, providing camera input, object detection, semantic segmentation, and traversability analysis.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Vision Pipeline                            │
│                                                                 │
│  ┌──────────┐    ┌───────────┐    ┌──────────────┐             │
│  │  Camera  │───▶│ Detection │───▶│ Traversability│──▶ Costmap │
│  │  (gscam) │    │  (YOLO)   │    │   Analysis    │             │
│  └──────────┘    └───────────┘    └──────────────┘             │
│       │                                   ▲                     │
│       │          ┌──────────────┐         │                     │
│       └─────────▶│ Segmentation │─────────┘                     │
│                  │ (SegFormer)  │                               │
│                  └──────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

## Hardware Requirements

- **Camera**: IMX219 (8MP, compatible with Raspberry Pi Camera v2)
- **Connection**: CSI connector on Jetson Orin Nano
- **Compute**: NVIDIA Jetson Orin Nano (8GB recommended)

## Camera Setup

### 1. Configure Camera in Jetson

```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
# Select: Configure Jetson Nano CSI Connector
# Select: Camera IMX219-A
# Save and reboot
```

### 2. Verify Camera Detection

```bash
# Check video device exists
ls /dev/video*

# Verify camera model
v4l2-ctl --list-devices

# Test capture
v4l2-ctl -d /dev/video0 --stream-mmap --stream-count=1 --stream-to=/tmp/test.raw
```

### 3. Install Dependencies

```bash
# GStreamer camera driver for ROS 2
sudo apt install ros-humble-gscam gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
```

## Usage

### Launch Camera Only

```bash
# Hardware camera (Jetson)
ros2 launch haller_vision camera.launch.py use_sim:=false

# Simulation (Gazebo provides camera)
ros2 launch haller_vision camera.launch.py use_sim:=true
```

### Launch Full Vision Pipeline

```bash
# Full pipeline on hardware
ros2 launch haller_vision vision_pipeline.launch.py use_sim:=false

# Full pipeline in simulation
ros2 launch haller_vision vision_pipeline.launch.py use_sim:=true

# Disable specific components
ros2 launch haller_vision vision_pipeline.launch.py \
    enable_detection:=false \
    enable_segmentation:=true \
    enable_traversability:=true
```

### Launch Individual Nodes

```bash
# Detection only
ros2 launch haller_vision detection.launch.py

# Segmentation only
ros2 launch haller_vision segmentation.launch.py
```

## Topics

### Published

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Raw camera image (1280x720 RGB) |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | Camera calibration data |
| `/detection/objects` | `vision_msgs/Detection2DArray` | Detected objects with bounding boxes |
| `/segmentation/mask` | `sensor_msgs/Image` | Semantic segmentation mask |
| `/traversability/costmap` | `nav_msgs/OccupancyGrid` | Traversability costmap for Nav2 |

### Subscribed

| Topic | Type | Node |
|-------|------|------|
| `/camera/image_raw` | `sensor_msgs/Image` | detection_node, segmentation_node |

## Configuration

### Camera (`config/camera/imx219_hardware.yaml`)

```yaml
gscam_node:
  ros__parameters:
    gscam_config: "v4l2src device=/dev/video0 ! video/x-bayer,format=rggb,width=1280,height=720,framerate=30/1 ! bayer2rgb ! videoconvert ! video/x-raw,format=RGB ! appsink"
    frame_id: "camera_optical_frame"
    image_width: 1280
    image_height: 720
```

### Detection (`config/detection/yolov8_config.yaml`)

Configure YOLO model path, confidence threshold, and target classes.

### Segmentation (`config/segmentation/segformer_config.yaml`)

Configure SegFormer model and class mappings.

### Traversability (`config/traversability/traversability.yaml`)

Configure costmap generation parameters and obstacle costs.

## Camera Calibration

For accurate 3D projection, calibrate the camera:

```bash
# Print a checkerboard pattern (8x6, 25mm squares)
# Then run calibration
ros2 run camera_calibration cameracalibrator \
    --size 8x6 \
    --square 0.025 \
    image:=/camera/image_raw \
    camera:=/camera

# Save calibration to file and update camera_info_url in config
```

## Models

Place trained models in the `models/` directory:

```
models/
├── yolov8n.onnx          # Object detection model
├── segformer_b0.onnx     # Segmentation model
└── README.md             # Model documentation
```

See `models/README.md` for model download and conversion instructions.

## Simulation vs Hardware

The vision pipeline automatically switches between hardware and simulation:

| Mode | Camera Source | Launch Argument |
|------|--------------|-----------------|
| Hardware | IMX219 via gscam | `use_sim:=false` |
| Simulation | Gazebo camera plugin | `use_sim:=true` |

In simulation, the Gazebo plugin (defined in `haller_description/urdf/gazebo_plugins.xacro`) provides camera images on the same topics.

## Troubleshooting

### Camera not detected

```bash
# Check if camera is recognized
v4l2-ctl --list-devices

# If not, verify CSI connection and run jetson-io.py again
sudo /opt/nvidia/jetson-io/jetson-io.py
```

### GStreamer pipeline errors

```bash
# Test pipeline manually
gst-launch-1.0 v4l2src device=/dev/video0 ! \
    'video/x-bayer,format=rggb,width=1280,height=720' ! \
    bayer2rgb ! videoconvert ! autovideosink
```

### No image published

```bash
# Check if topic exists
ros2 topic list | grep camera

# Check publish rate
ros2 topic hz /camera/image_raw
```

## License

Apache-2.0

