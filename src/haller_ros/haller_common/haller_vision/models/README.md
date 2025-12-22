# Vision Models

This directory contains TensorRT engine files for inference on Jetson Orin Nano.

## Required Models

### Object Detection (YOLOv8)

Download and convert YOLOv8 nano model:

```bash
# Install ultralytics
pip install ultralytics

# Export to ONNX
yolo export model=yolov8n.pt format=onnx

# Convert to TensorRT (on Jetson)
/usr/src/tensorrt/bin/trtexec \
  --onnx=yolov8n.onnx \
  --saveEngine=yolov8n.engine \
  --fp16
```

Place `yolov8n.engine` in this directory.

### Semantic Segmentation (SegFormer)

For segmentation, you can use:
1. NVIDIA's pre-trained PeopleSegNet from NGC
2. Custom SegFormer model trained on your environment

```bash
# Download from NGC (requires account)
ngc registry model download-version nvidia/tao/peoplesegnet:deployable_quantized_v2.0
```

## Model Specifications

| Model | Input Size | Format | Expected FPS |
|-------|------------|--------|--------------|
| yolov8n.engine | 640x480 | FP16 | ~30 FPS |
| segformer_b0.engine | 640x480 | FP16 | ~15 FPS |

## Notes

- TensorRT engines are hardware-specific. Generate on the target Jetson device.
- FP16 provides good balance of speed and accuracy on Orin Nano.
- For better performance, consider INT8 quantization with calibration data.

