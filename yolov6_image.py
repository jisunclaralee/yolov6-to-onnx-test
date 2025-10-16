import os
import sys
import subprocess
import urllib.request
import json

def install_system_dependencies():
    """Install system dependencies for OpenCV"""
    print("Installing system dependencies...")
    try:
        # Quick check if already installed
        result = subprocess.run(["which", "wget"], capture_output=True)
        if result.returncode == 0:
            print("✓ System dependencies already available")
            return
            
        print("Updating package list...")
        subprocess.run([
            "apt", "update", "-qq"
        ], check=True, capture_output=True)
        
        print("Installing essential packages...")
        subprocess.run([
            "apt", "install", "-y", "-qq",
            "libgl1-mesa-glx", "libglib2.0-0", "wget", "git"
        ], check=True, capture_output=True)
        print("✓ System dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install system dependencies: {e}")
        print("Continuing with headless OpenCV...")

def install_packages():
    """Install required packages"""
    # First uninstall any existing opencv packages to avoid conflicts
    print("Removing any existing OpenCV packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"], 
                     capture_output=True)
    except:
        pass
    
    packages = [
        "numpy==1.26.4",
        "onnxruntime==1.17.3", 
        "onnx==1.16.0",
        "opencv-python-headless==4.9.0.80",
        "torch",
        "torchvision", 
        "matplotlib",
        "requests",
        "pillow"
    ]
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--root-user-action=ignore", package], 
                         check=True, capture_output=True)
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")

# Install system dependencies and packages first
install_system_dependencies()
install_packages()

# Now import the packages
import re
import numpy as np
import cv2
import onnxruntime as ort
import requests
from PIL import Image
import matplotlib.pyplot as plt

# COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def download_coco_images():
    """Download sample COCO images for testing"""
    print("Downloading COCO sample images...")
    
    # COCO sample image URLs
    image_urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # cats
        "http://images.cocodataset.org/val2017/000000397133.jpg",  # person with dog
        "http://images.cocodataset.org/val2017/000000037777.jpg",  # stop sign
        "http://images.cocodataset.org/val2017/000000252219.jpg",  # kitchen
        "http://images.cocodataset.org/val2017/000000087038.jpg",  # pizza
    ]
    
    image_paths = []
    os.makedirs("coco_images", exist_ok=True)
    
    for i, url in enumerate(image_urls):
        filename = f"coco_images/sample_{i+1}.jpg"
        if not os.path.exists(filename):
            try:
                print(f"Downloading image {i+1}/5...")
                urllib.request.urlretrieve(url, filename)
                print(f"✓ Downloaded: {filename}")
                image_paths.append(filename)
            except Exception as e:
                print(f"✗ Failed to download {url}: {e}")
        else:
            print(f"✓ Already exists: {filename}")
            image_paths.append(filename)
    
    return image_paths

def setup_yolov6():
    """Clone and setup YOLOv6 repository"""
    if not os.path.exists("YOLOv6"):
        print("Cloning YOLOv6 repository...")
        subprocess.run(["git", "clone", "https://github.com/meituan/YOLOv6.git"], check=True)
    
    os.chdir("YOLOv6")
    subprocess.run(["git", "fetch", "--tags"], check=True)
    subprocess.run(["git", "checkout", "-f", "0.2.0"], check=True)
    print("Checked out to version 0.2.0")
    
    # Add to path
    sys.path.insert(0, os.getcwd())

def patch_pytorch_compatibility():
    """Patch for PyTorch >= 2.6 compatibility"""
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    if tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2, 6):
        print("Applying safe-load patch for PyTorch >= 2.6...")
        path = "yolov6/utils/checkpoint.py"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            content = re.sub(
                r"torch\.load\(\s*weights\s*,\s*map_location\s*=\s*map_location\s*\)",
                r"torch.load(weights, map_location=map_location, weights_only=False)",
                content
            )
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Patched: {path}")

def download_weights():
    """Download YOLOv6n weights"""
    weights_url = "https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6n.pt"
    weights_path = "../yolov6n.pt"
    
    if not os.path.exists(weights_path):
        print("Downloading YOLOv6n weights...")
        subprocess.run(["wget", weights_url, "-O", weights_path], check=True)
        print(f"Downloaded weights to {weights_path}")

def convert_to_onnx():
    """Convert PyTorch model to ONNX"""
    print("Converting to ONNX...")
    subprocess.run([
        sys.executable, "deploy/ONNX/export_onnx.py",
        "--weights", "../yolov6n.pt",
        "--img-size", "640",
        "--batch-size", "1",
        "--simplify"
    ], check=True)
    print("ONNX conversion completed!")

def nms(boxes, scores, iou_thres=0.45):
    """Non-Maximum Suppression"""
    if len(boxes) == 0:
        return np.array([])
        
    keep = []
    order = scores.argsort()[::-1]
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        
        iou = box_iou(boxes[i], boxes[order[1:]])
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    
    return np.array(keep)

def box_iou(box1, boxes):
    """Calculate IoU between box1 and boxes"""
    if len(boxes) == 0:
        return np.array([])
        
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = boxes.T
    
    x1_max, x1_min = x1 + w1/2, x1 - w1/2
    y1_max, y1_min = y1 + h1/2, y1 - h1/2
    x2_max, x2_min = x2 + w2/2, x2 - w2/2
    y2_max, y2_min = y2 + h2/2, y2 - h2/2
    
    inter_x_min = np.maximum(x1_min, x2_min)
    inter_y_min = np.maximum(y1_min, y2_min)
    inter_x_max = np.minimum(x1_max, x2_max)
    inter_y_max = np.minimum(y1_max, y2_max)
    
    inter_area = np.maximum(0, inter_x_max - inter_x_min) * np.maximum(0, inter_y_max - inter_y_min)
    union_area = w1 * h1 + w2 * h2 - inter_area
    
    return inter_area / (union_area + 1e-6)

def process_image_with_yolo(onnx_path, image_path, output_path, conf_thres=0.25):
    """Process single image with YOLOv6 ONNX model"""
    print(f"Processing image: {image_path}")
    
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Load and preprocess image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    orig_h, orig_w = frame.shape[:2]
    print(f"Original image size: {orig_w}x{orig_h}")
    
    # Preprocess
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blob = (img / 255.0).transpose(2, 0, 1).astype(np.float32)
    blob = np.expand_dims(blob, 0)
    
    # Inference
    outputs = session.run(None, {session.get_inputs()[0].name: blob})[0]
    
    # Post-process
    boxes = outputs[0, :, :4]
    conf = outputs[0, :, 4]
    cls_scores = outputs[0, :, 5:]
    cls_ids = np.argmax(cls_scores, axis=1)
    scores = conf * cls_scores[np.arange(len(cls_ids)), cls_ids]
    
    mask = scores > conf_thres
    boxes, scores, cls_ids = boxes[mask], scores[mask], cls_ids[mask]
    
    detections = []
    if len(boxes) > 0:
        keep = nms(boxes, scores)
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
        
        # Scale to original size
        boxes[:, 0] *= orig_w / 640
        boxes[:, 1] *= orig_h / 640
        boxes[:, 2] *= orig_w / 640
        boxes[:, 3] *= orig_h / 640
        
        # Draw boxes and collect detection info
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            x, y, w, h = box
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            
            class_name = COCO_CLASSES[cls_id]
            confidence = float(score)
            
            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })
            
            label = f"{class_name}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Save result
    cv2.imwrite(output_path, frame)
    print(f"✓ Result saved to: {output_path}")
    print(f"✓ Detected {len(detections)} objects:")
    for det in detections:
        print(f"  - {det['class']}: {det['confidence']:.3f}")
    
    return detections

def test_onnx_model(onnx_path):
    """Test ONNX model with random input"""
    print("Testing ONNX model...")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Create random test input
    test_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
    outputs = session.run(None, {session.get_inputs()[0].name: test_input})
    
    print(f"Model output shape: {outputs[0].shape}")
    print("ONNX model test successful!")

def create_detection_summary(results, output_file):
    """Create a summary of all detections"""
    print(f"Creating detection summary: {output_file}")
    
    summary = {
        "total_images": len(results),
        "total_detections": sum(len(detections) for detections in results.values()),
        "results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to: {output_file}")

if __name__ == "__main__":
    try:
        # Step 1: Download COCO images
        image_paths = download_coco_images()
        
        # Step 2: Setup YOLOv6
        setup_yolov6()
        
        # Step 3: Patch PyTorch compatibility
        patch_pytorch_compatibility()
        
        # Step 4: Download weights
        download_weights()
        
        # Step 5: Convert to ONNX
        convert_to_onnx()
        
        # Step 6: Test and process images
        os.chdir("..")  # Return to parent directory
        onnx_path = "yolov6n.onnx"
        
        if os.path.exists(onnx_path):
            # Test ONNX model first
            test_onnx_model(onnx_path)
            
            # Create output directory
            os.makedirs("detection_results", exist_ok=True)
            
            # Process each COCO image
            all_results = {}
            for i, image_path in enumerate(image_paths):
                if os.path.exists(image_path):
                    output_path = f"detection_results/result_{i+1}.jpg"
                    detections = process_image_with_yolo(onnx_path, image_path, output_path)
                    all_results[image_path] = detections
                    print("-" * 50)
            
            # Create detection summary
            create_detection_summary(all_results, "detection_results/summary.json")
            
            print("\n🎉 All processing completed!")
            print(f"📁 Results saved in: detection_results/")
            print(f"📊 Summary: detection_results/summary.json")
            
        else:
            print(f"❌ ONNX file not found: {onnx_path}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
