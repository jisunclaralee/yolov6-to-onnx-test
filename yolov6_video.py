import os
import sys
import subprocess
import json
import time
from pathlib import Path

def install_system_dependencies():
    """Install system dependencies for OpenCV"""
    print("Installing system dependencies...")
    try:
        # Quick check if already installed
        result = subprocess.run(["which", "ffmpeg"], capture_output=True)
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
            "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "wget"
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
        "tqdm"
    ]
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--root-user-action=ignore", package], 
                         check=True, capture_output=True)
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")

# Install dependencies first
install_system_dependencies()
install_packages()

# Now import the packages
import numpy as np
import cv2
import onnxruntime as ort
from tqdm import tqdm

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

# Color palette for different classes (BGR format)
CLASS_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (192, 192, 192), (128, 128, 128), (255, 165, 0),
    (255, 20, 147), (0, 191, 255), (255, 105, 180), (64, 224, 208), (255, 215, 0)
]

def get_class_color(class_id):
    """Get color for a specific class"""
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]

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

def preprocess_frame(frame):
    """Preprocess frame for YOLOv6 inference"""
    # Resize to model input size
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize and reshape
    blob = (img / 255.0).transpose(2, 0, 1).astype(np.float32)
    blob = np.expand_dims(blob, 0)
    
    return blob

def postprocess_outputs(outputs, orig_w, orig_h, conf_thres=0.25):
    """Post-process model outputs to get detections"""
    # Extract predictions
    boxes = outputs[0, :, :4]
    conf = outputs[0, :, 4]
    cls_scores = outputs[0, :, 5:]
    cls_ids = np.argmax(cls_scores, axis=1)
    scores = conf * cls_scores[np.arange(len(cls_ids)), cls_ids]
    
    # Filter by confidence threshold
    mask = scores > conf_thres
    boxes, scores, cls_ids = boxes[mask], scores[mask], cls_ids[mask]
    
    detections = []
    if len(boxes) > 0:
        # Apply NMS
        keep = nms(boxes, scores)
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
        
        # Scale boxes to original image size
        boxes[:, 0] *= orig_w / 640  # x center
        boxes[:, 1] *= orig_h / 640  # y center
        boxes[:, 2] *= orig_w / 640  # width
        boxes[:, 3] *= orig_h / 640  # height
        
        # Convert to detection format
        for box, score, cls_id in zip(boxes, scores, cls_ids):
            x, y, w, h = box
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'class_id': int(cls_id),
                'class_name': COCO_CLASSES[cls_id],
                'confidence': float(score)
            })
    
    return detections

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame"""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        class_id = det['class_id']
        class_name = det['class_name']
        confidence = det['confidence']
        
        # Get color for this class
        color = get_class_color(class_id)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label = f"{class_name}: {confidence:.2f}"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw label background
        cv2.rectangle(
            frame, 
            (x1, y1 - text_height - baseline - 5), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255),  # White text
            2
        )
    
    return frame

def process_video(onnx_path, input_path, output_path, conf_thres=0.25):
    """Process a single video file with YOLOv6 object detection"""
    print(f"Processing: {input_path}")
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Statistics
    total_detections = 0
    frame_detections = []
    
    # Process frames with progress bar
    pbar = tqdm(total=total_frames, desc="Processing frames")
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Preprocess frame
            blob = preprocess_frame(frame)
            
            # Run inference
            outputs = session.run(None, {session.get_inputs()[0].name: blob})[0]
            
            # Post-process outputs
            detections = postprocess_outputs(outputs, width, height, conf_thres)
            
            # Draw detections on frame
            annotated_frame = draw_detections(frame.copy(), detections)
            
            # Write frame to output video
            out.write(annotated_frame)
            
            # Update statistics
            total_detections += len(detections)
            frame_detections.append({
                'frame': frame_count,
                'detections': len(detections),
                'objects': [det['class_name'] for det in detections]
            })
            
            # Update progress bar
            pbar.update(1)
            
    finally:
        cap.release()
        out.release()
        pbar.close()
    
    print(f"✓ Processed {frame_count} frames")
    print(f"✓ Total detections: {total_detections}")
    print(f"✓ Average detections per frame: {total_detections/frame_count:.2f}")
    
    return {
        'input_file': input_path,
        'output_file': output_path,
        'frames_processed': frame_count,
        'total_detections': total_detections,
        'frame_detections': frame_detections
    }

def get_video_files(input_dir):
    """Get all video files from input directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    video_files = []
    
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return video_files
    
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(input_dir, file))
    
    return video_files

def process_all_videos(onnx_path, input_dir, output_dir, conf_thres=0.25):
    """Process all videos in input directory"""
    print(f"Looking for videos in: {input_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video files
    video_files = get_video_files(input_dir)
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return []
    
    print(f"Found {len(video_files)} video files")
    
    all_results = []
    
    for i, input_path in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing video {i}/{len(video_files)}")
        print(f"{'='*60}")
        
        # Generate output filename
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_detected{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            # Process video
            start_time = time.time()
            result = process_video(onnx_path, input_path, output_path, conf_thres)
            end_time = time.time()
            
            result['processing_time'] = end_time - start_time
            all_results.append(result)
            
            print(f"✓ Processing completed in {end_time - start_time:.2f} seconds")
            print(f"✓ Output saved to: {output_path}")
            
        except Exception as e:
            print(f"✗ Error processing {input_path}: {e}")
            continue
    
    return all_results

def save_processing_summary(results, output_dir):
    """Save processing summary to JSON file"""
    summary_file = os.path.join(output_dir, "processing_summary.json")
    
    summary = {
        "total_videos_processed": len(results),
        "total_detections": sum(r['total_detections'] for r in results),
        "total_processing_time": sum(r['processing_time'] for r in results),
        "results": results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Processing summary saved to: {summary_file}")

if __name__ == "__main__":
    # Configuration
    ONNX_MODEL_PATH = "/jisunlee/yolov6n.onnx"
    INPUT_DIR = "/jisunlee/input_videos"
    OUTPUT_DIR = "/jisunlee/output_videos"
    CONFIDENCE_THRESHOLD = 0.25
    
    try:
        print("🎬 YOLOv6 Video Object Detection")
        print("=" * 50)
        
        # Check if ONNX model exists
        if not os.path.exists(ONNX_MODEL_PATH):
            print(f"❌ ONNX model not found: {ONNX_MODEL_PATH}")
            print("Please run yolov6_image.py first to generate the ONNX model")
            sys.exit(1)
        
        # Check if input directory exists
        if not os.path.exists(INPUT_DIR):
            print(f"📁 Creating input directory: {INPUT_DIR}")
            os.makedirs(INPUT_DIR)
            print(f"Please place video files in {INPUT_DIR} and run again")
            sys.exit(0)
        
        print(f"📂 Input directory: {INPUT_DIR}")
        print(f"📂 Output directory: {OUTPUT_DIR}")
        print(f"🎯 ONNX model: {ONNX_MODEL_PATH}")
        print(f"🎚️ Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print()
        
        # Process all videos
        results = process_all_videos(
            ONNX_MODEL_PATH, 
            INPUT_DIR, 
            OUTPUT_DIR, 
            CONFIDENCE_THRESHOLD
        )
        
        if results:
            # Save processing summary
            save_processing_summary(results, OUTPUT_DIR)
            
            print(f"\n🎉 Processing completed!")
            print(f"📊 Processed {len(results)} videos")
            print(f"🎯 Total detections: {sum(r['total_detections'] for r in results)}")
            print(f"⏱️ Total time: {sum(r['processing_time'] for r in results):.2f} seconds")
            print(f"📁 Results saved in: {OUTPUT_DIR}")
        else:
            print("❌ No videos were processed successfully")
            
    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
