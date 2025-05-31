"""
MIT License

Copyright (c) 2024 kunalsingh2514@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
"""
Advanced Vehicle Detection System using YOLO

This module provides a unified, robust vehicle detection system that consolidates
and improves upon the existing detection implementations in the TMS project.
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
import threading
from dataclasses import dataclass

from ..utils.config_manager import get_config
from ..utils.logger import get_logger, performance_monitor
from ..utils.error_handler import (
    VehicleDetectionError, ModelLoadingError, CameraConnectionError,
    error_handler, safe_execute
)

@dataclass
class DetectionResult:
    """Data class for vehicle detection results."""
    vehicle_count: int
    detections: List[Dict[str, Any]]
    confidence_scores: List[float]
    processing_time: float
    frame_id: int
    timestamp: float

@dataclass
class BoundingBox:
    """Data class for bounding box information."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    class_id: int
    class_name: str

class VehicleDetector:
    """
    Advanced vehicle detection system using YOLO with improved error handling,
    performance monitoring, and configuration management.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the vehicle detector.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = get_config()
        self.logger = get_logger("VehicleDetector")

        # YOLO configuration
        self.weights_path = self.config.get('models.yolo.weights_path')
        self.config_path = self.config.get('models.yolo.config_path')
        self.classes_path = self.config.get('models.yolo.classes_path')
        self.confidence_threshold = self.config.get('models.yolo.confidence_threshold', 0.5)
        self.nms_threshold = self.config.get('models.yolo.nms_threshold', 0.4)
        self.input_size = tuple(self.config.get('models.yolo.input_size', [416, 416]))

        # Vehicle classes to detect
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorbike', 'bicycle']

        # Model components
        self.net = None
        self.output_layers = None
        self.classes = None
        self.vehicle_class_ids = []

        # Performance tracking
        self.frame_count = 0
        self.total_processing_time = 0.0
        self.detection_history = []

        # Thread safety
        self._lock = threading.Lock()

        # Initialize the model
        self._load_model()

    @error_handler(reraise=True)
    def _load_model(self) -> None:
        """Load YOLO model and initialize components."""
        try:
            self.logger.info("Loading YOLO model...")

            # Validate file paths
            if not Path(self.weights_path).exists():
                raise ModelLoadingError(
                    f"YOLO weights file not found: {self.weights_path}",
                    error_code="WEIGHTS_NOT_FOUND"
                )

            if not Path(self.config_path).exists():
                raise ModelLoadingError(
                    f"YOLO config file not found: {self.config_path}",
                    error_code="CONFIG_NOT_FOUND"
                )

            if not Path(self.classes_path).exists():
                raise ModelLoadingError(
                    f"Classes file not found: {self.classes_path}",
                    error_code="CLASSES_NOT_FOUND"
                )

            # Load YOLO network
            self.net = cv2.dnn.readNet(self.weights_path, self.config_path)

            # Get output layer names
            layer_names = self.net.getLayerNames()
            unconnected_out_layers = self.net.getUnconnectedOutLayers()

            # Handle different OpenCV versions
            if len(unconnected_out_layers.shape) == 1:
                self.output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
            else:
                self.output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]

            # Load class names
            with open(self.classes_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]

            # Find vehicle class IDs
            self.vehicle_class_ids = [
                i for i, class_name in enumerate(self.classes)
                if class_name.lower() in self.vehicle_classes
            ]

            self.logger.info(f"YOLO model loaded successfully. Vehicle classes: {self.vehicle_classes}")
            self.logger.info(f"Vehicle class IDs: {self.vehicle_class_ids}")

        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise ModelLoadingError(f"Model loading failed: {e}")

    @performance_monitor("VehicleDetector")
    def detect_vehicles(self, frame: np.ndarray, frame_id: int = 0) -> DetectionResult:
        """
        Detect vehicles in a single frame.

        Args:
            frame: Input frame as numpy array
            frame_id: Frame identifier for tracking

        Returns:
            DetectionResult object containing detection information
        """
        start_time = time.time()

        try:
            with self._lock:
                # Validate input
                if frame is None or frame.size == 0:
                    raise VehicleDetectionError("Invalid input frame")

                height, width = frame.shape[:2]

                # Prepare input blob
                blob = cv2.dnn.blobFromImage(
                    frame,
                    scalefactor=1/255.0,
                    size=self.input_size,
                    mean=(0, 0, 0),
                    swapRB=True,
                    crop=False
                )

                # Set input to the network
                self.net.setInput(blob)

                # Run forward pass
                detections = self.net.forward(self.output_layers)

                # Process detections
                boxes, confidences, class_ids = self._process_detections(
                    detections, width, height
                )

                # Apply Non-Maximum Suppression
                indices = cv2.dnn.NMSBoxes(
                    boxes, confidences,
                    self.confidence_threshold,
                    self.nms_threshold
                )

                # Extract final detections
                final_detections = self._extract_final_detections(
                    boxes, confidences, class_ids, indices
                )

                processing_time = time.time() - start_time

                # Update statistics
                self.frame_count += 1
                self.total_processing_time += processing_time

                # Create result
                result = DetectionResult(
                    vehicle_count=len(final_detections),
                    detections=final_detections,
                    confidence_scores=[d['confidence'] for d in final_detections],
                    processing_time=processing_time,
                    frame_id=frame_id,
                    timestamp=time.time()
                )

                # Log detection results
                self.logger.log_vehicle_detection(
                    frame_id, result.vehicle_count,
                    processing_time, result.confidence_scores
                )

                return result

        except Exception as e:
            self.logger.error(f"Vehicle detection failed: {e}")
            raise VehicleDetectionError(f"Detection failed: {e}")

    def _process_detections(self, detections: List[np.ndarray],
                          width: int, height: int) -> Tuple[List, List, List]:
        """Process raw YOLO detections."""
        boxes = []
        confidences = []
        class_ids = []

        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Filter by confidence and vehicle classes
                if (confidence > self.confidence_threshold and
                    class_id in self.vehicle_class_ids):

                    # Extract bounding box
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids

    def _extract_final_detections(self, boxes: List, confidences: List,
                                class_ids: List, indices: np.ndarray) -> List[Dict[str, Any]]:
        """Extract final detections after NMS."""
        final_detections = []

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]
                class_name = self.classes[class_id]

                detection = {
                    'bbox': BoundingBox(x, y, w, h, confidence, class_id, class_name),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'center_x': x + w // 2,
                    'center_y': y + h // 2
                }

                final_detections.append(detection)

        return final_detections

    def detect_vehicles_batch(self, frames: List[np.ndarray]) -> List[DetectionResult]:
        """
        Detect vehicles in multiple frames.

        Args:
            frames: List of input frames

        Returns:
            List of DetectionResult objects
        """
        results = []

        for i, frame in enumerate(frames):
            try:
                result = self.detect_vehicles(frame, frame_id=i)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process frame {i}: {e}")
                # Create empty result for failed frame
                results.append(DetectionResult(
                    vehicle_count=0,
                    detections=[],
                    confidence_scores=[],
                    processing_time=0.0,
                    frame_id=i,
                    timestamp=time.time()
                ))

        return results

    def draw_detections(self, frame: np.ndarray,
                       detection_result: DetectionResult) -> np.ndarray:
        """
        Draw detection results on frame.

        Args:
            frame: Input frame
            detection_result: Detection results to draw

        Returns:
            Frame with drawn detections
        """
        output_frame = frame.copy()

        for detection in detection_result.detections:
            bbox = detection['bbox']

            # Draw bounding box
            cv2.rectangle(
                output_frame,
                (bbox.x, bbox.y),
                (bbox.x + bbox.width, bbox.y + bbox.height),
                (0, 255, 0), 2
            )

            # Draw label
            label = f"{bbox.class_name}: {bbox.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            cv2.rectangle(
                output_frame,
                (bbox.x, bbox.y - label_size[1] - 10),
                (bbox.x + label_size[0], bbox.y),
                (0, 255, 0), -1
            )

            cv2.putText(
                output_frame, label,
                (bbox.x, bbox.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 2
            )

        # Draw summary information
        summary = f"Vehicles: {detection_result.vehicle_count} | Time: {detection_result.processing_time:.3f}s"
        cv2.putText(
            output_frame, summary,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2
        )

        return output_frame

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self.frame_count == 0:
            return {
                'frames_processed': 0,
                'average_fps': 0.0,
                'average_processing_time': 0.0,
                'total_processing_time': 0.0
            }

        avg_processing_time = self.total_processing_time / self.frame_count
        avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0

        return {
            'frames_processed': self.frame_count,
            'average_fps': avg_fps,
            'average_processing_time': avg_processing_time,
            'total_processing_time': self.total_processing_time
        }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        with self._lock:
            self.frame_count = 0
            self.total_processing_time = 0.0
            self.detection_history.clear()

        self.logger.info("Performance statistics reset")

    def is_model_loaded(self) -> bool:
        """Check if model is properly loaded."""
        return (self.net is not None and
                self.output_layers is not None and
                self.classes is not None)

    def cleanup(self) -> None:
        """Cleanup resources."""
        with self._lock:
            self.net = None
            self.output_layers = None
            self.classes = None
            self.detection_history.clear()

        self.logger.info("VehicleDetector resources cleaned up")
