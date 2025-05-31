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
Multi-Camera Coordination System - Phase 2D

This module provides comprehensive multi-camera coordination for intersection-wide
traffic monitoring with synchronized detection, data fusion, and unified results.

Phase 2D Features:
- Multiple camera sources per intersection
- Camera synchronization and timing coordination
- Overlapping coverage detection and handling
- Unified detection result aggregation
- Real-time data fusion from multiple viewpoints
- GPU acceleration support for multiple streams
- Performance optimization for sub-200ms latency
"""

import cv2
import numpy as np
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue

from .modern_vehicle_detector import ModernVehicleDetector, DetectionResult, BoundingBox, CameraInfo
from ..models.lstm_model import VehicleDetectionData
from ..utils.config_manager import get_config
from ..utils.logger import get_logger, performance_monitor
from ..utils.error_handler import CameraConnectionError, VehicleDetectionError, error_handler

@dataclass
class CameraConfiguration:
    """Configuration for a single camera in multi-camera setup."""
    camera_id: str
    camera_index: int
    intersection_id: str
    position: str  # 'north', 'south', 'east', 'west', 'overhead', 'corner'
    resolution: Tuple[int, int]
    fps: float
    roi: Optional[Tuple[int, int, int, int]] = None  # Region of Interest (x, y, w, h)
    transform_matrix: Optional[np.ndarray] = None  # For perspective correction
    overlap_zones: List[str] = field(default_factory=list)  # Other cameras with overlapping coverage
    priority: int = 1  # 1=highest, 5=lowest for conflict resolution

@dataclass
class SynchronizedFrame:
    """Synchronized frame data from multiple cameras."""
    timestamp: float
    intersection_id: str
    frames: Dict[str, np.ndarray]  # camera_id -> frame
    frame_ids: Dict[str, int]  # camera_id -> frame_id
    sync_quality: float  # 0.0-1.0, quality of synchronization

@dataclass
class FusedDetectionResult:
    """Fused detection result from multiple cameras."""
    intersection_id: str
    timestamp: float
    total_vehicle_count: int
    camera_results: Dict[str, DetectionResult]  # camera_id -> individual results
    fused_detections: List[Dict[str, Any]]  # Deduplicated and fused detections
    confidence_scores: List[float]
    processing_time: float
    sync_quality: float
    coverage_completeness: float  # Percentage of intersection covered

@dataclass
class CameraPerformanceMetrics:
    """Performance metrics for individual cameras."""
    camera_id: str
    frames_processed: int
    average_fps: float
    detection_accuracy: float
    sync_drift: float  # Milliseconds of sync drift
    gpu_utilization: float
    memory_usage: float

class CameraSynchronizer:
    """
    Handles synchronization of multiple camera streams.

    Features:
    - Frame timestamp alignment
    - Adaptive sync correction
    - Drift detection and compensation
    - Quality metrics
    """

    def __init__(self, target_fps: float = 5.0, sync_tolerance: float = 200.0):
        """
        Initialize camera synchronizer.

        Args:
            target_fps: Target frames per second for synchronization
            sync_tolerance: Maximum acceptable sync drift in milliseconds
        """
        self.target_fps = target_fps
        self.sync_tolerance = sync_tolerance
        self.frame_interval = 1.0 / target_fps

        self.logger = get_logger("CameraSynchronizer")

        # Synchronization state
        self.camera_buffers: Dict[str, deque] = {}
        self.last_sync_time = time.time()
        self.sync_stats = {}

        # Threading
        self._lock = threading.Lock()

    def add_camera(self, camera_id: str, buffer_size: int = 10) -> None:
        """Add a camera to synchronization."""
        with self._lock:
            self.camera_buffers[camera_id] = deque(maxlen=buffer_size)
            self.sync_stats[camera_id] = {
                'frames_received': 0,
                'frames_synced': 0,
                'drift_ms': 0.0,
                'last_timestamp': 0.0
            }

        self.logger.info(f"Camera {camera_id} added to synchronizer")

    def add_frame(self, camera_id: str, frame: np.ndarray, timestamp: float, frame_id: int) -> None:
        """Add frame from camera to synchronization buffer."""
        if camera_id not in self.camera_buffers:
            self.add_camera(camera_id)

        with self._lock:
            frame_data = {
                'frame': frame,
                'timestamp': timestamp,
                'frame_id': frame_id,
                'received_at': time.time()
            }

            self.camera_buffers[camera_id].append(frame_data)
            self.sync_stats[camera_id]['frames_received'] += 1
            self.sync_stats[camera_id]['last_timestamp'] = timestamp

    def get_synchronized_frames(self, intersection_id: str) -> Optional[SynchronizedFrame]:
        """
        Get synchronized frames from all cameras.

        Args:
            intersection_id: Intersection identifier

        Returns:
            SynchronizedFrame if synchronization successful, None otherwise
        """
        with self._lock:
            if not self.camera_buffers:
                return None

            # Find the best timestamp for synchronization
            target_timestamp = self._find_sync_timestamp()

            if target_timestamp is None:
                return None

            # Extract frames closest to target timestamp
            synchronized_frames = {}
            frame_ids = {}
            sync_errors = []

            for camera_id, buffer in self.camera_buffers.items():
                best_frame = self._find_closest_frame(buffer, target_timestamp)

                if best_frame is not None:
                    synchronized_frames[camera_id] = best_frame['frame']
                    frame_ids[camera_id] = best_frame['frame_id']

                    # Calculate sync error
                    sync_error = abs(best_frame['timestamp'] - target_timestamp) * 1000  # ms
                    sync_errors.append(sync_error)

                    self.sync_stats[camera_id]['frames_synced'] += 1
                    self.sync_stats[camera_id]['drift_ms'] = sync_error

            if not synchronized_frames:
                return None

            avg_sync_error = np.mean(sync_errors) if sync_errors else 0.0
            sync_quality = max(0.0, 1.0 - (avg_sync_error / self.sync_tolerance))

            return SynchronizedFrame(
                timestamp=target_timestamp,
                intersection_id=intersection_id,
                frames=synchronized_frames,
                frame_ids=frame_ids,
                sync_quality=sync_quality
            )

    def _find_sync_timestamp(self) -> Optional[float]:
        """Find optimal timestamp for synchronization."""
        if not self.camera_buffers:
            return None

        # Get latest timestamps from each camera
        latest_timestamps = []
        for buffer in self.camera_buffers.values():
            if buffer:
                latest_timestamps.append(buffer[-1]['timestamp'])

        if not latest_timestamps:
            return None

        # Use the earliest of the latest timestamps to ensure all cameras have data
        return min(latest_timestamps)

    def _find_closest_frame(self, buffer: deque, target_timestamp: float) -> Optional[Dict]:
        """Find frame closest to target timestamp in buffer."""
        if not buffer:
            return None

        best_frame = None
        best_diff = float('inf')

        for frame_data in buffer:
            diff = abs(frame_data['timestamp'] - target_timestamp)
            if diff < best_diff:
                best_diff = diff
                best_frame = frame_data

        return best_frame if best_diff <= (self.sync_tolerance / 1000.0) else None

    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        with self._lock:
            stats = {
                'cameras': len(self.camera_buffers),
                'target_fps': self.target_fps,
                'sync_tolerance_ms': self.sync_tolerance,
                'camera_stats': self.sync_stats.copy()
            }

            # Calculate overall sync quality
            if self.sync_stats:
                avg_drift = np.mean([s['drift_ms'] for s in self.sync_stats.values()])
                stats['average_drift_ms'] = avg_drift
                stats['sync_quality'] = max(0.0, 1.0 - (avg_drift / self.sync_tolerance))

            return stats


class DetectionFuser:
    """
    Fuses detection results from multiple cameras to eliminate duplicates
    and provide comprehensive intersection coverage.

    Features:
    - Overlap detection and deduplication
    - Confidence-based fusion
    - Spatial reasoning for vehicle tracking
    - Coverage completeness analysis
    """

    def __init__(self, intersection_id: str):
        """
        Initialize detection fuser.

        Args:
            intersection_id: Intersection identifier
        """
        self.intersection_id = intersection_id
        self.logger = get_logger(f"DetectionFuser-{intersection_id}")

        # Fusion parameters
        self.overlap_threshold = 0.5  # IoU threshold for overlap detection
        self.confidence_weight = 0.7  # Weight for confidence-based fusion
        self.spatial_weight = 0.3     # Weight for spatial reasoning

        # Camera configurations
        self.camera_configs: Dict[str, CameraConfiguration] = {}

        # Performance tracking
        self.fusion_stats = {
            'fusions_performed': 0,
            'duplicates_removed': 0,
            'confidence_improvements': 0
        }

    def add_camera_config(self, config: CameraConfiguration) -> None:
        """Add camera configuration for fusion."""
        self.camera_configs[config.camera_id] = config
        self.logger.info(f"Camera {config.camera_id} added to fusion system")

    def fuse_detections(self, camera_results: Dict[str, DetectionResult],
                       sync_quality: float) -> FusedDetectionResult:
        """
        Fuse detection results from multiple cameras.

        Args:
            camera_results: Dictionary mapping camera_id to DetectionResult
            sync_quality: Quality of frame synchronization

        Returns:
            FusedDetectionResult with deduplicated and enhanced detections
        """
        start_time = time.time()

        try:
            # Collect all detections with camera source info
            all_detections = []
            total_vehicles = 0
            all_confidences = []

            for camera_id, result in camera_results.items():
                for detection in result.detections:
                    detection_with_source = detection.copy()
                    detection_with_source['camera_id'] = camera_id
                    detection_with_source['camera_priority'] = self.camera_configs.get(camera_id, CameraConfiguration(
                        camera_id=camera_id, camera_index=0, intersection_id=self.intersection_id, position='unknown'
                    )).priority
                    all_detections.append(detection_with_source)

                total_vehicles += result.vehicle_count
                all_confidences.extend(result.confidence_scores)

            # Perform deduplication and fusion
            fused_detections = self._deduplicate_detections(all_detections)

            coverage_completeness = self._calculate_coverage_completeness(camera_results)

            self.fusion_stats['fusions_performed'] += 1
            duplicates_removed = len(all_detections) - len(fused_detections)
            self.fusion_stats['duplicates_removed'] += duplicates_removed

            processing_time = time.time() - start_time

            result = FusedDetectionResult(
                intersection_id=self.intersection_id,
                timestamp=time.time(),
                total_vehicle_count=len(fused_detections),
                camera_results=camera_results,
                fused_detections=fused_detections,
                confidence_scores=[d.get('confidence', 0.0) for d in fused_detections],
                processing_time=processing_time,
                sync_quality=sync_quality,
                coverage_completeness=coverage_completeness
            )

            self.logger.debug(f"Fusion completed: {len(all_detections)} -> {len(fused_detections)} detections "
                            f"({duplicates_removed} duplicates removed)")

            return result

        except Exception as e:
            self.logger.error(f"Detection fusion failed: {e}")
            raise VehicleDetectionError(f"Fusion failed: {e}")

    def _deduplicate_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate detections using spatial overlap analysis."""
        if len(detections) <= 1:
            return detections

        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0.0), reverse=True)

        fused_detections = []
        used_indices = set()

        for i, detection in enumerate(sorted_detections):
            if i in used_indices:
                continue

            # Find overlapping detections
            overlapping = [detection]
            overlapping_indices = [i]

            for j, other_detection in enumerate(sorted_detections[i+1:], i+1):
                if j in used_indices:
                    continue

                # Calculate IoU (Intersection over Union)
                iou = self._calculate_iou(detection, other_detection)

                if iou > self.overlap_threshold:
                    overlapping.append(other_detection)
                    overlapping_indices.append(j)

            # Fuse overlapping detections
            if len(overlapping) > 1:
                fused_detection = self._fuse_overlapping_detections(overlapping)
                self.fusion_stats['confidence_improvements'] += 1
            else:
                fused_detection = detection

            fused_detections.append(fused_detection)
            used_indices.update(overlapping_indices)

        return fused_detections

    def _calculate_iou(self, det1: Dict[str, Any], det2: Dict[str, Any]) -> float:
        """Calculate Intersection over Union for two detections."""
        try:
            # Extract bounding boxes
            x1_1, y1_1, w1, h1 = det1['bbox']
            x2_1, y2_1 = x1_1 + w1, y1_1 + h1

            x1_2, y1_2, w2, h2 = det2['bbox']
            x2_2, y2_2 = x1_2 + w2, y1_2 + h2

            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)

            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0

            intersection = (x2_i - x1_i) * (y2_i - y1_i)

            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0.0

        except Exception:
            return 0.0

    def _fuse_overlapping_detections(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fuse multiple overlapping detections into a single detection."""
        # Use highest confidence detection as base
        base_detection = detections[0]

        # Calculate weighted average of bounding boxes
        total_confidence = sum(d.get('confidence', 0.0) for d in detections)

        if total_confidence > 0:
            weighted_bbox = [0, 0, 0, 0]
            for detection in detections:
                weight = detection.get('confidence', 0.0) / total_confidence
                bbox = detection['bbox']
                for i in range(4):
                    weighted_bbox[i] += bbox[i] * weight

            # Create fused detection
            fused_detection = base_detection.copy()
            fused_detection['bbox'] = [int(x) for x in weighted_bbox]
            fused_detection['confidence'] = min(1.0, total_confidence / len(detections))
            fused_detection['fused_from'] = [d['camera_id'] for d in detections]
            fused_detection['fusion_confidence'] = len(detections) / len(self.camera_configs)

            return fused_detection

        return base_detection

    def _calculate_coverage_completeness(self, camera_results: Dict[str, DetectionResult]) -> float:
        """Calculate how completely the intersection is covered by active cameras."""
        if not self.camera_configs:
            return 0.0

        # Simple coverage calculation based on active cameras
        active_cameras = len(camera_results)
        total_cameras = len(self.camera_configs)

        if total_cameras == 0:
            return 0.0

        base_coverage = active_cameras / total_cameras

        # Bonus for diverse camera positions
        positions = set()
        for camera_id in camera_results.keys():
            config = self.camera_configs.get(camera_id)
            if config:
                positions.add(config.position)

        position_bonus = len(positions) / 4.0  # Assuming 4 main positions (N, S, E, W)

        return min(1.0, base_coverage * 0.7 + position_bonus * 0.3)

    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get fusion performance statistics."""
        return {
            'intersection_id': self.intersection_id,
            'fusion_stats': self.fusion_stats.copy(),
            'camera_count': len(self.camera_configs),
            'overlap_threshold': self.overlap_threshold,
            'confidence_weight': self.confidence_weight
        }


class MultiCameraCoordinator:
    """
    Main coordinator for multi-camera traffic monitoring system.

    Features:
    - Manages multiple camera streams per intersection
    - Coordinates detection across cameras
    - Provides unified detection results
    - GPU acceleration support
    - Real-time performance optimization
    - Integration with LSTM and RL systems
    """

    def __init__(self, intersection_configs: Dict[str, List[CameraConfiguration]]):
        """
        Initialize multi-camera coordinator.

        Args:
            intersection_configs: Dictionary mapping intersection_id to list of camera configs
        """
        self.config = get_config()
        self.logger = get_logger("MultiCameraCoordinator")

        # Configuration
        self.intersection_configs = intersection_configs
        self.gpu_acceleration = self.config.get('models.yolo.gpu_acceleration', True)
        self.max_concurrent_streams = self.config.get('multi_camera.max_concurrent_streams', 8)
        self.target_fps = self.config.get('multi_camera.target_fps', 5.0)

        # Core components
        self.detectors: Dict[str, ModernVehicleDetector] = {}  # camera_id -> detector
        self.synchronizers: Dict[str, CameraSynchronizer] = {}  # intersection_id -> synchronizer
        self.fusers: Dict[str, DetectionFuser] = {}  # intersection_id -> fuser

        # Video streams and processing
        self.video_streams: Dict[str, cv2.VideoCapture] = {}  # camera_id -> capture
        self.processing_threads: Dict[str, threading.Thread] = {}  # camera_id -> thread
        self.frame_queues: Dict[str, queue.Queue] = {}  # camera_id -> frame queue

        # Performance tracking
        self.performance_metrics: Dict[str, CameraPerformanceMetrics] = {}
        self.total_frames_processed = 0
        self.total_processing_time = 0.0

        # Threading and control
        self._lock = threading.Lock()
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=self.max_concurrent_streams)

        # Initialize system
        self._initialize_components()

        self.logger.info(f"Multi-camera coordinator initialized for {len(intersection_configs)} intersections")

    def _initialize_components(self) -> None:
        """Initialize all coordinator components."""
        try:
            for intersection_id, camera_configs in self.intersection_configs.items():
                self.synchronizers[intersection_id] = CameraSynchronizer(
                    target_fps=self.target_fps,
                    sync_tolerance=200.0  # 200ms tolerance for 5 FPS
                )

                # Create detection fuser for intersection
                self.fusers[intersection_id] = DetectionFuser(intersection_id)

                for camera_config in camera_configs:
                    self._initialize_camera(camera_config)

                    self.fusers[intersection_id].add_camera_config(camera_config)

                    self.synchronizers[intersection_id].add_camera(camera_config.camera_id)

            self.logger.info("All coordinator components initialized successfully")

        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise CameraConnectionError(f"Coordinator initialization failed: {e}")

    def _initialize_camera(self, camera_config: CameraConfiguration) -> None:
        """Initialize a single camera and its detector."""
        try:
            camera_id = camera_config.camera_id

            # Create detector for camera
            detector = ModernVehicleDetector()

            # Configure GPU acceleration if available
            if self.gpu_acceleration:
                detector.enable_gpu_acceleration()

            self.detectors[camera_id] = detector

            # Initialize performance metrics
            self.performance_metrics[camera_id] = CameraPerformanceMetrics(
                camera_id=camera_id,
                frames_processed=0,
                average_fps=0.0,
                detection_accuracy=0.0,
                sync_drift=0.0,
                gpu_utilization=0.0,
                memory_usage=0.0
            )

            # Create frame queue
            self.frame_queues[camera_id] = queue.Queue(maxsize=10)

            self.logger.info(f"Camera {camera_id} initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize camera {camera_id}: {e}")
            raise CameraConnectionError(f"Camera initialization failed: {e}")

    def start_multi_camera_processing(self) -> None:
        """Start multi-camera processing for all intersections."""
        try:
            with self._lock:
                if self._running:
                    self.logger.warning("Multi-camera processing already running")
                    return

                self._running = True

            for intersection_id, camera_configs in self.intersection_configs.items():
                for camera_config in camera_configs:
                    self._start_camera_stream(camera_config)

            self.logger.info("Multi-camera processing started for all intersections")

        except Exception as e:
            self.logger.error(f"Failed to start multi-camera processing: {e}")
            self.stop_multi_camera_processing()
            raise

    def _start_camera_stream(self, camera_config: CameraConfiguration) -> None:
        """Start processing for a single camera stream."""
        try:
            camera_id = camera_config.camera_id

            # Open video capture
            cap = cv2.VideoCapture(camera_config.camera_index)

            if not cap.isOpened():
                raise CameraConnectionError(f"Failed to open camera {camera_id}")

            # Configure camera
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, camera_config.fps)

            self.video_streams[camera_id] = cap

            # Start processing thread
            thread = threading.Thread(
                target=self._camera_processing_loop,
                args=(camera_config,),
                daemon=True
            )
            thread.start()
            self.processing_threads[camera_id] = thread

            self.logger.info(f"Camera stream started: {camera_id}")

        except Exception as e:
            self.logger.error(f"Failed to start camera stream {camera_id}: {e}")
            raise

    def _camera_processing_loop(self, camera_config: CameraConfiguration) -> None:
        """Main processing loop for a single camera."""
        camera_id = camera_config.camera_id
        intersection_id = camera_config.intersection_id
        cap = self.video_streams[camera_id]
        detector = self.detectors[camera_id]
        synchronizer = self.synchronizers[intersection_id]

        frame_count = 0
        start_time = time.time()

        try:
            while self._running:
                ret, frame = cap.read()

                if not ret:
                    self.logger.warning(f"Failed to read frame from camera {camera_id}")
                    time.sleep(0.1)
                    continue

                timestamp = time.time()
                frame_count += 1

                if camera_config.roi:
                    x, y, w, h = camera_config.roi
                    frame = frame[y:y+h, x:x+w]

                if camera_config.transform_matrix is not None:
                    frame = cv2.warpPerspective(frame, camera_config.transform_matrix,
                                              (frame.shape[1], frame.shape[0]))

                synchronizer.add_frame(camera_id, frame, timestamp, frame_count)

                # Update performance metrics
                self._update_camera_performance(camera_id, frame_count, start_time)

                # Control frame rate
                time.sleep(max(0, 1.0/camera_config.fps - 0.001))

        except Exception as e:
            self.logger.error(f"Camera processing loop failed for {camera_id}: {e}")
        finally:
            if cap:
                cap.release()

    @performance_monitor("MultiCameraDetection")
    def get_fused_detection_results(self, intersection_id: str) -> Optional[FusedDetectionResult]:
        """
        Get fused detection results for an intersection.

        Args:
            intersection_id: Intersection identifier

        Returns:
            FusedDetectionResult if successful, None otherwise
        """
        try:
            if intersection_id not in self.synchronizers:
                self.logger.error(f"Unknown intersection: {intersection_id}")
                return None

            synchronizer = self.synchronizers[intersection_id]
            fuser = self.fusers[intersection_id]

            sync_frames = synchronizer.get_synchronized_frames(intersection_id)

            if sync_frames is None:
                return None

            # Run detection on all synchronized frames
            camera_results = {}

            for camera_id, frame in sync_frames.frames.items():
                if camera_id in self.detectors:
                    detector = self.detectors[camera_id]
                    frame_id = sync_frames.frame_ids[camera_id]

                    try:
                        detection_result = detector.detect_vehicles(frame, frame_id)
                        camera_results[camera_id] = detection_result
                    except Exception as e:
                        self.logger.error(f"Detection failed for camera {camera_id}: {e}")

            if not camera_results:
                return None

            # Fuse detection results
            fused_result = fuser.fuse_detections(camera_results, sync_frames.sync_quality)

            with self._lock:
                self.total_frames_processed += len(camera_results)
                self.total_processing_time += fused_result.processing_time

            return fused_result

        except Exception as e:
            self.logger.error(f"Failed to get fused detection results: {e}")
            return None

    def get_all_intersection_results(self) -> Dict[str, FusedDetectionResult]:
        """Get fused detection results for all intersections."""
        results = {}

        for intersection_id in self.intersection_configs.keys():
            result = self.get_fused_detection_results(intersection_id)
            if result:
                results[intersection_id] = result

        return results

    def _update_camera_performance(self, camera_id: str, frame_count: int, start_time: float) -> None:
        """Update performance metrics for a camera."""
        if camera_id in self.performance_metrics:
            metrics = self.performance_metrics[camera_id]

            # Update basic metrics
            metrics.frames_processed = frame_count
            elapsed_time = time.time() - start_time
            metrics.average_fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0

            intersection_id = None
            for iid, configs in self.intersection_configs.items():
                if any(c.camera_id == camera_id for c in configs):
                    intersection_id = iid
                    break

            if intersection_id and intersection_id in self.synchronizers:
                sync_stats = self.synchronizers[intersection_id].get_sync_statistics()
                camera_stats = sync_stats.get('camera_stats', {}).get(camera_id, {})
                metrics.sync_drift = camera_stats.get('drift_ms', 0.0)

    def get_comprehensive_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics for all cameras and intersections."""
        stats = {
            'total_intersections': len(self.intersection_configs),
            'total_cameras': len(self.detectors),
            'total_frames_processed': self.total_frames_processed,
            'average_processing_time': (self.total_processing_time / max(1, self.total_frames_processed)),
            'gpu_acceleration_enabled': self.gpu_acceleration,
            'target_fps': self.target_fps,
            'running': self._running
        }

        # Individual camera performance
        stats['camera_performance'] = {
            camera_id: {
                'frames_processed': metrics.frames_processed,
                'average_fps': metrics.average_fps,
                'sync_drift_ms': metrics.sync_drift,
                'gpu_utilization': metrics.gpu_utilization
            }
            for camera_id, metrics in self.performance_metrics.items()
        }

        # Intersection-level statistics
        stats['intersection_stats'] = {}
        for intersection_id in self.intersection_configs.keys():
            if intersection_id in self.synchronizers and intersection_id in self.fusers:
                sync_stats = self.synchronizers[intersection_id].get_sync_statistics()
                fusion_stats = self.fusers[intersection_id].get_fusion_statistics()

                stats['intersection_stats'][intersection_id] = {
                    'synchronization': sync_stats,
                    'fusion': fusion_stats,
                    'camera_count': len(self.intersection_configs[intersection_id])
                }

        return stats

    def stop_multi_camera_processing(self) -> None:
        """Stop multi-camera processing for all intersections."""
        try:
            with self._lock:
                self._running = False

            for camera_id, thread in self.processing_threads.items():
                if thread.is_alive():
                    thread.join(timeout=5.0)

            # Release all video captures
            for camera_id, cap in self.video_streams.items():
                if cap:
                    cap.release()

            self.video_streams.clear()
            self.processing_threads.clear()

            # Shutdown executor
            self._executor.shutdown(wait=True)

            self.logger.info("Multi-camera processing stopped")

        except Exception as e:
            self.logger.error(f"Error stopping multi-camera processing: {e}")

    def cleanup(self) -> None:
        """Cleanup all coordinator resources."""
        try:
            self.stop_multi_camera_processing()

            # Cleanup detectors
            for detector in self.detectors.values():
                detector.cleanup()

            self.detectors.clear()
            self.synchronizers.clear()
            self.fusers.clear()
            self.performance_metrics.clear()

            self.logger.info("Multi-camera coordinator cleaned up")

        except Exception as e:
            self.logger.error(f"Coordinator cleanup failed: {e}")
