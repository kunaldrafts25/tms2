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
Data Processing Utilities for Traffic Management System

This module provides comprehensive data processing capabilities including
video processing, data validation, feature extraction, and data transformation.
"""

import cv2
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Generator
from pathlib import Path
import threading
from dataclasses import dataclass
from datetime import datetime
import json

from .config_manager import get_config
from .logger import get_logger, performance_monitor
from .error_handler import (
    DataProcessingError, CameraConnectionError, error_handler, safe_execute
)

@dataclass
class VideoStreamInfo:
    """Information about a video stream."""
    source: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    is_live: bool

@dataclass
class ProcessedFrame:
    """Processed frame data."""
    frame: np.ndarray
    frame_id: int
    timestamp: float
    metadata: Dict[str, Any]

class DataProcessor:
    """
    Comprehensive data processing system for traffic management.

    Features:
    - Video stream processing
    - Data validation and cleaning
    - Feature extraction
    - Real-time data transformation
    - Batch processing capabilities
    - Multi-source data handling
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data processor.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = get_config()
        self.logger = get_logger("DataProcessor")

        # Processing configuration
        self.resize_frames = self.config.get('data.preprocessing.resize_frames', True)
        self.target_size = tuple(self.config.get('data.preprocessing.target_size', [640, 480]))
        self.normalize = self.config.get('data.preprocessing.normalize', True)
        self.frame_skip_ratio = self.config.get('performance.frame_skip_ratio', 2)

        # Data storage paths
        self.input_path = self.config.get('data.input_sources', [])
        self.output_path = self.config.get('data.output_path', 'data/processed/')
        self.backup_path = self.config.get('data.backup_path', 'data/backup/')

        # Active video streams
        self.video_streams: Dict[str, cv2.VideoCapture] = {}
        self.stream_info: Dict[str, VideoStreamInfo] = {}

        # Processing statistics
        self.frames_processed = 0
        self.total_processing_time = 0.0
        self.errors_count = 0

        # Thread safety
        self._lock = threading.Lock()

        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories for data processing."""
        for path in [self.output_path, self.backup_path]:
            Path(path).mkdir(parents=True, exist_ok=True)

        self.logger.info("Data processing directories created")

    @error_handler(reraise=True)
    def open_video_stream(self, source: Union[str, int], stream_id: str) -> VideoStreamInfo:
        """
        Open a video stream from file or camera.

        Args:
            source: Video file path or camera index
            stream_id: Unique identifier for the stream

        Returns:
            VideoStreamInfo object containing stream information
        """
        try:
            # Close existing stream if it exists
            if stream_id in self.video_streams:
                self.close_video_stream(stream_id)

            # Open video capture
            cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                raise CameraConnectionError(f"Failed to open video source: {source}")

            # Get stream information
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Determine if it's a live stream
            is_live = isinstance(source, int) or str(source).startswith(('rtmp://', 'http://', 'rtsp://'))

            # Calculate duration for video files
            duration = frame_count / fps if fps > 0 and not is_live else 0.0

            # Store stream
            self.video_streams[stream_id] = cap

            # Create stream info
            stream_info = VideoStreamInfo(
                source=str(source),
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration=duration,
                is_live=is_live
            )

            self.stream_info[stream_id] = stream_info

            self.logger.info(f"Video stream opened: {stream_id} ({width}x{height}, {fps:.1f} FPS)")
            return stream_info

        except Exception as e:
            self.logger.error(f"Failed to open video stream {stream_id}: {e}")
            raise CameraConnectionError(f"Video stream opening failed: {e}")

    def close_video_stream(self, stream_id: str) -> None:
        """Close a video stream."""
        with self._lock:
            if stream_id in self.video_streams:
                self.video_streams[stream_id].release()
                del self.video_streams[stream_id]

                if stream_id in self.stream_info:
                    del self.stream_info[stream_id]

                self.logger.info(f"Video stream closed: {stream_id}")

    def close_all_streams(self) -> None:
        """Close all video streams."""
        stream_ids = list(self.video_streams.keys())
        for stream_id in stream_ids:
            self.close_video_stream(stream_id)

    @performance_monitor("DataProcessor")
    def read_frame(self, stream_id: str) -> Optional[ProcessedFrame]:
        """
        Read and process a frame from a video stream.

        Args:
            stream_id: Stream identifier

        Returns:
            ProcessedFrame object or None if no frame available
        """
        if stream_id not in self.video_streams:
            raise DataProcessingError(f"Stream not found: {stream_id}")

        try:
            cap = self.video_streams[stream_id]
            ret, frame = cap.read()

            if not ret or frame is None:
                return None

            # Process frame
            processed_frame = self._process_frame(frame, stream_id)

            # Update statistics
            self.frames_processed += 1

            return processed_frame

        except Exception as e:
            self.errors_count += 1
            self.logger.error(f"Failed to read frame from {stream_id}: {e}")
            raise DataProcessingError(f"Frame reading failed: {e}")

    def _process_frame(self, frame: np.ndarray, stream_id: str) -> ProcessedFrame:
        """Process a single frame."""
        start_time = time.time()

        metadata = {
            'stream_id': stream_id,
            'original_shape': frame.shape,
            'processing_timestamp': time.time()
        }

        # Resize frame if configured
        if self.resize_frames and frame.shape[:2] != self.target_size[::-1]:
            frame = cv2.resize(frame, self.target_size)
            metadata['resized'] = True
            metadata['target_size'] = self.target_size
        else:
            metadata['resized'] = False

        # Normalize frame if configured
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
            metadata['normalized'] = True
        else:
            metadata['normalized'] = False

        processing_time = time.time() - start_time
        metadata['processing_time'] = processing_time
        self.total_processing_time += processing_time

        return ProcessedFrame(
            frame=frame,
            frame_id=self.frames_processed,
            timestamp=time.time(),
            metadata=metadata
        )

    def read_frames_batch(self, stream_id: str, batch_size: int = 10) -> List[ProcessedFrame]:
        """Read multiple frames from a stream."""
        frames = []

        for _ in range(batch_size):
            frame = self.read_frame(stream_id)
            if frame is None:
                break
            frames.append(frame)

        return frames

    def process_video_file(self, video_path: str,
                          output_callback: Optional[callable] = None) -> Generator[ProcessedFrame, None, None]:
        """
        Process an entire video file frame by frame.

        Args:
            video_path: Path to video file
            output_callback: Optional callback function for each processed frame

        Yields:
            ProcessedFrame objects
        """
        stream_id = f"file_{Path(video_path).stem}"

        try:
            # Open video stream
            stream_info = self.open_video_stream(video_path, stream_id)

            frame_count = 0
            while True:
                # Skip frames based on frame_skip_ratio
                if frame_count % self.frame_skip_ratio != 0:
                    frame_count += 1
                    continue

                # Read frame
                processed_frame = self.read_frame(stream_id)
                if processed_frame is None:
                    break

                # Call output callback if provided
                if output_callback:
                    output_callback(processed_frame)

                yield processed_frame
                frame_count += 1

        finally:
            # Clean up
            self.close_video_stream(stream_id)

    def extract_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from a frame for analysis.

        Args:
            frame: Input frame

        Returns:
            Dictionary containing extracted features
        """
        features = {}

        try:
            # Basic image statistics
            if len(frame.shape) == 3:
                # Color image
                features['mean_rgb'] = np.mean(frame, axis=(0, 1)).tolist()
                features['std_rgb'] = np.std(frame, axis=(0, 1)).tolist()

                # Convert to grayscale for additional features
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                # Grayscale image
                gray = frame
                features['mean_intensity'] = float(np.mean(gray))
                features['std_intensity'] = float(np.std(gray))

            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = float(np.sum(edges > 0) / edges.size)

            # Texture features (simplified)
            features['contrast'] = float(np.std(gray))
            features['brightness'] = float(np.mean(gray))

            # Motion estimation (requires previous frame - simplified here)
            features['estimated_motion'] = 0.0  # Placeholder

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            features['error'] = str(e)

        return features

    def validate_data(self, data: Union[np.ndarray, pd.DataFrame, Dict]) -> Tuple[bool, List[str]]:
        """
        Validate data quality and format.

        Args:
            data: Data to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        try:
            if isinstance(data, np.ndarray):
                # Validate numpy array
                if data.size == 0:
                    errors.append("Array is empty")

                if np.isnan(data).any():
                    errors.append("Array contains NaN values")

                if np.isinf(data).any():
                    errors.append("Array contains infinite values")

                # Check for reasonable value ranges (for image data)
                if len(data.shape) >= 2:
                    if data.dtype == np.uint8:
                        if data.min() < 0 or data.max() > 255:
                            errors.append("Image values out of range [0, 255]")
                    elif data.dtype == np.float32 or data.dtype == np.float64:
                        if data.min() < 0 or data.max() > 1:
                            errors.append("Normalized image values out of range [0, 1]")

            elif isinstance(data, pd.DataFrame):
                # Validate DataFrame
                if data.empty:
                    errors.append("DataFrame is empty")

                if data.isnull().any().any():
                    errors.append("DataFrame contains null values")

                # Check for duplicate indices
                if data.index.duplicated().any():
                    errors.append("DataFrame has duplicate indices")

            elif isinstance(data, dict):
                # Validate dictionary
                if not data:
                    errors.append("Dictionary is empty")

                # Check for None values
                none_keys = [k for k, v in data.items() if v is None]
                if none_keys:
                    errors.append(f"Dictionary contains None values for keys: {none_keys}")

        except Exception as e:
            errors.append(f"Validation error: {e}")

        return len(errors) == 0, errors

    def save_processed_data(self, data: Any, filename: str,
                          format_type: str = 'json') -> str:
        """
        Save processed data to file.

        Args:
            data: Data to save
            filename: Output filename
            format_type: Format type ('json', 'csv', 'npy')

        Returns:
            Path to saved file
        """
        output_path = Path(self.output_path) / filename

        try:
            if format_type == 'json':
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)

            elif format_type == 'csv' and isinstance(data, pd.DataFrame):
                data.to_csv(output_path, index=False)

            elif format_type == 'npy' and isinstance(data, np.ndarray):
                np.save(output_path, data)

            else:
                raise DataProcessingError(f"Unsupported format: {format_type}")

            self.logger.info(f"Data saved to {output_path}")
            return str(output_path)

        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
            raise DataProcessingError(f"Data saving failed: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get data processing performance statistics."""
        avg_processing_time = (self.total_processing_time / self.frames_processed
                             if self.frames_processed > 0 else 0.0)

        return {
            'frames_processed': self.frames_processed,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_processing_time,
            'errors_count': self.errors_count,
            'active_streams': len(self.video_streams),
            'processing_fps': 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0
        }

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        with self._lock:
            self.frames_processed = 0
            self.total_processing_time = 0.0
            self.errors_count = 0

        self.logger.info("Data processing statistics reset")

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.close_all_streams()
        self.logger.info("DataProcessor resources cleaned up")
