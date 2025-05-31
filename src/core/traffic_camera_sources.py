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
Traffic Camera Data Sources Manager

This module provides access to publicly available traffic camera feeds
including DOT cameras, city webcams, and traffic camera APIs.
"""

import cv2
import requests
import numpy as np
import time
import threading
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import asyncio
import aiohttp
from urllib.parse import urlparse
import logging

from ..utils.config_manager import get_config
from ..utils.logger import get_logger, performance_monitor
from ..utils.error_handler import CameraConnectionError, error_handler


@dataclass
class TrafficCameraSource:
    """Configuration for a traffic camera data source."""
    id: str
    name: str
    url: str
    source_type: str  # 'rtsp', 'http_stream', 'api', 'mjpeg'
    location: Tuple[float, float]  # (latitude, longitude)
    intersection_id: str
    priority: int = 1
    auth_required: bool = False
    auth_token: Optional[str] = None
    refresh_interval: int = 30  # seconds
    timeout: int = 10
    retry_count: int = 3
    is_active: bool = True
    last_update: Optional[datetime] = None
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraFeedStatus:
    """Status information for a camera feed."""
    camera_id: str
    is_online: bool
    last_frame_time: Optional[datetime]
    fps: float
    resolution: Tuple[int, int]
    error_message: Optional[str] = None
    latency_ms: float = 0.0
    data_quality: float = 1.0  # 0.0 to 1.0


class PublicTrafficCameraManager:
    """
    Manager for public traffic camera data sources.

    Supports multiple camera feed types:
    - RTSP streams
    - HTTP video streams
    - MJPEG streams
    - API-based camera data
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the traffic camera manager."""
        self.config = get_config()
        self.logger = get_logger("PublicTrafficCameraManager")

        # Camera sources configuration
        self.camera_sources: Dict[str, TrafficCameraSource] = {}
        self.active_feeds: Dict[str, cv2.VideoCapture] = {}
        self.feed_status: Dict[str, CameraFeedStatus] = {}

        # Threading for async operations
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._feed_threads: Dict[str, threading.Thread] = {}

        # Performance tracking
        self.total_frames_processed = 0
        self.total_connection_attempts = 0
        self.successful_connections = 0

        # Load camera sources from configuration
        self._load_camera_sources()

        # Initialize known public camera sources
        self._initialize_default_sources()

    def _load_camera_sources(self) -> None:
        """Load camera sources from configuration."""
        try:
            sources_config = self.config.get('traffic_cameras.public_sources', [])

            for source_config in sources_config:
                camera_source = TrafficCameraSource(
                    id=source_config['id'],
                    name=source_config['name'],
                    url=source_config['url'],
                    source_type=source_config.get('type', 'rtsp'),
                    location=(source_config.get('lat', 0.0), source_config.get('lon', 0.0)),
                    intersection_id=source_config.get('intersection_id', 'unknown'),
                    priority=source_config.get('priority', 1),
                    auth_required=source_config.get('auth_required', False),
                    auth_token=source_config.get('auth_token'),
                    refresh_interval=source_config.get('refresh_interval', 30),
                    timeout=source_config.get('timeout', 10),
                    retry_count=source_config.get('retry_count', 3)
                )

                self.camera_sources[camera_source.id] = camera_source

            self.logger.info(f"Loaded {len(self.camera_sources)} camera sources from configuration")

        except Exception as e:
            self.logger.error(f"Failed to load camera sources from configuration: {e}")

    def _initialize_default_sources(self) -> None:
        """Initialize default public traffic camera sources."""
        # Example public traffic camera sources (replace with actual working URLs)
        default_sources = [
            {
                'id': 'dot_cam_001',
                'name': 'DOT Highway Camera 1',
                'url': 'rtsp://example-dot-camera1.gov/stream',
                'type': 'rtsp',
                'lat': 40.7128,
                'lon': -74.0060,
                'intersection_id': 'highway_001'
            },
            {
                'id': 'city_cam_001',
                'name': 'City Traffic Camera 1',
                'url': 'http://example-city-camera1.gov/mjpeg',
                'type': 'mjpeg',
                'lat': 40.7589,
                'lon': -73.9851,
                'intersection_id': 'city_001'
            },
            {
                'id': 'traffic_api_001',
                'name': 'Traffic API Camera 1',
                'url': 'https://api.traffic-cameras.example.com/v1/camera/001/stream',
                'type': 'api',
                'lat': 40.7505,
                'lon': -73.9934,
                'intersection_id': 'api_001'
            }
        ]

        # Add default sources if not already configured
        for source_data in default_sources:
            if source_data['id'] not in self.camera_sources:
                camera_source = TrafficCameraSource(
                    id=source_data['id'],
                    name=source_data['name'],
                    url=source_data['url'],
                    source_type=source_data['type'],
                    location=(source_data['lat'], source_data['lon']),
                    intersection_id=source_data['intersection_id']
                )
                self.camera_sources[camera_source.id] = camera_source

        self.logger.info(f"Initialized {len(default_sources)} default camera sources")

    @error_handler
    def connect_to_camera(self, camera_id: str) -> bool:
        """
        Connect to a specific camera source.

        Args:
            camera_id: ID of the camera to connect to

        Returns:
            True if connection successful, False otherwise
        """
        if camera_id not in self.camera_sources:
            self.logger.error(f"Camera source {camera_id} not found")
            return False

        camera_source = self.camera_sources[camera_id]
        self.total_connection_attempts += 1

        try:
            if camera_source.source_type == 'rtsp':
                return self._connect_rtsp_stream(camera_source)
            elif camera_source.source_type == 'http_stream':
                return self._connect_http_stream(camera_source)
            elif camera_source.source_type == 'mjpeg':
                return self._connect_mjpeg_stream(camera_source)
            elif camera_source.source_type == 'api':
                return self._connect_api_source(camera_source)
            else:
                self.logger.error(f"Unsupported camera source type: {camera_source.source_type}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to connect to camera {camera_id}: {e}")
            camera_source.error_count += 1
            return False

    def _connect_rtsp_stream(self, camera_source: TrafficCameraSource) -> bool:
        """Connect to RTSP stream."""
        try:
            cap = cv2.VideoCapture(camera_source.url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
            cap.set(cv2.CAP_PROP_TIMEOUT, camera_source.timeout * 1000)

            if cap.isOpened():
                # Test frame read
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.active_feeds[camera_source.id] = cap
                    self._update_feed_status(camera_source.id, True, frame.shape[:2][::-1])
                    self.successful_connections += 1
                    self.logger.info(f"Successfully connected to RTSP stream: {camera_source.name}")
                    return True
                else:
                    cap.release()
                    self.logger.error(f"Failed to read frame from RTSP stream: {camera_source.name}")
                    return False
            else:
                self.logger.error(f"Failed to open RTSP stream: {camera_source.name}")
                return False

        except Exception as e:
            self.logger.error(f"RTSP connection error for {camera_source.name}: {e}")
            return False

    def _connect_mjpeg_stream(self, camera_source: TrafficCameraSource) -> bool:
        """Connect to MJPEG stream."""
        try:
            cap = cv2.VideoCapture(camera_source.url)

            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.active_feeds[camera_source.id] = cap
                    self._update_feed_status(camera_source.id, True, frame.shape[:2][::-1])
                    self.successful_connections += 1
                    self.logger.info(f"Successfully connected to MJPEG stream: {camera_source.name}")
                    return True
                else:
                    cap.release()
                    return False
            else:
                return False

        except Exception as e:
            self.logger.error(f"MJPEG connection error for {camera_source.name}: {e}")
            return False

    def _connect_http_stream(self, camera_source: TrafficCameraSource) -> bool:
        """Connect to HTTP video stream."""
        # Similar implementation to MJPEG but with additional HTTP handling
        return self._connect_mjpeg_stream(camera_source)

    def _connect_api_source(self, camera_source: TrafficCameraSource) -> bool:
        """Connect to API-based camera source."""
        try:
            headers = {}
            if camera_source.auth_required and camera_source.auth_token:
                headers['Authorization'] = f"Bearer {camera_source.auth_token}"

            response = requests.get(
                camera_source.url,
                headers=headers,
                timeout=camera_source.timeout,
                stream=True
            )

            if response.status_code == 200:
                # For API sources, we might get image URLs or direct image data
                # This is a simplified implementation
                self._update_feed_status(camera_source.id, True, (640, 480))
                self.successful_connections += 1
                self.logger.info(f"Successfully connected to API source: {camera_source.name}")
                return True
            else:
                self.logger.error(f"API connection failed with status {response.status_code}: {camera_source.name}")
                return False

        except Exception as e:
            self.logger.error(f"API connection error for {camera_source.name}: {e}")
            return False

    def _update_feed_status(self, camera_id: str, is_online: bool,
                          resolution: Tuple[int, int] = (0, 0),
                          error_message: Optional[str] = None) -> None:
        """Update the status of a camera feed."""
        self.feed_status[camera_id] = CameraFeedStatus(
            camera_id=camera_id,
            is_online=is_online,
            last_frame_time=datetime.now() if is_online else None,
            fps=30.0 if is_online else 0.0,  # Default FPS
            resolution=resolution,
            error_message=error_message
        )

    def start_monitoring(self) -> None:
        """Start monitoring all camera feeds."""
        if self._running:
            self.logger.warning("Camera monitoring already running")
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()

        self.logger.info("Started camera feed monitoring")

    def stop_monitoring(self) -> None:
        """Stop monitoring camera feeds."""
        self._running = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

        for camera_id, cap in self.active_feeds.items():
            try:
                cap.release()
            except Exception as e:
                self.logger.error(f"Error closing camera feed {camera_id}: {e}")

        self.active_feeds.clear()
        self.logger.info("Stopped camera feed monitoring")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop for camera feeds."""
        while self._running:
            try:
                # Check each camera source
                for camera_id, camera_source in self.camera_sources.items():
                    if not camera_source.is_active:
                        continue

                    # Check if camera is connected
                    if camera_id not in self.active_feeds:
                        # Try to connect
                        if self.connect_to_camera(camera_id):
                            self.logger.info(f"Connected to camera {camera_id}")
                    else:
                        # Check if feed is still alive
                        if not self._check_feed_health(camera_id):
                            self.logger.warning(f"Camera feed {camera_id} appears unhealthy, reconnecting...")
                            self._reconnect_camera(camera_id)

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)

    def _check_feed_health(self, camera_id: str) -> bool:
        """Check if a camera feed is healthy."""
        if camera_id not in self.active_feeds:
            return False

        try:
            cap = self.active_feeds[camera_id]
            if not cap.isOpened():
                return False

            # Try to read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                self._update_feed_status(camera_id, True, frame.shape[:2][::-1])
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"Health check failed for camera {camera_id}: {e}")
            return False

    def _reconnect_camera(self, camera_id: str) -> None:
        """Reconnect to a camera after connection loss."""
        # Close existing connection
        if camera_id in self.active_feeds:
            try:
                self.active_feeds[camera_id].release()
                del self.active_feeds[camera_id]
            except Exception as e:
                self.logger.error(f"Error closing camera {camera_id}: {e}")

        # Update status
        self._update_feed_status(camera_id, False, error_message="Reconnecting...")

        # Wait before reconnecting
        time.sleep(2)

        # Try to reconnect
        if self.connect_to_camera(camera_id):
            self.logger.info(f"Successfully reconnected to camera {camera_id}")
        else:
            self.logger.error(f"Failed to reconnect to camera {camera_id}")

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Get the latest frame from a camera.

        Args:
            camera_id: ID of the camera

        Returns:
            Frame as numpy array or None if unavailable
        """
        if camera_id not in self.active_feeds:
            return None

        try:
            cap = self.active_feeds[camera_id]
            ret, frame = cap.read()

            if ret and frame is not None:
                self.total_frames_processed += 1
                self._update_feed_status(camera_id, True, frame.shape[:2][::-1])
                return frame
            else:
                self._update_feed_status(camera_id, False, error_message="Failed to read frame")
                return None

        except Exception as e:
            self.logger.error(f"Error reading frame from camera {camera_id}: {e}")
            self._update_feed_status(camera_id, False, error_message=str(e))
            return None

    def get_all_active_cameras(self) -> List[str]:
        """Get list of all active camera IDs."""
        return [camera_id for camera_id, status in self.feed_status.items()
                if status.is_online]

    def get_camera_status(self, camera_id: str) -> Optional[CameraFeedStatus]:
        """Get status of a specific camera."""
        return self.feed_status.get(camera_id)

    def get_all_camera_status(self) -> Dict[str, CameraFeedStatus]:
        """Get status of all cameras."""
        return self.feed_status.copy()

    def add_camera_source(self, camera_source: TrafficCameraSource) -> None:
        """Add a new camera source."""
        self.camera_sources[camera_source.id] = camera_source
        self.logger.info(f"Added camera source: {camera_source.name}")

    def remove_camera_source(self, camera_id: str) -> None:
        """Remove a camera source."""
        if camera_id in self.camera_sources:
            if camera_id in self.active_feeds:
                self.active_feeds[camera_id].release()
                del self.active_feeds[camera_id]

            del self.camera_sources[camera_id]

            if camera_id in self.feed_status:
                del self.feed_status[camera_id]

            self.logger.info(f"Removed camera source: {camera_id}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        active_cameras = len(self.get_all_active_cameras())
        total_cameras = len(self.camera_sources)

        connection_success_rate = (
            self.successful_connections / self.total_connection_attempts
            if self.total_connection_attempts > 0 else 0.0
        )

        return {
            'total_cameras': total_cameras,
            'active_cameras': active_cameras,
            'connection_success_rate': connection_success_rate,
            'total_frames_processed': self.total_frames_processed,
            'total_connection_attempts': self.total_connection_attempts,
            'successful_connections': self.successful_connections
        }

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.stop_monitoring()
        self.camera_sources.clear()
        self.feed_status.clear()
        self.logger.info("PublicTrafficCameraManager cleaned up")


class PublicCameraCapture:
    """
    VideoCapture-like interface for public traffic cameras.

    This class provides a cv2.VideoCapture-compatible interface
    for reading frames from public traffic camera sources.
    """

    def __init__(self, camera_manager: PublicTrafficCameraManager, camera_id: str):
        """
        Initialize the public camera capture.

        Args:
            camera_manager: The public traffic camera manager
            camera_id: ID of the camera to capture from
        """
        self.camera_manager = camera_manager
        self.camera_id = camera_id
        self.logger = get_logger(f"PublicCameraCapture_{camera_id}")
        self._is_opened = True

    def isOpened(self) -> bool:
        """Check if the camera is opened."""
        return self._is_opened and self.camera_id in self.camera_manager.active_feeds

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the public camera.

        Returns:
            Tuple of (success, frame)
        """
        if not self.isOpened():
            return False, None

        try:
            frame = self.camera_manager.get_frame(self.camera_id)
            if frame is not None:
                return True, frame
            else:
                return False, None
        except Exception as e:
            self.logger.error(f"Error reading frame: {e}")
            return False, None

    def get(self, prop_id: int) -> float:
        """
        Get camera property.

        Args:
            prop_id: Property ID (cv2.CAP_PROP_*)

        Returns:
            Property value
        """
        if not self.isOpened():
            return 0.0

        status = self.camera_manager.get_camera_status(self.camera_id)
        if not status:
            return 0.0

        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(status.resolution[0])
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(status.resolution[1])
        elif prop_id == cv2.CAP_PROP_FPS:
            return status.fps
        else:
            return 0.0

    def set(self, prop_id: int, value: float) -> bool:
        """
        Set camera property (not supported for public cameras).

        Args:
            prop_id: Property ID
            value: Property value

        Returns:
            False (not supported)
        """
        self.logger.warning("Setting properties not supported for public cameras")
        return False

    def release(self) -> None:
        """Release the camera."""
        self._is_opened = False
        self.logger.info(f"Released public camera {self.camera_id}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
