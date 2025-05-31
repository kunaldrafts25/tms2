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
Video Source Manager for TMS2

This module manages video sources including local video files, cameras,
and public traffic camera feeds with intelligent fallback mechanisms.
"""

import os
import cv2
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import time

from .config_manager import get_config
from .logger import get_logger
from .error_handler import CameraConnectionError, error_handler


@dataclass
class VideoSource:
    """Information about a video source."""
    id: str
    name: str
    path: str
    source_type: str  # 'file', 'camera', 'public', 'stream'
    resolution: Tuple[int, int]
    fps: float
    duration: float  # 0 for live sources
    file_size: int  # 0 for live sources
    is_available: bool
    priority: int  # Lower number = higher priority


class VideoSourceManager:
    """
    Manages video sources with intelligent fallback mechanisms.

    Features:
    - Automatic detection of local video files
    - Camera enumeration and testing
    - Public traffic camera integration
    - Intelligent fallback when sources fail
    - Source prioritization and selection
    """

    def __init__(self):
        """Initialize the video source manager."""
        self.config = get_config()
        self.logger = get_logger("VideoSourceManager")

        # Video source storage
        self.video_sources: Dict[str, VideoSource] = {}
        self.last_scan_time = 0
        self.scan_interval = 30  # Rescan every 30 seconds

        # Supported video formats
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v']

        # Search directories for video files
        self.search_directories = [
            Path('.'),  # Current directory
            Path('./tms'),  # TMS directory
            Path('./data/sample_videos'),  # Sample videos
            Path('./videos'),  # Videos directory
            Path('./test_videos'),  # Test videos
        ]

        self._scan_video_sources()

    def _scan_video_sources(self) -> None:
        """Scan for available video sources."""
        current_time = time.time()

        # Skip if recently scanned
        if current_time - self.last_scan_time < self.scan_interval:
            return

        self.logger.info("Scanning for available video sources...")

        # Clear existing sources
        self.video_sources.clear()

        # Scan for local video files
        self._scan_local_video_files()

        # Scan for cameras (if enabled)
        if self.config.get('video_sources.scan_cameras', True):
            self._scan_cameras()

        # Add public camera sources (if configured)
        if self.config.get('traffic_cameras.public_feeds_enabled', False):
            self._add_public_camera_sources()

        self.last_scan_time = current_time

        self.logger.info(f"Found {len(self.video_sources)} video sources")

    def _scan_local_video_files(self) -> None:
        """Scan for local video files."""
        priority = 10  # Start with lower priority for files

        for search_dir in self.search_directories:
            if not search_dir.exists():
                continue

            self.logger.debug(f"Scanning directory: {search_dir}")

            for format_ext in self.supported_formats:
                pattern = str(search_dir / f"*{format_ext}")
                video_files = glob.glob(pattern)

                for video_file in video_files:
                    try:
                        video_path = Path(video_file)

                        # Skip if already added
                        source_id = f"file_{video_path.stem}"
                        if source_id in self.video_sources:
                            continue

                        # Test if video is readable
                        cap = cv2.VideoCapture(str(video_path))
                        if not cap.isOpened():
                            self.logger.debug(f"Cannot open video file: {video_path}")
                            continue

                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        duration = frame_count / fps if fps > 0 else 0
                        file_size = video_path.stat().st_size

                        cap.release()

                        video_source = VideoSource(
                            id=source_id,
                            name=f"Local Video: {video_path.name}",
                            path=str(video_path),
                            source_type='file',
                            resolution=(width, height),
                            fps=fps,
                            duration=duration,
                            file_size=file_size,
                            is_available=True,
                            priority=priority
                        )

                        self.video_sources[source_id] = video_source
                        self.logger.debug(f"Added video source: {video_source.name}")

                        priority += 1

                    except Exception as e:
                        self.logger.debug(f"Error processing video file {video_file}: {e}")

    def _scan_cameras(self) -> None:
        """Scan for available cameras."""
        self.logger.debug("Scanning for cameras...")

        # Test camera indices 0-5
        for camera_index in range(6):
            try:
                cap = cv2.VideoCapture(camera_index)

                if cap.isOpened():
                    # Test if we can read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Get camera properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)

                        # Create camera source
                        source_id = f"camera_{camera_index}"
                        video_source = VideoSource(
                            id=source_id,
                            name=f"Camera {camera_index}",
                            path=str(camera_index),
                            source_type='camera',
                            resolution=(width, height),
                            fps=fps,
                            duration=0,  # Live source
                            file_size=0,  # Live source
                            is_available=True,
                            priority=camera_index + 1  # Higher priority for lower indices
                        )

                        self.video_sources[source_id] = video_source
                        self.logger.debug(f"Added camera source: {video_source.name}")

                cap.release()

            except Exception as e:
                self.logger.debug(f"Error testing camera {camera_index}: {e}")

    def _add_public_camera_sources(self) -> None:
        """Add public camera sources from configuration."""
        try:
            public_sources = self.config.get('traffic_cameras.public_sources', [])

            for i, source_config in enumerate(public_sources):
                source_id = f"public_{source_config['id']}"

                video_source = VideoSource(
                    id=source_id,
                    name=f"Public Camera: {source_config['name']}",
                    path=source_config['url'],
                    source_type='public',
                    resolution=(640, 480),  # Default resolution
                    fps=30.0,  # Default FPS
                    duration=0,  # Live source
                    file_size=0,  # Live source
                    is_available=True,  # Assume available until tested
                    priority=source_config.get('priority', 5)
                )

                self.video_sources[source_id] = video_source
                self.logger.debug(f"Added public camera source: {video_source.name}")

        except Exception as e:
            self.logger.debug(f"Error adding public camera sources: {e}")

    def get_available_sources(self, source_type: Optional[str] = None) -> List[VideoSource]:
        """
        Get list of available video sources.

        Args:
            source_type: Filter by source type ('file', 'camera', 'public', 'stream')

        Returns:
            List of available video sources sorted by priority
        """
        self._scan_video_sources()  # Refresh if needed

        sources = [source for source in self.video_sources.values() if source.is_available]

        if source_type:
            sources = [source for source in sources if source.source_type == source_type]

        # Sort by priority (lower number = higher priority)
        sources.sort(key=lambda x: x.priority)

        return sources

    def get_best_source(self, prefer_type: Optional[str] = None) -> Optional[VideoSource]:
        """
        Get the best available video source.

        Args:
            prefer_type: Preferred source type

        Returns:
            Best available video source or None
        """
        sources = self.get_available_sources()

        if not sources:
            return None

        # If preference specified, try to find that type first
        if prefer_type:
            preferred_sources = [s for s in sources if s.source_type == prefer_type]
            if preferred_sources:
                return preferred_sources[0]

        return sources[0]

    @error_handler
    def open_video_source(self, source_spec: Union[str, int], fallback: bool = True) -> Tuple[cv2.VideoCapture, VideoSource]:
        """
        Open a video source with fallback support.

        Args:
            source_spec: Source specification ('public', camera index, file path, etc.)
            fallback: Whether to use fallback sources if primary fails

        Returns:
            Tuple of (VideoCapture object, VideoSource info)
        """
        self.logger.info(f"Opening video source: {source_spec}")

        if source_spec == 'public' or source_spec == 'traffic':
            return self._open_public_source(fallback)
        elif isinstance(source_spec, int) or (isinstance(source_spec, str) and source_spec.isdigit()):
            return self._open_camera_source(int(source_spec), fallback)
        elif isinstance(source_spec, str):
            return self._open_file_source(source_spec, fallback)
        else:
            raise CameraConnectionError(f"Invalid source specification: {source_spec}")

    def _open_public_source(self, fallback: bool) -> Tuple[cv2.VideoCapture, VideoSource]:
        """Open public traffic camera source."""
        self.logger.info("Attempting to open public traffic camera...")

        # Try public cameras first
        public_sources = self.get_available_sources('public')

        for source in public_sources:
            try:
                self.logger.info(f"Trying public camera: {source.name} ({source.path})")

                # Try to open the camera URL
                cap = cv2.VideoCapture(source.path)

                # Set timeout for network streams
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 second timeout
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)   # 5 second read timeout

                if cap.isOpened():
                    # Test frame read
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.logger.info(f"Successfully connected to public camera: {source.name}")
                        return cap, source
                    else:
                        self.logger.warning(f"Public camera {source.name} opened but cannot read frames")
                        cap.release()
                else:
                    self.logger.warning(f"Cannot open public camera: {source.name}")
                    cap.release()

            except Exception as e:
                self.logger.warning(f"Failed to open public camera {source.name}: {e}")

        if fallback:
            self.logger.info("Public cameras failed, falling back to local sources...")
            return self._open_fallback_source()
        else:
            raise CameraConnectionError("No public traffic cameras available")

    def _open_camera_source(self, camera_index: int, fallback: bool) -> Tuple[cv2.VideoCapture, VideoSource]:
        """Open camera source."""
        self.logger.info(f"Attempting to open camera {camera_index}...")

        try:
            cap = cv2.VideoCapture(camera_index)

            if cap.isOpened():
                # Test frame read
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Find or create source info
                    source_id = f"camera_{camera_index}"
                    if source_id in self.video_sources:
                        source = self.video_sources[source_id]
                    else:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)

                        source = VideoSource(
                            id=source_id,
                            name=f"Camera {camera_index}",
                            path=str(camera_index),
                            source_type='camera',
                            resolution=(width, height),
                            fps=fps,
                            duration=0,
                            file_size=0,
                            is_available=True,
                            priority=camera_index + 1
                        )

                    self.logger.info(f"Successfully opened camera {camera_index}")
                    return cap, source
                else:
                    cap.release()
                    raise CameraConnectionError(f"Cannot read from camera {camera_index}")
            else:
                raise CameraConnectionError(f"Cannot open camera {camera_index}")

        except Exception as e:
            self.logger.warning(f"Failed to open camera {camera_index}: {e}")

            if fallback:
                self.logger.info("Camera failed, falling back to local video files...")
                return self._open_fallback_source()
            else:
                raise CameraConnectionError(f"Camera {camera_index} not available: {e}")

    def _open_file_source(self, file_path: str, fallback: bool) -> Tuple[cv2.VideoCapture, VideoSource]:
        """Open file source."""
        self.logger.info(f"Attempting to open video file: {file_path}")

        try:
            cap = cv2.VideoCapture(file_path)

            if cap.isOpened():
                # Get file info
                path_obj = Path(file_path)
                source_id = f"file_{path_obj.stem}"

                if source_id in self.video_sources:
                    source = self.video_sources[source_id]
                else:
                    # Create temporary source info
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0

                    source = VideoSource(
                        id=source_id,
                        name=f"Video File: {path_obj.name}",
                        path=file_path,
                        source_type='file',
                        resolution=(width, height),
                        fps=fps,
                        duration=duration,
                        file_size=path_obj.stat().st_size if path_obj.exists() else 0,
                        is_available=True,
                        priority=10
                    )

                self.logger.info(f"Successfully opened video file: {file_path}")
                return cap, source
            else:
                raise CameraConnectionError(f"Cannot open video file: {file_path}")

        except Exception as e:
            self.logger.warning(f"Failed to open video file {file_path}: {e}")

            if fallback:
                self.logger.info("Video file failed, falling back to other sources...")
                return self._open_fallback_source()
            else:
                raise CameraConnectionError(f"Video file not available: {e}")

    def _open_fallback_source(self) -> Tuple[cv2.VideoCapture, VideoSource]:
        """Open the best available fallback source."""
        self.logger.info("Looking for fallback video sources...")

        file_sources = self.get_available_sources('file')

        if file_sources:
            best_source = file_sources[0]
            self.logger.info(f"Using fallback source: {best_source.name}")

            try:
                cap = cv2.VideoCapture(best_source.path)
                if cap.isOpened():
                    return cap, best_source
                else:
                    cap.release()
            except Exception as e:
                self.logger.error(f"Failed to open fallback source {best_source.name}: {e}")

        # If no video files, try cameras
        camera_sources = self.get_available_sources('camera')
        if camera_sources:
            best_source = camera_sources[0]
            self.logger.info(f"Using camera fallback: {best_source.name}")

            try:
                cap = cv2.VideoCapture(int(best_source.path))
                if cap.isOpened():
                    return cap, best_source
                else:
                    cap.release()
            except Exception as e:
                self.logger.error(f"Failed to open camera fallback {best_source.name}: {e}")

        raise CameraConnectionError("No fallback video sources available")

    def list_sources(self) -> None:
        """Print list of available video sources."""
        sources = self.get_available_sources()

        if not sources:
            print("No video sources found.")
            return

        print(f"\nFound {len(sources)} video sources:")
        print("-" * 80)

        for source in sources:
            print(f"ID: {source.id}")
            print(f"  Name: {source.name}")
            print(f"  Type: {source.source_type}")
            print(f"  Path: {source.path}")
            print(f"  Resolution: {source.resolution[0]}x{source.resolution[1]}")
            print(f"  FPS: {source.fps:.1f}")
            if source.duration > 0:
                print(f"  Duration: {source.duration:.1f}s")
            if source.file_size > 0:
                print(f"  Size: {source.file_size / (1024*1024):.1f} MB")
            print(f"  Priority: {source.priority}")
            print()

    def get_source_recommendations(self) -> Dict[str, str]:
        """Get recommendations for using video sources."""
        recommendations = {}

        file_sources = self.get_available_sources('file')
        camera_sources = self.get_available_sources('camera')
        public_sources = self.get_available_sources('public')

        if file_sources:
            best_file = file_sources[0]
            recommendations['best_file'] = f"python main.py detect --source \"{best_file.path}\" --display"

        if camera_sources:
            best_camera = camera_sources[0]
            recommendations['best_camera'] = f"python main.py detect --source {best_camera.path} --display"

        if public_sources:
            recommendations['public'] = "python main.py detect --source public --display"

        recommendations['fallback'] = "python main.py detect --source public --display  # Will fallback to local files"

        return recommendations
