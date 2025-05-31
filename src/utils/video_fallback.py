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

Video Fallback System for Streamlit Cloud Deployment

Provides fallback functionality when sample videos are not available,
creating synthetic video data for demonstration purposes.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import os

def get_available_sample_videos() -> List[Dict[str, Any]]:
    """
    Get list of available sample videos with fallback to synthetic data.
    
    Returns:
        List of video information dictionaries
    """
    sample_videos_dir = Path("data/sample_videos")
    video_list = []
    
    # Check for actual video files
    if sample_videos_dir.exists():
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.m4v']
        for ext in video_extensions:
            for video_file in sample_videos_dir.glob(f"*{ext}"):
                if video_file.is_file():
                    video_list.append({
                        'name': video_file.stem.replace('_', ' ').title(),
                        'path': str(video_file),
                        'type': 'file',
                        'size': video_file.stat().st_size if video_file.exists() else 0
                    })
    
    # If no videos found, provide synthetic options
    if not video_list:
        video_list = [
            {
                'name': 'Urban Traffic Simulation',
                'path': 'synthetic_urban',
                'type': 'synthetic',
                'description': 'Simulated urban intersection with moderate traffic'
            },
            {
                'name': 'Highway Traffic Simulation', 
                'path': 'synthetic_highway',
                'type': 'synthetic',
                'description': 'Simulated highway traffic with varying density'
            },
            {
                'name': 'Rush Hour Simulation',
                'path': 'synthetic_rush',
                'type': 'synthetic', 
                'description': 'Simulated rush hour traffic patterns'
            },
            {
                'name': 'Light Traffic Simulation',
                'path': 'synthetic_light',
                'type': 'synthetic',
                'description': 'Simulated light traffic conditions'
            }
        ]
    
    return video_list

def create_synthetic_frame(frame_number: int, scenario: str = 'urban') -> np.ndarray:
    """
    Create a synthetic traffic frame for demonstration.
    
    Args:
        frame_number: Current frame number for animation
        scenario: Type of traffic scenario
        
    Returns:
        Synthetic frame as numpy array
    """
    # Create base frame (640x480, 3 channels)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Background (road)
    frame[:, :] = (50, 50, 50)  # Dark gray road
    
    # Road markings
    cv2.line(frame, (0, 240), (640, 240), (255, 255, 255), 2)  # Center line
    cv2.line(frame, (320, 0), (320, 480), (255, 255, 255), 2)  # Intersection
    
    # Simulate vehicles based on scenario
    if scenario == 'urban':
        vehicle_count = 8 + int(3 * np.sin(frame_number * 0.1))
    elif scenario == 'highway':
        vehicle_count = 12 + int(5 * np.sin(frame_number * 0.05))
    elif scenario == 'rush':
        vehicle_count = 15 + int(7 * np.sin(frame_number * 0.02))
    else:  # light
        vehicle_count = 3 + int(2 * np.sin(frame_number * 0.15))
    
    # Draw vehicles as rectangles
    for i in range(vehicle_count):
        # Vehicle position with animation
        x = int(50 + (i * 70 + frame_number * 2) % 540)
        y = int(200 + 80 * (i % 3) + 10 * np.sin(frame_number * 0.1 + i))
        
        # Vehicle color (different for each vehicle)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        color = colors[i % len(colors)]
        
        # Draw vehicle
        cv2.rectangle(frame, (x, y), (x + 40, y + 20), color, -1)
        cv2.rectangle(frame, (x, y), (x + 40, y + 20), (255, 255, 255), 1)
    
    # Add scenario text
    cv2.putText(frame, f"{scenario.title()} Traffic - Frame {frame_number}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def get_synthetic_video_generator(scenario: str = 'urban', duration_frames: int = 300):
    """
    Generator that yields synthetic video frames.
    
    Args:
        scenario: Traffic scenario type
        duration_frames: Number of frames to generate
        
    Yields:
        Synthetic video frames
    """
    for frame_num in range(duration_frames):
        yield create_synthetic_frame(frame_num, scenario)

def create_fallback_video_info() -> Dict[str, Any]:
    """
    Create video information for fallback scenarios.
    
    Returns:
        Dictionary with video metadata
    """
    return {
        'fps': 30,
        'width': 640,
        'height': 480,
        'total_frames': 300,
        'duration': 10.0,
        'format': 'synthetic'
    }

def download_kaggle_datasets_info() -> List[Dict[str, str]]:
    """
    Provide information about Kaggle datasets for download.
    
    Returns:
        List of dataset information
    """
    return [
        {
            'name': 'chicicecream/720p-road-and-traffic-video-for-object-detection',
            'description': '720p road and traffic videos for object detection',
            'command': 'kaggle datasets download chicicecream/720p-road-and-traffic-video-for-object-detection',
            'extract_to': 'data/kaggle/720p-road-and-traffic-video/'
        },
        {
            'name': 'aryashah2k/highway-traffic-videos-dataset', 
            'description': 'Highway traffic videos dataset',
            'command': 'kaggle datasets download aryashah2k/highway-traffic-videos-dataset',
            'extract_to': 'data/kaggle/highway-traffic-videos/'
        }
    ]
