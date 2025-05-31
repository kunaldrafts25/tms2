#!/usr/bin/env python3
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
Traffic Management System (TMS2) - Main Application Entry Point

This is the main entry point for the advanced traffic management system.
It provides command-line interface for running different components of the system.
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config_manager import init_config, get_config
from src.utils.logger import setup_logging, get_logger
from src.utils.error_handler import init_error_handler, get_error_handler
from src.core.vehicle_detector import VehicleDetector
from src.core.modern_vehicle_detector import ModernVehicleDetector
from src.core.traffic_predictor import TrafficPredictor
from src.core.signal_controller import SignalController
from src.utils.data_processor import DataProcessor


def setup_system(config_path: str = None, environment: str = "development") -> None:
    """
    Initialize the TMS system with configuration and logging.

    Args:
        config_path: Path to configuration file
        environment: Environment name (development, production, testing)
    """
    # Initialize configuration
    config = init_config(config_path, environment)

    # Setup logging
    logging_config = config.get_section('logging')
    setup_logging(logging_config)

    # Initialize error handler
    logger = get_logger("TMS")
    init_error_handler(logger)

    logger.info(f"TMS system initialized in {environment} mode")


def list_cameras(args):
    """List available cameras on the system."""
    logger = get_logger("CameraEnumeration")

    try:
        # Use modern detector for camera enumeration
        detector = ModernVehicleDetector()
        cameras = detector.list_cameras()

        if not cameras:
            logger.warning("No cameras found on the system")
            print("\n❌ No cameras detected")
            print("\nTroubleshooting:")
            print(detector.get_camera_troubleshooting())
            return 1

        print(f"\n📹 Found {len(cameras)} available cameras:")
        print("-" * 60)

        for camera in cameras:
            print(f"Camera {camera.index}: {camera.name}")
            print(f"  Resolution: {camera.resolution[0]}x{camera.resolution[1]}")
            print(f"  FPS: {camera.fps:.1f}")
            print(f"  Status: {'✅ Available' if camera.is_available else '❌ Not available'}")
            if camera.error_message:
                print(f"  Error: {camera.error_message}")
            print()

        # Show best camera recommendation
        best_camera = detector.camera_manager.get_best_camera()
        if best_camera:
            print(f"🎯 Recommended camera: {best_camera.index} ({best_camera.resolution[0]}x{best_camera.resolution[1]})")
            print(f"   Test with: python main.py detect --source {best_camera.index} --display")

        return 0

    except Exception as e:
        logger.error(f"Camera enumeration failed: {e}")
        return 1


def list_video_sources(args):
    """List all available video sources."""
    logger = get_logger("VideoSourceEnumeration")

    try:
        # Use modern detector to access video source manager
        detector = ModernVehicleDetector()

        if hasattr(detector, 'list_video_sources'):
            detector.list_video_sources()
        else:
            logger.warning("Video source manager not available")
            # Fall back to basic camera listing
            return list_cameras(args)

        return 0

    except Exception as e:
        logger.error(f"Video source enumeration failed: {e}")
        return 1


def run_vehicle_detection(args):
    """Run vehicle detection on video source."""
    import numpy as np

    logger = get_logger("VehicleDetection")

    try:
        # Handle special commands
        if hasattr(args, 'list_cameras') and args.list_cameras:
            return list_cameras(args)

        if hasattr(args, 'list_sources') and args.list_sources:
            return list_video_sources(args)

        # Check for multi-camera mode
        if hasattr(args, 'multi_camera') and args.multi_camera:
            return run_multi_camera_detection(args)

        # Check if source is provided (unless listing cameras)
        if not hasattr(args, 'source') or args.source is None:
            logger.error("No video source specified. Use --source or --list-cameras")
            logger.info("Examples:")
            logger.info("  python main.py detect --source 0 --display")
            logger.info("  python main.py detect --list-cameras")
            return 1

        # Choose detector based on configuration or user preference
        use_modern = not (hasattr(args, 'use_legacy') and args.use_legacy)

        if use_modern:
            try:
                detector = ModernVehicleDetector()

                # Override model if specified
                if hasattr(args, 'model') and args.model:
                    detector.model_name = args.model
                    detector._load_model()  # Reload with new model

                logger.info(f"Using modern YOLOv8/YOLOv11 detector with model: {detector.model_name}")
            except Exception as e:
                logger.warning(f"Modern detector failed, falling back to legacy: {e}")
                detector = VehicleDetector()
                use_modern = False
        else:
            detector = VehicleDetector()
            logger.info("Using legacy YOLOv4 detector")

        logger.info(f"Starting vehicle detection on: {args.source}")

        # Handle camera source with intelligent fallback
        if isinstance(args.source, str) and args.source.isdigit():
            args.source = int(args.source)

        if use_modern and isinstance(args.source, int):
            # Use modern detector's intelligent camera handling
            try:
                cap = detector.open_video_source(args.source)
                logger.info("Successfully opened video source with intelligent fallback")
            except Exception as e:
                logger.error(f"Failed to open video source: {e}")

                # Show available alternatives
                try:
                    cameras = detector.list_cameras()
                    if cameras:
                        logger.info("Available cameras:")
                        for cam in cameras:
                            logger.info(f"  Camera {cam.index}: {cam.resolution[0]}x{cam.resolution[1]}")
                    else:
                        logger.info("No cameras available. Try using a video file:")
                        logger.info("  python main.py detect --source data/sample_videos/traffic.mp4")
                except:
                    logger.info("Camera enumeration failed. Try using a video file:")
                    logger.info("  python main.py detect --source data/sample_videos/traffic.mp4")

                return 1
        else:
            # Use data processor for file/stream sources
            data_processor = DataProcessor()
            stream_info = data_processor.open_video_stream(args.source, "main_stream")
            logger.info(f"Video stream opened: {stream_info.width}x{stream_info.height} @ {stream_info.fps} FPS")
            cap = None

        frame_count = 0
        total_vehicles = 0

        # Main processing loop
        while True:
            if cap:
                # Read directly from camera
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                # Read from data processor
                processed_frame = data_processor.read_frame("main_stream")
                if processed_frame is None:
                    break

                # Convert normalized frame back to uint8 for detection
                if processed_frame.frame.dtype == np.float32:
                    frame = (processed_frame.frame * 255).astype(np.uint8)
                else:
                    frame = processed_frame.frame

            # Detect vehicles
            detection_result = detector.detect_vehicles(frame, frame_count)
            total_vehicles += detection_result.vehicle_count

            # Display results if requested
            if args.display:
                import cv2
                output_frame = detector.draw_detections(frame, detection_result)
                cv2.imshow('TMS2 Vehicle Detection', output_frame)

                # Show instructions
                if frame_count == 0:
                    logger.info("Press 'q' to quit, 's' to save screenshot")

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = f"detection_screenshot_{frame_count}.jpg"
                    cv2.imwrite(screenshot_path, output_frame)
                    logger.info(f"Screenshot saved: {screenshot_path}")

            frame_count += 1

            # Print progress
            if frame_count % 30 == 0:
                avg_vehicles = total_vehicles / frame_count
                logger.info(f"Processed {frame_count} frames, detected {total_vehicles} vehicles (avg: {avg_vehicles:.1f}/frame)")

        # Print final statistics
        stats = detector.get_performance_stats()
        logger.info(f"Detection completed: {stats}")

        # Cleanup
        if cap:
            cap.release()
        else:
            data_processor.cleanup()
        detector.cleanup()

        if args.display:
            import cv2
            cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"Vehicle detection failed: {e}")

        # Provide helpful error guidance
        if "camera" in str(e).lower() or "video" in str(e).lower():
            logger.info("\nTroubleshooting suggestions:")
            logger.info("1. List available sources: python main.py detect --list-sources")
            logger.info("2. Try public cameras with fallback: python main.py detect --source public")
            logger.info("3. Use a specific video file from tms directory")
            logger.info("4. Try different camera: python main.py detect --source 1")

            # Try to show available video sources
            try:
                detector = ModernVehicleDetector()
                if hasattr(detector, 'list_video_sources'):
                    logger.info("\nAvailable video sources:")
                    detector.list_video_sources()
            except Exception:
                logger.info("5. Check if video files exist in ./tms/ directory")

        return 1

    return 0


def run_multi_camera_detection(args):
    """Run multi-camera detection with coordination and fusion."""
    import time

    logger = get_logger("MultiCameraDetection")

    try:
        logger.info("Starting multi-camera detection system...")

        try:
            from src.core.multi_camera_coordinator import (
                MultiCameraCoordinator, CameraConfiguration
            )
        except ImportError as e:
            logger.error(f"Multi-camera coordinator not available: {e}")
            logger.info("Falling back to single camera mode")
            return run_vehicle_detection(args)

        # Create camera configurations
        intersection_id = getattr(args, 'intersection_id', 'main')
        cameras = getattr(args, 'cameras', ['0', '1', '2', '3'])

        camera_configs = []
        for i, camera_index in enumerate(cameras):
            try:
                cam_index = int(camera_index)
                config = CameraConfiguration(
                    camera_id=f"camera_{cam_index}",
                    camera_index=cam_index,
                    intersection_id=intersection_id,
                    position=['north', 'south', 'east', 'west'][i % 4],
                    resolution=(640, 480),
                    fps=30.0,
                    priority=i + 1
                )
                camera_configs.append(config)
            except ValueError:
                logger.warning(f"Invalid camera index: {camera_index}")

        if not camera_configs:
            logger.error("No valid camera configurations")
            return 1

        intersection_configs = {intersection_id: camera_configs}
        coordinator = MultiCameraCoordinator(intersection_configs)

        logger.info(f"Multi-camera coordinator initialized with {len(camera_configs)} cameras")

        coordinator.start_multi_camera_processing()

        # Main processing loop
        frame_count = 0
        start_time = time.time()

        try:
            while True:
                # Get fused detection results
                fused_result = coordinator.get_fused_detection_results(intersection_id)

                if fused_result:
                    frame_count += 1

                    # Log results
                    if frame_count % 30 == 0:
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

                        logger.info(f"Frame {frame_count}: {fused_result.total_vehicle_count} vehicles detected "
                                  f"(FPS: {fps:.1f}, Sync: {fused_result.sync_quality:.2f}, "
                                  f"Coverage: {fused_result.coverage_completeness:.2f})")

                        # Show performance stats
                        stats = coordinator.get_comprehensive_performance_stats()
                        logger.info(f"Performance: {stats['total_cameras']} cameras, "
                                  f"avg processing: {stats['average_processing_time']:.3f}s")

                # Control frame rate
                time.sleep(1.0 / 5.0)  # 5 FPS

        except KeyboardInterrupt:
            logger.info("Multi-camera detection interrupted by user")

        coordinator.stop_multi_camera_processing()
        coordinator.cleanup()

        logger.info("Multi-camera detection completed successfully")

    except Exception as e:
        logger.error(f"Multi-camera detection failed: {e}")

        # Provide helpful guidance
        logger.info("\n🔧 Multi-camera troubleshooting:")
        logger.info("1. Check all camera connections")
        logger.info("2. Reduce number of cameras: --cameras 0 1")
        logger.info("3. Check GPU acceleration: --gpu-acceleration")
        logger.info("4. Try single camera mode: remove --multi-camera")

        return 1

    return 0


def run_traffic_prediction(args):
    """Run traffic prediction system."""
    logger = get_logger("TrafficPrediction")

    try:
        # Initialize predictor
        predictor = TrafficPredictor()

        logger.info("Traffic prediction system started")

        # This would typically run as a service, processing real-time data
        # For demo purposes, we'll just show the system is ready
        logger.info("Traffic predictor ready for real-time data")

        # Cleanup
        predictor.cleanup()

    except Exception as e:
        logger.error(f"Traffic prediction failed: {e}")
        return 1

    return 0


def run_signal_control(args):
    """Run enhanced traffic signal control system with RL and LSTM."""
    logger = get_logger("EnhancedSignalControl")

    try:
        # Determine intersections to control
        intersections = args.intersections if hasattr(args, 'intersections') else ['main']
        if hasattr(args, 'intersection') and args.intersection != 'main':
            intersections = [args.intersection]

        logger.info(f"Initializing enhanced signal control for intersections: {intersections}")

        if hasattr(args, 'mode') and args.mode in ['rl', 'coordinated']:
            try:
                from src.core.enhanced_signal_controller import EnhancedSignalController
                controller = EnhancedSignalController(intersections)
                logger.info(f"Enhanced signal controller initialized with {args.rl_agent if hasattr(args, 'rl_agent') else 'DoubleDQN'} agent")
            except ImportError:
                logger.warning("Enhanced signal controller not available, falling back to basic controller")
                controller = SignalController()
        else:
            # Use basic signal controller
            controller = SignalController()

        logger.info("Traffic signal control system started")

        if hasattr(controller, 'start_real_time_control'):
            controller.start_real_time_control()
        else:
            controller.start_control_system()

        import time
        duration = args.duration if hasattr(args, 'duration') else 60
        logger.info(f"Signal controller running for {duration} seconds... (Press Ctrl+C to stop)")

        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                time.sleep(1)

                if int(time.time()) % 10 == 0:
                    if hasattr(controller, 'get_control_performance'):
                        stats = controller.get_control_performance()
                        logger.info(f"Enhanced control stats: decisions={stats.get('control_metrics', {}).get('decisions_made', 0)}, "
                                  f"avg_time={stats.get('control_metrics', {}).get('average_decision_time', 0):.3f}s")
                    else:
                        stats = controller.get_performance_stats()
                        logger.info(f"Signal control stats: {stats}")

        except KeyboardInterrupt:
            logger.info("Stopping signal control system...")

        if hasattr(controller, 'stop_real_time_control'):
            controller.stop_real_time_control()
        else:
            controller.stop_control_system()
        controller.cleanup()

        logger.info("Enhanced signal control completed successfully")

    except Exception as e:
        logger.error(f"Enhanced signal control failed: {e}")

        # Provide helpful guidance
        logger.info("\n🔧 Troubleshooting suggestions:")
        logger.info("1. Check configuration: config/config.yaml")
        logger.info("2. Install dependencies: pip install tensorflow scikit-learn")
        logger.info("3. Try basic mode: python main.py control --mode automatic")
        logger.info("4. Test single intersection: python main.py control --intersection main")

        return 1

    return 0


def run_full_system(args):
    """Run the complete TMS system."""
    logger = get_logger("FullSystem")

    try:
        logger.info("Starting complete TMS system...")

        # Initialize all components with modern detector
        try:
            # Use modern YOLOv8/YOLOv11 detector by default
            detector = ModernVehicleDetector()
            logger.info(f"Using modern YOLOv8/YOLOv11 detector with model: {detector.model_name}")
        except Exception as detector_error:
            logger.warning(f"Modern detector failed, falling back to legacy: {detector_error}")
            try:
                # Fallback to legacy detector if modern fails
                detector = VehicleDetector()
                logger.info("Using legacy YOLOv4 detector")
            except Exception as legacy_error:
                logger.error(f"Both detectors failed: {legacy_error}")
                logger.info("Running in simulation mode without vehicle detection")
                detector = None

        # Initialize other components
        try:
            predictor = TrafficPredictor()
            logger.info("Traffic predictor initialized")
        except Exception as e:
            logger.warning(f"Traffic predictor failed to initialize: {e}")
            predictor = None

        try:
            # Try enhanced signal controller first
            from src.core.enhanced_signal_controller import EnhancedSignalController
            controller = EnhancedSignalController()
            logger.info("Enhanced signal controller initialized")
        except Exception as e:
            logger.warning(f"Enhanced controller failed, using basic controller: {e}")
            controller = SignalController()
            logger.info("Basic signal controller initialized")

        try:
            data_processor = DataProcessor()
            logger.info("Data processor initialized")
        except Exception as e:
            logger.warning(f"Data processor failed to initialize: {e}")
            data_processor = None

        logger.info("All available components initialized successfully")

        # Start signal control system
        if controller:
            try:
                if hasattr(controller, 'start_control_system'):
                    controller.start_control_system()
                    logger.info("Signal control system started")
                else:
                    logger.warning("Controller does not support start_control_system")
            except Exception as e:
                logger.warning(f"Failed to start signal control: {e}")

        # Launch Smart Traffic Dashboard automatically (unless disabled)
        dashboard_process = None
        if not getattr(args, 'no_dashboard', False):
            logger.info("🚀 Launching Smart Traffic Dashboard...")
            try:
                import subprocess

                # Dashboard path
                dashboard_path = Path(__file__).parent / "src" / "dashboard" / "quick_dashboard.py"

                if dashboard_path.exists():
                    # Get dashboard settings from args
                    dashboard_host = getattr(args, 'dashboard_host', 'localhost')
                    dashboard_port = getattr(args, 'dashboard_port', 8501)

                    # Streamlit command for dashboard
                    dashboard_cmd = [
                        sys.executable, "-m", "streamlit", "run",
                        str(dashboard_path),
                        "--server.address", dashboard_host,
                        "--server.port", str(dashboard_port),
                        "--server.headless", "true",
                        "--browser.gatherUsageStats", "false",
                        "--server.fileWatcherType", "poll",
                        "--server.enableCORS", "false",
                        "--server.enableXsrfProtection", "false"
                    ]

                    # Launch dashboard in background
                    dashboard_process = subprocess.Popen(
                        dashboard_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )

                    # Give dashboard time to start
                    import time
                    time.sleep(3)

                    logger.info("✅ Smart Traffic Dashboard launched successfully!")
                    logger.info(f"🌐 Dashboard URL: http://{dashboard_host}:{dashboard_port}")
                    logger.info("📱 Access from any device on your network")
                else:
                    logger.warning("Smart Traffic Dashboard not found, continuing without dashboard")
            except Exception as dashboard_error:
                logger.warning(f"Failed to launch dashboard: {dashboard_error}")
                logger.info("Continuing with system components only")
        else:
            logger.info("Dashboard launch skipped (--no-dashboard flag used)")

        # Integration demonstration
        logger.info("🎯 Full TMS system running with integrated components...")
        logger.info("🚦 Traffic signal control active")
        if detector:
            logger.info("🚗 Vehicle detection active")
        if predictor:
            logger.info("🧠 Traffic prediction active")
        if data_processor:
            logger.info("📊 Data processing active")

        logger.info("")
        logger.info("🎮 System Features Available:")
        logger.info("  • Real-time vehicle detection and tracking")
        logger.info("  • AI-powered traffic signal optimization")
        logger.info("  • Live traffic analytics and reporting")
        logger.info("  • Interactive dashboard with visual controls")
        logger.info("")
        logger.info("⏹️  Press Ctrl+C to stop the entire system")

        try:
            import time
            while True:
                time.sleep(5)

                # Print system status with safe access
                status_parts = []

                if detector and hasattr(detector, 'get_performance_stats'):
                    try:
                        detector_stats = detector.get_performance_stats()
                        frames = detector_stats.get('frames_processed', 0)
                        status_parts.append(f"Detector: {frames} frames")
                    except Exception:
                        status_parts.append("Detector: active")

                if predictor and hasattr(predictor, 'get_performance_stats'):
                    try:
                        predictor_stats = predictor.get_performance_stats()
                        predictions = predictor_stats.get('predictions_made', 0)
                        status_parts.append(f"Predictor: {predictions} predictions")
                    except Exception:
                        status_parts.append("Predictor: active")

                if controller and hasattr(controller, 'get_performance_stats'):
                    try:
                        controller_stats = controller.get_performance_stats()
                        changes = controller_stats.get('signal_changes', 0)
                        status_parts.append(f"Controller: {changes} signal changes")
                    except Exception:
                        status_parts.append("Controller: active")

                if status_parts:
                    logger.info(f"System status - {', '.join(status_parts)}")
                else:
                    logger.info("System status - All components running in simulation mode")

        except KeyboardInterrupt:
            logger.info("Shutting down TMS system...")

        # Cleanup dashboard process
        if 'dashboard_process' in locals() and dashboard_process and dashboard_process.poll() is None:
            try:
                logger.info("Stopping Smart Traffic Dashboard...")
                dashboard_process.terminate()
                dashboard_process.wait(timeout=5)
                logger.info("Smart Traffic Dashboard stopped")
            except Exception as e:
                logger.warning(f"Error stopping dashboard: {e}")
                try:
                    dashboard_process.kill()
                except Exception:
                    pass

        # Cleanup all components safely
        if controller and hasattr(controller, 'stop_control_system'):
            try:
                controller.stop_control_system()
                logger.info("Signal control system stopped")
            except Exception as e:
                logger.warning(f"Error stopping controller: {e}")

        if data_processor and hasattr(data_processor, 'cleanup'):
            try:
                data_processor.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up data processor: {e}")

        if detector and hasattr(detector, 'cleanup'):
            try:
                detector.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up detector: {e}")

        if predictor and hasattr(predictor, 'cleanup'):
            try:
                predictor.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up predictor: {e}")

        if controller and hasattr(controller, 'cleanup'):
            try:
                controller.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up controller: {e}")

        logger.info("TMS system stopped successfully")

    except Exception as e:
        logger.error(f"Full system failed: {e}")
        return 1

    return 0


def run_dashboard(args):
    """Launch the Smart Traffic Dashboard (quick_dashboard.py)."""
    import subprocess

    logger = get_logger("Dashboard")

    try:
        logger.info("Launching TMS2 Smart Traffic Dashboard...")

        # Check if Streamlit is available
        try:
            import streamlit
        except ImportError:
            logger.error("Streamlit not installed. Install with: pip install streamlit plotly")
            return 1

        # Dashboard path - Updated to use quick_dashboard.py (Smart Traffic Dashboard)
        dashboard_path = Path(__file__).parent / "src" / "dashboard" / "quick_dashboard.py"

        if not dashboard_path.exists():
            logger.error(f"Smart Traffic Dashboard file not found: {dashboard_path}")
            logger.info("Available dashboard files:")
            dashboard_dir = Path(__file__).parent / "src" / "dashboard"
            if dashboard_dir.exists():
                for file in dashboard_dir.glob("*.py"):
                    logger.info(f"  - {file.name}")
            return 1

        # Streamlit command with optimized settings for Smart Traffic Dashboard
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.address", args.host,
            "--server.port", str(args.port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "poll",  # Better compatibility
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ]

        if args.debug:
            cmd.extend(["--logger.level", "debug"])

        logger.info("🚦 Smart Traffic Dashboard Features:")
        logger.info("  • Live video feeds with vehicle detection")
        logger.info("  • Real-time traffic signal visualization")
        logger.info("  • AI-powered traffic management demonstration")
        logger.info("  • 4-way intersection analysis")
        logger.info("  • Environmental impact analytics")
        logger.info("  • Comprehensive session reporting")
        logger.info("")
        logger.info(f"🌐 Dashboard URL: http://{args.host}:{args.port}")
        logger.info("📱 Access from any device on your network")
        logger.info("⏹️  Press Ctrl+C to stop the dashboard")

        # Launch Streamlit
        subprocess.run(cmd, check=True)

        return 0

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch Smart Traffic Dashboard: {e}")
        logger.info("\n🔧 Troubleshooting suggestions:")
        logger.info("1. Check if port is available: netstat -an | findstr :8501")
        logger.info("2. Try different port: python main.py dashboard --port 8502")
        logger.info("3. Check Streamlit installation: pip install streamlit plotly")
        logger.info("4. Verify dashboard file exists: src/dashboard/quick_dashboard.py")
        return 1
    except KeyboardInterrupt:
        logger.info("Smart Traffic Dashboard stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Smart Traffic Dashboard error: {e}")
        return 1


def run_analytics_dashboard(args):
    """Launch the Real-time Analytics Dashboard (real_time_dashboard.py)."""
    import subprocess

    logger = get_logger("AnalyticsDashboard")

    try:
        logger.info("Launching TMS2 Real-time Analytics Dashboard...")

        # Check if Streamlit is available
        try:
            import streamlit
        except ImportError:
            logger.error("Streamlit not installed. Install with: pip install streamlit plotly")
            return 1

        # Dashboard path - Analytics dashboard with trained models
        dashboard_path = Path(__file__).parent / "src" / "dashboard" / "real_time_dashboard.py"

        if not dashboard_path.exists():
            logger.error(f"Analytics Dashboard file not found: {dashboard_path}")
            return 1

        # Streamlit command
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.address", args.host,
            "--server.port", str(args.port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "poll"
        ]

        if args.debug:
            cmd.extend(["--logger.level", "debug"])

        logger.info("📊 Real-time Analytics Dashboard Features:")
        logger.info("  • Trained model performance monitoring")
        logger.info("  • LSTM prediction analytics")
        logger.info("  • RL agent decision tracking")
        logger.info("  • Multi-camera coordination metrics")
        logger.info("  • System performance analytics")
        logger.info("")
        logger.info(f"🌐 Analytics Dashboard URL: http://{args.host}:{args.port}")
        logger.info("⏹️  Press Ctrl+C to stop the analytics dashboard")

        # Launch Streamlit
        subprocess.run(cmd, check=True)

        return 0

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch Analytics Dashboard: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Analytics Dashboard stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Analytics Dashboard error: {e}")
        return 1


def run_training(args):
    """Run AI model training."""
    logger = get_logger("Training")

    try:
        logger.info("Starting TMS2 AI model training...")

        # Import training module
        from src.training.train_models import TMS2ModelTrainer

        # Initialize trainer
        trainer = TMS2ModelTrainer()

        if args.mode in ['data', 'all']:
            # Collect training data
            video_dir = Path(args.video_dir)
            if video_dir.exists():
                video_files = list(video_dir.glob('*.avi'))[:args.max_videos]
                video_paths = [str(f) for f in video_files]

                print(f"🎥 Collecting training data from {len(video_paths)} videos...")
                trainer.collect_training_data(video_paths, args.max_frames)
            else:
                print(f"⚠️ Video directory not found: {video_dir}")

        if args.mode in ['lstm', 'all']:
            # Train LSTM model
            print(f"🧠 Training {args.model_type} LSTM model...")
            lstm_results = trainer.train_lstm_model(args.model_type, args.lstm_epochs)
            print(f"✅ LSTM training completed")

        if args.mode in ['rl', 'all']:
            # Train RL agent
            print(f"🤖 Training RL agent...")
            rl_results = trainer.train_rl_agent(args.rl_episodes)
            print(f"✅ RL training completed")

        # Save trained models
        saved_models = trainer.save_trained_models()
        print(f"💾 Models saved: {list(saved_models.keys())}")

        # Launch training dashboard if requested
        if args.dashboard:
            return run_training_dashboard(args)

        print("🎉 Training pipeline completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        return 1
    finally:
        if 'trainer' in locals():
            trainer.cleanup()


def run_training_dashboard(args):
    """Launch the training dashboard."""
    import subprocess

    logger = get_logger("TrainingDashboard")

    try:
        logger.info("Launching TMS2 Training Dashboard...")

        # Check if Streamlit is available
        try:
            import streamlit
        except ImportError:
            logger.error("Streamlit not installed. Install with: pip install streamlit plotly")
            return 1

        # Dashboard path
        dashboard_path = Path(__file__).parent / "src" / "dashboard" / "training_dashboard.py"

        if not dashboard_path.exists():
            logger.error(f"Training dashboard file not found: {dashboard_path}")
            return 1

        # Streamlit command
        port = getattr(args, 'port', 8502)
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.address", "localhost",
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]

        logger.info(f"Training Dashboard URL: http://localhost:{port}")
        logger.info("Press Ctrl+C to stop the training dashboard")

        # Launch Streamlit
        subprocess.run(cmd, check=True)

        return 0

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch training dashboard: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Training dashboard stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Training dashboard error: {e}")
        return 1


def run_tests(args):
    """Run the test suite."""
    import pytest

    test_args = ['tests/']

    if args.verbose:
        test_args.append('-v')

    if args.coverage:
        test_args.extend(['--cov=src', '--cov-report=html'])

    if args.test_pattern:
        test_args.extend(['-k', args.test_pattern])

    return pytest.main(test_args)


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Traffic Management System (TMS2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py detect --source 0 --display                    # Detect from camera with display
  python main.py detect --source video.mp4                      # Detect from video file
  python main.py run                                            # Run full system with integrated dashboard
  python main.py run --no-dashboard                             # Run full system without dashboard
  python main.py run --dashboard-port 8502                      # Run with dashboard on custom port
  python main.py dashboard                                       # Launch Smart Traffic Dashboard only
  python main.py analytics                                       # Launch Analytics Dashboard (trained models)
  python main.py predict                                         # Run traffic prediction
  python main.py control                                         # Run signal control
  python main.py test --coverage                                # Run tests with coverage
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )

    parser.add_argument(
        '--environment', '-e',
        type=str,
        default='development',
        choices=['development', 'production', 'testing'],
        help='Environment mode'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Vehicle detection command
    detect_parser = subparsers.add_parser('detect', help='Run vehicle detection')
    detect_parser.add_argument('--source', '-s',
                              help='Video source (file path or camera index)')
    detect_parser.add_argument('--display', '-d', action='store_true',
                              help='Display detection results')
    detect_parser.add_argument('--list-cameras', action='store_true',
                              help='List available cameras and exit')
    detect_parser.add_argument('--list-sources', action='store_true',
                              help='List all available video sources and exit')
    detect_parser.add_argument('--use-legacy', action='store_true',
                              help='Force use of legacy YOLOv4 detector')
    detect_parser.add_argument('--model',
                              choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                              help='YOLO model to use (modern detector only)')

    # Phase 2D Multi-Camera Features
    detect_parser.add_argument('--multi-camera', action='store_true',
                              help='Enable multi-camera coordination mode')
    detect_parser.add_argument('--cameras', type=str, nargs='+',
                              default=['0', '1', '2', '3'],
                              help='List of camera indices for multi-camera mode')
    detect_parser.add_argument('--intersection-id', type=str, default='main',
                              help='Intersection identifier for multi-camera setup')
    detect_parser.add_argument('--sync-tolerance', type=float, default=33.0,
                              help='Camera synchronization tolerance in milliseconds')
    detect_parser.add_argument('--fusion-enabled', action='store_true', default=True,
                              help='Enable detection result fusion')
    detect_parser.add_argument('--gpu-acceleration', action='store_true', default=True,
                              help='Enable GPU acceleration for multi-camera processing')

    # Traffic prediction command
    predict_parser = subparsers.add_parser('predict', help='Run traffic prediction')

    # Enhanced signal control command - Phase 2C
    control_parser = subparsers.add_parser('control', help='Enhanced traffic signal control with RL and LSTM')
    control_parser.add_argument('--intersection', type=str, default='main',
                               help='Intersection ID to control')
    control_parser.add_argument('--duration', type=int, default=60,
                               help='Control duration in seconds')
    control_parser.add_argument('--mode', choices=['manual', 'automatic', 'rl', 'coordinated'],
                               default='rl', help='Control mode')
    control_parser.add_argument('--rl-agent', choices=['DQN', 'DoubleDQN', 'DuelingDQN', 'ActorCritic'],
                               default='DoubleDQN', help='RL agent type for intelligent control')
    control_parser.add_argument('--multi-intersection', action='store_true',
                               help='Enable multi-intersection coordination')
    control_parser.add_argument('--lstm-integration', action='store_true', default=True,
                               help='Enable LSTM prediction integration')
    control_parser.add_argument('--intersections', type=str, nargs='+',
                               default=['intersection_1', 'intersection_2', 'intersection_3'],
                               help='List of intersection IDs for coordination')

    # Full system command
    run_parser = subparsers.add_parser('run', help='Run complete TMS system with integrated dashboard')
    run_parser.add_argument('--no-dashboard', action='store_true',
                           help='Skip launching the Smart Traffic Dashboard')
    run_parser.add_argument('--dashboard-port', type=int, default=8501,
                           help='Port for the Smart Traffic Dashboard')
    run_parser.add_argument('--dashboard-host', default='localhost',
                           help='Host for the Smart Traffic Dashboard')

    # Smart Traffic Dashboard command (main dashboard)
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch Smart Traffic Dashboard (main dashboard)')
    dashboard_parser.add_argument('--host', default='localhost',
                                 help='Dashboard host address')
    dashboard_parser.add_argument('--port', type=int, default=8501,
                                 help='Dashboard port')
    dashboard_parser.add_argument('--debug', action='store_true',
                                 help='Enable debug mode')

    # Analytics Dashboard command (trained models dashboard)
    analytics_parser = subparsers.add_parser('analytics', help='Launch Real-time Analytics Dashboard (trained models)')
    analytics_parser.add_argument('--host', default='localhost',
                                 help='Analytics dashboard host address')
    analytics_parser.add_argument('--port', type=int, default=8503,
                                 help='Analytics dashboard port')
    analytics_parser.add_argument('--debug', action='store_true',
                                 help='Enable debug mode')

    # Training command
    train_parser = subparsers.add_parser('train', help='Train AI models')
    train_parser.add_argument('--mode', choices=['data', 'lstm', 'rl', 'all'],
                             default='all', help='Training mode')
    train_parser.add_argument('--video-dir', default='data/kaggle/highway-traffic-videos',
                             help='Directory containing training videos')
    train_parser.add_argument('--max-videos', type=int, default=10,
                             help='Maximum number of videos to process')
    train_parser.add_argument('--max-frames', type=int, default=1000,
                             help='Maximum frames per video')
    train_parser.add_argument('--lstm-epochs', type=int, default=50,
                             help='LSTM training epochs')
    train_parser.add_argument('--rl-episodes', type=int, default=1000,
                             help='RL training episodes')
    train_parser.add_argument('--model-type', default='standard',
                             choices=['standard', 'bidirectional', 'attention'],
                             help='LSTM model type')
    train_parser.add_argument('--dashboard', action='store_true',
                             help='Launch training dashboard')

    # Training dashboard command
    train_dash_parser = subparsers.add_parser('train-dashboard', help='Launch training dashboard')
    train_dash_parser.add_argument('--port', type=int, default=8502, help='Training dashboard port')

    # Test command
    test_parser = subparsers.add_parser('test', help='Run test suite')
    test_parser.add_argument('--verbose', '-v', action='store_true',
                            help='Verbose test output')
    test_parser.add_argument('--coverage', action='store_true',
                            help='Generate coverage report')
    test_parser.add_argument('--test-pattern', '-k',
                            help='Run tests matching pattern')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    setup_system(args.config, args.environment)

    # Route to appropriate function
    if args.command == 'detect':
        return run_vehicle_detection(args)
    elif args.command == 'predict':
        return run_traffic_prediction(args)
    elif args.command == 'control':
        return run_signal_control(args)
    elif args.command == 'run':
        return run_full_system(args)
    elif args.command == 'dashboard':
        return run_dashboard(args)
    elif args.command == 'analytics':
        return run_analytics_dashboard(args)
    elif args.command == 'train':
        return run_training(args)
    elif args.command == 'train-dashboard':
        return run_training_dashboard(args)
    elif args.command == 'test':
        return run_tests(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
