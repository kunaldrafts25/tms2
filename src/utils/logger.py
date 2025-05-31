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
Advanced Logging System for Traffic Management System

This module provides comprehensive logging capabilities with support for
multiple log levels, file rotation, structured logging, and performance monitoring.
"""

import logging
import logging.handlers
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
import traceback
from functools import wraps

class TMSFormatter(logging.Formatter):
    """Custom formatter for TMS logs with structured output."""

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured information."""
        # Base log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else 'Unknown',
                'message': str(record.exc_info[1]) if record.exc_info[1] else '',
                'traceback': traceback.format_exception(*record.exc_info)
            }

        if self.include_extra:
            extra_fields = {k: v for k, v in record.__dict__.items()
                          if k not in ['name', 'msg', 'args', 'levelname', 'levelno',
                                     'pathname', 'filename', 'module', 'lineno',
                                     'funcName', 'created', 'msecs', 'relativeCreated',
                                     'thread', 'threadName', 'processName', 'process',
                                     'getMessage', 'exc_info', 'exc_text', 'stack_info']}
            if extra_fields:
                log_data['extra'] = extra_fields

        return json.dumps(log_data, default=str)

class PerformanceLogger:
    """Performance monitoring and logging utilities."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers: Dict[str, float] = {}

    def start_timer(self, name: str) -> None:
        """Start a performance timer."""
        self.timers[name] = time.time()

    def end_timer(self, name: str, log_level: int = logging.INFO) -> float:
        """End a performance timer and log the duration."""
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0

        duration = time.time() - self.timers[name]
        del self.timers[name]

        self.logger.log(log_level, f"Performance: {name} completed",
                       extra={'duration_seconds': duration, 'operation': name})
        return duration

    def log_memory_usage(self) -> None:
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            self.logger.info("Memory usage", extra={
                'memory_rss_mb': memory_info.rss / 1024 / 1024,
                'memory_vms_mb': memory_info.vms / 1024 / 1024,
                'memory_percent': process.memory_percent()
            })
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")

def performance_monitor(logger_name: Optional[str] = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name or func.__module__)
            perf_logger = PerformanceLogger(logger)

            timer_name = f"{func.__name__}"
            perf_logger.start_timer(timer_name)

            try:
                result = func(*args, **kwargs)
                perf_logger.end_timer(timer_name)
                return result
            except Exception as e:
                perf_logger.end_timer(timer_name, logging.ERROR)
                logger.error(f"Error in {func.__name__}",
                           extra={'function': func.__name__, 'error': str(e)})
                raise
        return wrapper
    return decorator

class Logger:
    """
    Advanced logging system for TMS with multiple handlers and structured logging.
    """

    def __init__(self, name: str = "TMS", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the logger.

        Args:
            name: Logger name
            config: Logging configuration dictionary
        """
        self.name = name
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(name)
        self.performance = PerformanceLogger(self.logger)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_logger()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration."""
        return {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file_path': 'logs/tms.log',
            'max_file_size': '10MB',
            'backup_count': 5,
            'console_output': True,
            'structured_logging': False
        }

    def _setup_logger(self) -> None:
        """Set up logger with handlers and formatters."""
        # Merge with defaults to ensure all keys exist
        default_config = self._get_default_config()
        merged_config = {**default_config, **self.config}
        self.config = merged_config

        # Set logger level
        level = getattr(logging, self.config['level'].upper(), logging.INFO)
        self.logger.setLevel(level)

        # Create logs directory if it doesn't exist
        log_file_path = self.config['file_path']
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # File handler with rotation and UTF-8 encoding
        max_bytes = self._parse_size(self.config['max_file_size'])
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=self.config['backup_count'],
            encoding='utf-8'
        )

        # Use structured logging if enabled
        if self.config.get('structured_logging', False):
            console_formatter = TMSFormatter()
            file_formatter = TMSFormatter()
        else:
            console_formatter = logging.Formatter(self.config['format'])
            file_formatter = logging.Formatter(self.config['format'])

        # Console handler with Unicode support
        if self.config.get('console_output', True):
            try:
                # Configure console handler for Unicode support on Windows
                if sys.platform.startswith('win'):
                    # Try to use UTF-8 encoding for Windows console
                    try:
                        import io
                        console_stream = io.TextIOWrapper(
                            sys.stdout.buffer,
                            encoding='utf-8',
                            errors='replace'
                        )
                        console_handler = logging.StreamHandler(console_stream)
                    except (AttributeError, OSError):
                        # Fallback to standard stdout if buffer access fails
                        console_handler = logging.StreamHandler(sys.stdout)
                else:
                    console_handler = logging.StreamHandler(sys.stdout)

                console_handler.setFormatter(console_formatter)
                self.logger.addHandler(console_handler)
            except (OSError, AttributeError) as e:
                # If console handler fails, continue without it (file logging still works)
                pass

        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def _parse_size(self, size_str: str) -> int:
        """Parse size string (e.g., '10MB') to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(message, extra=kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, extra=kwargs)

    def log_vehicle_detection(self, frame_id: int, vehicle_count: int,
                            processing_time: float, confidence_scores: list) -> None:
        """Log vehicle detection results."""
        self.info("Vehicle detection completed",
                 frame_id=frame_id,
                 vehicle_count=vehicle_count,
                 processing_time_ms=processing_time * 1000,
                 avg_confidence=sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                 operation_type="vehicle_detection")

    def log_traffic_prediction(self, intersection_id: str, predicted_density: float,
                             confidence: float, prediction_horizon: int) -> None:
        """Log traffic prediction results."""
        self.info("Traffic prediction generated",
                 intersection_id=intersection_id,
                 predicted_density=predicted_density,
                 confidence=confidence,
                 prediction_horizon_minutes=prediction_horizon,
                 operation_type="traffic_prediction")

    def log_signal_change(self, intersection_id: str, from_state: str,
                         to_state: str, duration: int, reason: str) -> None:
        """Log traffic signal state changes."""
        self.info("Traffic signal state changed",
                 intersection_id=intersection_id,
                 from_state=from_state,
                 to_state=to_state,
                 duration_seconds=duration,
                 reason=reason,
                 operation_type="signal_control")

    def log_system_performance(self, fps: float, cpu_usage: float,
                             memory_usage: float, gpu_usage: Optional[float] = None) -> None:
        """Log system performance metrics."""
        extra_data = {
            'fps': fps,
            'cpu_usage_percent': cpu_usage,
            'memory_usage_percent': memory_usage,
            'operation_type': 'system_performance'
        }

        if gpu_usage is not None:
            extra_data['gpu_usage_percent'] = gpu_usage

        self.info("System performance metrics", **extra_data)


# Global logger instances
_loggers: Dict[str, Logger] = {}

def get_logger(name: str = "TMS", config: Optional[Dict[str, Any]] = None) -> Logger:
    """Get or create a logger instance."""
    if name not in _loggers:
        _loggers[name] = Logger(name, config)
    return _loggers[name]

def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration globally."""
    logging.basicConfig(
        level=getattr(logging, config.get('level', 'INFO').upper()),
        format=config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )

    get_logger("TMS", config)
