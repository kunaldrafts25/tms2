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
Comprehensive Error Handling System for Traffic Management System

This module provides centralized error handling, custom exceptions,
and recovery mechanisms for robust system operation.
"""

import logging
import traceback
import functools
import time
from typing import Dict, Any, Optional, Callable, Type, Union
from enum import Enum
import sys

class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TMSError(Exception):
    """Base exception class for TMS-specific errors."""

    def __init__(self, message: str, error_code: str = None,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()

class VehicleDetectionError(TMSError):
    """Errors related to vehicle detection operations."""
    pass

class TrafficPredictionError(TMSError):
    """Errors related to traffic prediction operations."""
    pass

class SignalControlError(TMSError):
    """Errors related to traffic signal control operations."""
    pass

class ConfigurationError(TMSError):
    """Errors related to configuration loading and validation."""
    pass

class DataProcessingError(TMSError):
    """Errors related to data processing operations."""
    pass

class ModelLoadingError(TMSError):
    """Errors related to ML model loading and initialization."""
    pass

class CameraConnectionError(TMSError):
    """Errors related to camera/video stream connections."""
    pass

class DatabaseError(TMSError):
    """Errors related to database operations."""
    pass

class ErrorHandler:
    """
    Centralized error handling system with logging, recovery, and alerting.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the error handler.

        Args:
            logger: Logger instance for error reporting
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        self.alert_thresholds: Dict[str, int] = {
            'VehicleDetectionError': 5,
            'CameraConnectionError': 3,
            'ModelLoadingError': 1,
            'DatabaseError': 3
        }

    def register_recovery_strategy(self, exception_type: Type[Exception],
                                 recovery_func: Callable) -> None:
        """
        Register a recovery strategy for a specific exception type.

        Args:
            exception_type: Exception class to handle
            recovery_func: Function to call for recovery
        """
        self.recovery_strategies[exception_type] = recovery_func
        self.logger.info(f"Registered recovery strategy for {exception_type.__name__}")

    def handle_error(self, error: Exception, context: Dict[str, Any] = None,
                    attempt_recovery: bool = True) -> bool:
        """
        Handle an error with logging, recovery, and alerting.

        Args:
            error: Exception to handle
            context: Additional context information
            attempt_recovery: Whether to attempt automatic recovery

        Returns:
            True if error was handled successfully, False otherwise
        """
        error_type = type(error).__name__
        context = context or {}

        # Increment error count
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Log the error
        self._log_error(error, context)

        # Check if alert threshold is reached
        if self._should_alert(error_type):
            self._send_alert(error, context)

        # Attempt recovery if enabled and strategy exists
        if attempt_recovery and type(error) in self.recovery_strategies:
            try:
                recovery_func = self.recovery_strategies[type(error)]
                recovery_result = recovery_func(error, context)

                if recovery_result:
                    self.logger.info(f"Successfully recovered from {error_type}")
                    return True
                else:
                    self.logger.warning(f"Recovery attempt failed for {error_type}")
            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed: {recovery_error}")

        return False

    def _log_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log error with detailed information."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context
        }

        # Add TMS-specific error information if available
        if isinstance(error, TMSError):
            error_info.update({
                'error_code': error.error_code,
                'severity': error.severity.value,
                'error_context': error.context,
                'timestamp': error.timestamp
            })

        # Log based on severity
        if isinstance(error, TMSError):
            if error.severity == ErrorSeverity.CRITICAL:
                self.logger.critical("Critical error occurred", extra=error_info)
            elif error.severity == ErrorSeverity.HIGH:
                self.logger.error("High severity error occurred", extra=error_info)
            elif error.severity == ErrorSeverity.MEDIUM:
                self.logger.warning("Medium severity error occurred", extra=error_info)
            else:
                self.logger.info("Low severity error occurred", extra=error_info)
        else:
            self.logger.error("Unhandled exception occurred", extra=error_info)

    def _should_alert(self, error_type: str) -> bool:
        """Check if error count has reached alert threshold."""
        threshold = self.alert_thresholds.get(error_type, 10)
        return self.error_counts.get(error_type, 0) >= threshold

    def _send_alert(self, error: Exception, context: Dict[str, Any]) -> None:
        """Send alert for critical errors (placeholder for actual implementation)."""
        alert_message = f"TMS Alert: {type(error).__name__} threshold reached"
        self.logger.critical(alert_message, extra={
            'alert_type': 'error_threshold',
            'error_type': type(error).__name__,
            'error_count': self.error_counts.get(type(error).__name__, 0),
            'context': context
        })

        # TODO: Implement actual alerting (email, SMS, webhook, etc.)

    def reset_error_counts(self) -> None:
        """Reset error counts (useful for periodic cleanup)."""
        self.error_counts.clear()
        self.logger.info("Error counts reset")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            'error_counts': self.error_counts.copy(),
            'total_errors': sum(self.error_counts.values()),
            'error_types': list(self.error_counts.keys())
        }

def error_handler(func_or_handler: Union[Callable, ErrorHandler, None] = None,
                 reraise: bool = False, default_return: Any = None):
    """
    Decorator for automatic error handling.

    Args:
        func_or_handler: Function to decorate or ErrorHandler instance to use
        reraise: Whether to reraise the exception after handling
        default_return: Default value to return if error occurs
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Determine the error handler to use
                if isinstance(func_or_handler, ErrorHandler):
                    handler = func_or_handler
                else:
                    handler = _get_default_error_handler()

                # Create context from function info
                context = {
                    'function': func.__name__,
                    'module': getattr(func, '__module__', 'unknown'),
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }

                # Handle the error
                handled = handler.handle_error(e, context)

                if reraise or not handled:
                    raise

                return default_return
        return wrapper

    # Handle different usage patterns
    if callable(func_or_handler) and not isinstance(func_or_handler, ErrorHandler):
        # Used as @error_handler without parentheses
        return decorator(func_or_handler)
    else:
        # Used as @error_handler() with parentheses or with arguments
        return decorator

def safe_execute(func: Callable, *args, error_handler_instance: Optional[ErrorHandler] = None,
                default_return: Any = None, **kwargs) -> Any:
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        *args: Function arguments
        error_handler_instance: ErrorHandler instance to use
        default_return: Default value to return if error occurs
        **kwargs: Function keyword arguments

    Returns:
        Function result or default_return if error occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handler = error_handler_instance or _get_default_error_handler()

        context = {
            'function': func.__name__,
            'module': getattr(func, '__module__', 'unknown'),
            'args_count': len(args),
            'kwargs_keys': list(kwargs.keys())
        }

        handler.handle_error(e, context)
        return default_return

# Recovery strategies
def camera_connection_recovery(error: CameraConnectionError, context: Dict[str, Any]) -> bool:
    """Recovery strategy for camera connection errors."""
    logger = logging.getLogger(__name__)
    logger.info("Attempting camera connection recovery")

    # Wait and retry logic
    time.sleep(2)

    # TODO: Implement actual camera reconnection logic
    # For now, return False to indicate recovery failed
    return False

def model_loading_recovery(error: ModelLoadingError, context: Dict[str, Any]) -> bool:
    """Recovery strategy for model loading errors."""
    logger = logging.getLogger(__name__)
    logger.info("Attempting model loading recovery")

    # TODO: Implement fallback model loading or model download
    return False

# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None

def _get_default_error_handler() -> ErrorHandler:
    """Get or create default error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()

        # Register default recovery strategies
        _global_error_handler.register_recovery_strategy(
            CameraConnectionError, camera_connection_recovery
        )
        _global_error_handler.register_recovery_strategy(
            ModelLoadingError, model_loading_recovery
        )

    return _global_error_handler

def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    return _get_default_error_handler()

def init_error_handler(logger: Optional[logging.Logger] = None) -> ErrorHandler:
    """Initialize global error handler with custom logger."""
    global _global_error_handler
    _global_error_handler = ErrorHandler(logger)
    return _global_error_handler
