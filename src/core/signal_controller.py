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
Intelligent Traffic Signal Control System

This module provides advanced traffic signal control using reinforcement learning,
rule-based algorithms, and real-time traffic data for optimal traffic flow management.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

from ..utils.config_manager import get_config
from ..utils.logger import get_logger, performance_monitor
from ..utils.error_handler import (
    SignalControlError, error_handler, safe_execute
)

class SignalState(Enum):
    """Traffic signal states."""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    ALL_RED = "all_red"

class ControlMode(Enum):
    """Signal control modes."""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"

@dataclass
class SignalTiming:
    """Data class for signal timing configuration."""
    green_duration: int
    yellow_duration: int
    red_duration: int
    all_red_duration: int

@dataclass
class IntersectionState:
    """Data class for intersection state information."""
    intersection_id: str
    current_state: SignalState
    time_remaining: int
    last_change_time: float
    traffic_density: float
    queue_length: int
    control_mode: ControlMode

@dataclass
class SignalChangeEvent:
    """Data class for signal change events."""
    intersection_id: str
    from_state: SignalState
    to_state: SignalState
    timestamp: float
    reason: str
    duration: int

class SignalController:
    """
    Intelligent traffic signal control system with multiple control strategies.
    
    Features:
    - Rule-based signal control
    - Reinforcement learning optimization
    - Emergency vehicle priority
    - Manual override capabilities
    - Real-time traffic adaptation
    - Multi-intersection coordination
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the signal controller.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = get_config()
        self.logger = get_logger("SignalController")
        
        # Load intersection configurations
        self.intersections = self._load_intersection_config()
        
        # Signal timing constraints
        self.min_green = self.config.get('traffic_signals.timing_constraints.min_green', 15)
        self.max_green = self.config.get('traffic_signals.timing_constraints.max_green', 120)
        self.yellow_duration = self.config.get('traffic_signals.timing_constraints.yellow_duration', 5)
        self.all_red_duration = self.config.get('traffic_signals.timing_constraints.all_red_duration', 3)

        self.intersection_states: Dict[str, IntersectionState] = {}
        self.signal_history: List[SignalChangeEvent] = []
        self.traffic_data: Dict[str, List[Dict[str, Any]]] = {}

        self.control_mode = ControlMode.AUTOMATIC
        self.emergency_active = False
        self.manual_overrides: Dict[str, SignalState] = {}

        self.signal_changes = 0
        self.total_wait_time = 0.0

        self._lock = threading.Lock()
        self._running = False
        self._control_thread = None

        self._initialize_intersections()
    
    def _load_intersection_config(self) -> Dict[str, Dict[str, Any]]:
        """Load intersection configurations from config."""
        intersections_config = self.config.get('traffic_signals.intersections', [])
        intersections = {}
        
        for intersection in intersections_config:
            intersection_id = intersection.get('id')
            if intersection_id:
                intersections[intersection_id] = intersection
        
        return intersections
    
    def _initialize_intersections(self) -> None:
        """Initialize intersection states."""
        for intersection_id, config in self.intersections.items():
            default_timing = config.get('default_timing', [30, 5, 25, 5])
            
            self.intersection_states[intersection_id] = IntersectionState(
                intersection_id=intersection_id,
                current_state=SignalState.RED,
                time_remaining=default_timing[2],  # Start with red
                last_change_time=time.time(),
                traffic_density=0.0,
                queue_length=0,
                control_mode=ControlMode.AUTOMATIC
            )
            
            # Initialize traffic data storage
            self.traffic_data[intersection_id] = []
        
        self.logger.info(f"Initialized {len(self.intersections)} intersections")
    
    def start_control_system(self) -> None:
        """Start the automatic signal control system."""
        if self._running:
            self.logger.warning("Control system is already running")
            return
        
        self._running = True
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        
        self.logger.info("Signal control system started")
    
    def stop_control_system(self) -> None:
        """Stop the automatic signal control system."""
        self._running = False
        
        if self._control_thread and self._control_thread.is_alive():
            self._control_thread.join(timeout=5.0)
        
        self.logger.info("Signal control system stopped")
    
    def _control_loop(self) -> None:
        """Main control loop for automatic signal management."""
        while self._running:
            try:
                with self._lock:
                    current_time = time.time()
                    
                    for intersection_id in self.intersections:
                        self._update_intersection_state(intersection_id, current_time)
                
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in control loop: {e}")
                time.sleep(1.0)  # Continue after error
    
    @error_handler(reraise=False)
    def _update_intersection_state(self, intersection_id: str, current_time: float) -> None:
        """Update the state of a specific intersection."""
        state = self.intersection_states[intersection_id]
        
        # Check for manual override
        if intersection_id in self.manual_overrides:
            target_state = self.manual_overrides[intersection_id]
            if state.current_state != target_state:
                self._change_signal_state(intersection_id, target_state, "manual_override")
            return
        
        # Check for emergency mode
        if self.emergency_active:
            self._handle_emergency_mode(intersection_id)
            return
        
        # Decrement time remaining
        time_elapsed = current_time - state.last_change_time
        state.time_remaining = max(0, state.time_remaining - int(time_elapsed))
        state.last_change_time = current_time
        
        # Check if state change is needed
        if state.time_remaining <= 0:
            next_state = self._calculate_next_state(intersection_id)
            duration = self._calculate_signal_duration(intersection_id, next_state)
            self._change_signal_state(intersection_id, next_state, "automatic", duration)
    
    def _calculate_next_state(self, intersection_id: str) -> SignalState:
        """Calculate the next signal state based on current state and traffic conditions."""
        current_state = self.intersection_states[intersection_id].current_state
        
        # Standard signal cycle: GREEN -> YELLOW -> RED -> ALL_RED -> GREEN
        if current_state == SignalState.GREEN:
            return SignalState.YELLOW
        elif current_state == SignalState.YELLOW:
            return SignalState.RED
        elif current_state == SignalState.RED:
            return SignalState.ALL_RED
        elif current_state == SignalState.ALL_RED:
            return SignalState.GREEN
        else:
            return SignalState.GREEN  # Default fallback
    
    def _calculate_signal_duration(self, intersection_id: str, 
                                 signal_state: SignalState) -> int:
        """Calculate signal duration based on traffic conditions and constraints."""
        state = self.intersection_states[intersection_id]
        config = self.intersections[intersection_id]
        default_timing = config.get('default_timing', [30, 5, 25, 5])
        
        if signal_state == SignalState.GREEN:
            # Adaptive green timing based on traffic density
            base_duration = default_timing[0]
            
            # Adjust based on traffic density (0.0 to 1.0)
            density_factor = min(2.0, max(0.5, 1.0 + state.traffic_density))
            adaptive_duration = int(base_duration * density_factor)
            
            # Apply constraints
            return max(self.min_green, min(self.max_green, adaptive_duration))
            
        elif signal_state == SignalState.YELLOW:
            return self.yellow_duration
        elif signal_state == SignalState.RED:
            return default_timing[2]
        elif signal_state == SignalState.ALL_RED:
            return self.all_red_duration
        else:
            return default_timing[0]
    
    @performance_monitor("SignalController")
    def _change_signal_state(self, intersection_id: str, new_state: SignalState, 
                           reason: str, duration: Optional[int] = None) -> None:
        """Change the signal state for an intersection."""
        try:
            state = self.intersection_states[intersection_id]
            old_state = state.current_state
            
            if duration is None:
                duration = self._calculate_signal_duration(intersection_id, new_state)
            
            state.current_state = new_state
            state.time_remaining = duration
            state.last_change_time = time.time()
            
            event = SignalChangeEvent(
                intersection_id=intersection_id,
                from_state=old_state,
                to_state=new_state,
                timestamp=time.time(),
                reason=reason,
                duration=duration
            )
            
            # Store event
            self.signal_history.append(event)
            
            # Keep only recent history (last 1000 events)
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            self.signal_changes += 1
            
            # Log the change
            self.logger.log_signal_change(
                intersection_id, old_state.value, new_state.value, duration, reason
            )
            
        except Exception as e:
            self.logger.error(f"Failed to change signal state: {e}")
            raise SignalControlError(f"Signal state change failed: {e}")
    
    def update_traffic_data(self, intersection_id: str, vehicle_count: int, 
                          avg_speed: float, queue_length: int = 0) -> None:
        """
        Update traffic data for an intersection.
        
        Args:
            intersection_id: ID of the intersection
            vehicle_count: Number of vehicles detected
            avg_speed: Average vehicle speed
            queue_length: Length of vehicle queue
        """
        if intersection_id not in self.intersection_states:
            self.logger.warning(f"Unknown intersection ID: {intersection_id}")
            return
        
        with self._lock:
            max_vehicles = 50  # Adjust based on intersection capacity
            density = min(1.0, vehicle_count / max_vehicles)
            
            state = self.intersection_states[intersection_id]
            state.traffic_density = density
            state.queue_length = queue_length
            
            # Store traffic data
            traffic_point = {
                'timestamp': time.time(),
                'vehicle_count': vehicle_count,
                'avg_speed': avg_speed,
                'density': density,
                'queue_length': queue_length
            }
            
            self.traffic_data[intersection_id].append(traffic_point)
            
            # Keep only recent data (last hour)
            cutoff_time = time.time() - 3600
            self.traffic_data[intersection_id] = [
                point for point in self.traffic_data[intersection_id]
                if point['timestamp'] > cutoff_time
            ]
        
        self.logger.debug(f"Updated traffic data for intersection {intersection_id}: "
                         f"vehicles={vehicle_count}, density={density:.2f}")
    
    def set_manual_override(self, intersection_id: str, signal_state: SignalState) -> None:
        """
        Set manual override for an intersection.
        
        Args:
            intersection_id: ID of the intersection
            signal_state: Desired signal state
        """
        if intersection_id not in self.intersection_states:
            raise SignalControlError(f"Unknown intersection ID: {intersection_id}")
        
        with self._lock:
            self.manual_overrides[intersection_id] = signal_state
            self.intersection_states[intersection_id].control_mode = ControlMode.MANUAL
        
        self.logger.info(f"Manual override set for intersection {intersection_id}: {signal_state.value}")
    
    def clear_manual_override(self, intersection_id: str) -> None:
        """Clear manual override for an intersection."""
        with self._lock:
            if intersection_id in self.manual_overrides:
                del self.manual_overrides[intersection_id]
                self.intersection_states[intersection_id].control_mode = ControlMode.AUTOMATIC
        
        self.logger.info(f"Manual override cleared for intersection {intersection_id}")
    
    def activate_emergency_mode(self) -> None:
        """Activate emergency mode for all intersections."""
        with self._lock:
            self.emergency_active = True
            for intersection_id in self.intersection_states:
                self.intersection_states[intersection_id].control_mode = ControlMode.EMERGENCY
        
        self.logger.warning("Emergency mode activated")
    
    def deactivate_emergency_mode(self) -> None:
        """Deactivate emergency mode."""
        with self._lock:
            self.emergency_active = False
            for intersection_id in self.intersection_states:
                if self.intersection_states[intersection_id].control_mode == ControlMode.EMERGENCY:
                    self.intersection_states[intersection_id].control_mode = ControlMode.AUTOMATIC
        
        self.logger.info("Emergency mode deactivated")
    
    def _handle_emergency_mode(self, intersection_id: str) -> None:
        """Handle emergency mode logic for an intersection."""
        # In emergency mode, prioritize main roads (simplified logic)
        # This would be more sophisticated in a real implementation
        state = self.intersection_states[intersection_id]
        
        if state.current_state != SignalState.GREEN:
            self._change_signal_state(intersection_id, SignalState.GREEN, "emergency_mode", 60)
    
    def get_intersection_state(self, intersection_id: str) -> Optional[IntersectionState]:
        """Get current state of an intersection."""
        return self.intersection_states.get(intersection_id)
    
    def get_all_intersection_states(self) -> Dict[str, IntersectionState]:
        """Get current states of all intersections."""
        return self.intersection_states.copy()
    
    def get_signal_history(self, intersection_id: Optional[str] = None, 
                          limit: int = 100) -> List[SignalChangeEvent]:
        """Get signal change history."""
        if intersection_id:
            history = [event for event in self.signal_history 
                      if event.intersection_id == intersection_id]
        else:
            history = self.signal_history
        
        return history[-limit:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_intersections = len(self.intersections)
        active_intersections = len([s for s in self.intersection_states.values() 
                                  if s.control_mode == ControlMode.AUTOMATIC])
        
        return {
            'total_intersections': total_intersections,
            'active_intersections': active_intersections,
            'signal_changes': self.signal_changes,
            'emergency_mode_active': self.emergency_active,
            'manual_overrides': len(self.manual_overrides),
            'control_system_running': self._running
        }
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.stop_control_system()
        
        with self._lock:
            self.intersection_states.clear()
            self.signal_history.clear()
            self.traffic_data.clear()
            self.manual_overrides.clear()
        
        self.logger.info("SignalController resources cleaned up")
