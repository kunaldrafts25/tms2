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
Intelligent Traffic Light Controller for 4-Way Intersection
Implements proper traffic light sequencing with safety rules and priority logic.
"""

import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

class TrafficLightState(Enum):
    """Individual traffic light states."""
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"

class IntersectionPhase(Enum):
    """4-way intersection phases with proper sequencing."""
    NORTH_SOUTH_GREEN = "NORTH_SOUTH_GREEN"      # North: GREEN, South: YELLOW, East: RED, West: RED
    NORTH_SOUTH_YELLOW = "NORTH_SOUTH_YELLOW"    # North: YELLOW, South: YELLOW, East: RED, West: RED
    EAST_WEST_GREEN = "EAST_WEST_GREEN"          # North: RED, South: RED, East: GREEN, West: YELLOW
    EAST_WEST_YELLOW = "EAST_WEST_YELLOW"        # North: RED, South: RED, East: YELLOW, West: YELLOW
    ALL_RED = "ALL_RED"                          # All directions: RED (safety clearance)
    EMERGENCY = "EMERGENCY"                      # All directions: RED (emergency override)

@dataclass
class DirectionState:
    """State information for a single direction."""
    direction: str
    current_light: TrafficLightState
    time_remaining: float
    vehicle_count: int = 0
    avg_speed_kmh: float = 0.0
    waiting_time: float = 0.0  # How long this direction has been RED
    priority_score: float = 0.0
    last_green_time: Optional[datetime] = None

@dataclass
class IntersectionState:
    """Complete state of a 4-way intersection."""
    intersection_id: str
    current_phase: IntersectionPhase
    phase_start_time: datetime
    time_remaining: float
    directions: Dict[str, DirectionState] = field(default_factory=dict)
    emergency_mode: bool = False
    manual_override: bool = False
    next_phase: Optional[IntersectionPhase] = None

    def __post_init__(self):
        """Initialize direction states."""
        if not self.directions:
            for direction in ['North', 'South', 'East', 'West']:
                self.directions[direction] = DirectionState(
                    direction=direction,
                    current_light=TrafficLightState.RED,
                    time_remaining=0.0
                )

class IntelligentTrafficController:
    """
    Intelligent traffic light controller with proper sequencing and safety rules.
    """

    # Phase timing configuration (seconds)
    TIMING_CONFIG = {
        'green_min': 15,      # Minimum green time
        'green_max': 60,      # Maximum green time
        'yellow_duration': 4,  # Yellow light duration
        'all_red_duration': 2, # All-red clearance time
        'emergency_duration': 30  # Emergency mode duration
    }

    # Phase sequencing rules
    PHASE_SEQUENCE = {
        IntersectionPhase.NORTH_SOUTH_GREEN: {
            'North': TrafficLightState.GREEN,
            'South': TrafficLightState.YELLOW,
            'East': TrafficLightState.RED,
            'West': TrafficLightState.RED
        },
        IntersectionPhase.NORTH_SOUTH_YELLOW: {
            'North': TrafficLightState.YELLOW,
            'South': TrafficLightState.YELLOW,
            'East': TrafficLightState.RED,
            'West': TrafficLightState.RED
        },
        IntersectionPhase.EAST_WEST_GREEN: {
            'North': TrafficLightState.RED,
            'South': TrafficLightState.RED,
            'East': TrafficLightState.GREEN,
            'West': TrafficLightState.YELLOW
        },
        IntersectionPhase.EAST_WEST_YELLOW: {
            'North': TrafficLightState.RED,
            'South': TrafficLightState.RED,
            'East': TrafficLightState.YELLOW,
            'West': TrafficLightState.YELLOW
        },
        IntersectionPhase.ALL_RED: {
            'North': TrafficLightState.RED,
            'South': TrafficLightState.RED,
            'East': TrafficLightState.RED,
            'West': TrafficLightState.RED
        },
        IntersectionPhase.EMERGENCY: {
            'North': TrafficLightState.RED,
            'South': TrafficLightState.RED,
            'East': TrafficLightState.RED,
            'West': TrafficLightState.RED
        }
    }

    def __init__(self, intersection_id: str = "main"):
        """Initialize the intelligent traffic controller."""
        self.intersection_id = intersection_id
        self.logger = logging.getLogger(f"IntelligentTrafficController_{intersection_id}")

        # State change history for logging (initialize before applying phase)
        self.state_history: List[Dict[str, Any]] = []

        # Time tracking for updates
        self.last_update_time = time.time()

        self.state = IntersectionState(
            intersection_id=intersection_id,
            current_phase=IntersectionPhase.NORTH_SOUTH_GREEN,
            phase_start_time=datetime.now(),
            time_remaining=30.0
        )

        self._apply_phase(self.state.current_phase)

        self.logger.info(f"Intelligent traffic controller initialized for {intersection_id}")

    def update_traffic_data(self, direction: str, vehicle_count: int,
                          avg_speed_kmh: float) -> None:
        """Update traffic data for a specific direction."""
        if direction not in self.state.directions:
            self.logger.warning(f"Unknown direction: {direction}")
            return

        direction_state = self.state.directions[direction]
        direction_state.vehicle_count = vehicle_count
        direction_state.avg_speed_kmh = avg_speed_kmh

        # Update waiting time for RED lights
        if direction_state.current_light == TrafficLightState.RED:
            if direction_state.last_green_time:
                direction_state.waiting_time = (
                    datetime.now() - direction_state.last_green_time
                ).total_seconds()

        # Calculate priority score
        direction_state.priority_score = self._calculate_priority_score(direction_state)

    def _calculate_priority_score(self, direction_state: DirectionState) -> float:
        """Calculate priority score for a direction based on traffic conditions."""
        score = 0.0

        # Vehicle count factor (more vehicles = higher priority)
        score += direction_state.vehicle_count * 2.0

        # Speed factor (slower speeds indicate congestion)
        if direction_state.avg_speed_kmh > 0:
            # Lower speeds get higher priority (congestion needs relief)
            speed_factor = max(0, 50 - direction_state.avg_speed_kmh) / 50.0
            score += speed_factor * 10.0

        # Waiting time factor (longer wait = higher priority)
        waiting_minutes = direction_state.waiting_time / 60.0
        score += waiting_minutes * 5.0

        return score

    def _apply_phase(self, phase: IntersectionPhase) -> None:
        """Apply a traffic light phase to all directions."""
        if phase not in self.PHASE_SEQUENCE:
            self.logger.error(f"Unknown phase: {phase}")
            return

        phase_config = self.PHASE_SEQUENCE[phase]

        for direction, light_state in phase_config.items():
            if direction in self.state.directions:
                direction_state = self.state.directions[direction]
                old_state = direction_state.current_light
                direction_state.current_light = light_state

                # Track when direction goes from GREEN to non-GREEN
                if old_state == TrafficLightState.GREEN and light_state != TrafficLightState.GREEN:
                    direction_state.last_green_time = datetime.now()
                    direction_state.waiting_time = 0.0

        # Log state change
        self._log_state_change(phase)

    def _log_state_change(self, new_phase: IntersectionPhase) -> None:
        """Log traffic light state changes."""
        state_info = {
            'timestamp': datetime.now().isoformat(),
            'intersection_id': self.intersection_id,
            'phase': new_phase.value,
            'directions': {
                direction: {
                    'light': state.current_light.value,
                    'vehicles': state.vehicle_count,
                    'speed': state.avg_speed_kmh,
                    'priority': state.priority_score
                }
                for direction, state in self.state.directions.items()
            }
        }

        self.state_history.append(state_info)

        # Keep only last 100 state changes
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]

        self.logger.info(f"Phase changed to {new_phase.value}")

    def get_direction_state(self, direction: str) -> Optional[DirectionState]:
        """Get the current state for a specific direction."""
        return self.state.directions.get(direction)

    def get_intersection_state(self) -> IntersectionState:
        """Get the complete intersection state."""
        return self.state

    def validate_safety(self) -> bool:
        """Validate that no conflicting directions have GREEN lights."""
        green_directions = [
            direction for direction, state in self.state.directions.items()
            if state.current_light == TrafficLightState.GREEN
        ]

        # Safety rule: No more than one GREEN light at a time
        # OR only opposite directions can be GREEN (North-South or East-West)
        if len(green_directions) == 0:
            return True  # All RED is safe
        elif len(green_directions) == 1:
            return True  # Single GREEN is safe
        elif len(green_directions) == 2:
            # Check if they are opposite directions
            if set(green_directions) in [{'North', 'South'}, {'East', 'West'}]:
                return True

        # More than 2 GREEN or conflicting directions
        self.logger.error(f"SAFETY VIOLATION: Multiple conflicting GREEN lights: {green_directions}")
        return False

    def emergency_override(self) -> None:
        """Activate emergency mode - all lights RED."""
        self.logger.warning("EMERGENCY OVERRIDE ACTIVATED")
        self.state.emergency_mode = True
        self.state.current_phase = IntersectionPhase.EMERGENCY
        self.state.phase_start_time = datetime.now()
        self.state.time_remaining = self.TIMING_CONFIG['emergency_duration']

        self._apply_phase(IntersectionPhase.EMERGENCY)

    def clear_emergency(self) -> None:
        """Clear emergency mode and resume normal operation."""
        if self.state.emergency_mode:
            self.logger.info("Emergency mode cleared, resuming normal operation")
            self.state.emergency_mode = False
            # Transition to ALL_RED first for safety
            self.state.current_phase = IntersectionPhase.ALL_RED
            self.state.phase_start_time = datetime.now()
            self.state.time_remaining = self.TIMING_CONFIG['all_red_duration']
            self._apply_phase(IntersectionPhase.ALL_RED)

    def update(self, dt: float = 1.0) -> bool:
        """
        Update the traffic controller state.

        Args:
            dt: Time delta in seconds

        Returns:
            True if phase changed, False otherwise
        """
        if self.state.emergency_mode:
            # In emergency mode, just count down
            self.state.time_remaining -= dt
            if self.state.time_remaining <= 0:
                self.clear_emergency()
            return False

        # Update time remaining
        self.state.time_remaining -= dt

        # Check if phase should change
        if self.state.time_remaining <= 0:
            return self._advance_phase()

        return False

    def _advance_phase(self) -> bool:
        """Advance to the next traffic light phase based on intelligent logic."""
        current_phase = self.state.current_phase
        next_phase = self._determine_next_phase()

        if next_phase != current_phase:
            duration = self._calculate_phase_duration(next_phase)

            self.state.current_phase = next_phase
            self.state.phase_start_time = datetime.now()
            self.state.time_remaining = duration

            self._apply_phase(next_phase)

            if not self.validate_safety():
                self.emergency_override()
                return True

            return True

        return False

    def _determine_next_phase(self) -> IntersectionPhase:
        """Determine the next phase based on current state and traffic conditions."""
        current = self.state.current_phase

        # Standard phase progression with intelligent modifications
        if current == IntersectionPhase.NORTH_SOUTH_GREEN:
            return IntersectionPhase.NORTH_SOUTH_YELLOW

        elif current == IntersectionPhase.NORTH_SOUTH_YELLOW:
            return IntersectionPhase.ALL_RED

        elif current == IntersectionPhase.ALL_RED:
            # Intelligent decision: which corridor should go next?
            if self._should_prioritize_east_west():
                return IntersectionPhase.EAST_WEST_GREEN
            else:
                return IntersectionPhase.NORTH_SOUTH_GREEN

        elif current == IntersectionPhase.EAST_WEST_GREEN:
            return IntersectionPhase.EAST_WEST_YELLOW

        elif current == IntersectionPhase.EAST_WEST_YELLOW:
            return IntersectionPhase.ALL_RED

        else:  # Emergency or unknown state
            return IntersectionPhase.ALL_RED

    def _should_prioritize_east_west(self) -> bool:
        """Determine if East-West corridor should get priority over North-South."""
        ns_priority = (
            self.state.directions['North'].priority_score +
            self.state.directions['South'].priority_score
        )

        ew_priority = (
            self.state.directions['East'].priority_score +
            self.state.directions['West'].priority_score
        )

        if self.state.current_phase in [IntersectionPhase.NORTH_SOUTH_GREEN, IntersectionPhase.NORTH_SOUTH_YELLOW]:
            # North-South just had green, give slight bias to East-West
            ew_priority += 2.0
        elif self.state.current_phase in [IntersectionPhase.EAST_WEST_GREEN, IntersectionPhase.EAST_WEST_YELLOW]:
            # East-West just had green, give slight bias to North-South
            ns_priority += 2.0

        return ew_priority > ns_priority

    def _calculate_phase_duration(self, phase: IntersectionPhase) -> float:
        """Calculate the duration for a specific phase based on traffic conditions."""
        if phase == IntersectionPhase.ALL_RED:
            return self.TIMING_CONFIG['all_red_duration']

        elif phase in [IntersectionPhase.NORTH_SOUTH_YELLOW, IntersectionPhase.EAST_WEST_YELLOW]:
            return self.TIMING_CONFIG['yellow_duration']

        elif phase == IntersectionPhase.EMERGENCY:
            return self.TIMING_CONFIG['emergency_duration']

        elif phase == IntersectionPhase.NORTH_SOUTH_GREEN:
            # Calculate duration based on North-South traffic
            ns_demand = (
                self.state.directions['North'].priority_score +
                self.state.directions['South'].priority_score
            )
            return self._adaptive_green_duration(ns_demand)

        elif phase == IntersectionPhase.EAST_WEST_GREEN:
            # Calculate duration based on East-West traffic
            ew_demand = (
                self.state.directions['East'].priority_score +
                self.state.directions['West'].priority_score
            )
            return self._adaptive_green_duration(ew_demand)

        else:
            return 30.0  # Default duration

    def _adaptive_green_duration(self, demand_score: float) -> float:
        """Calculate adaptive green duration based on traffic demand."""
        min_duration = self.TIMING_CONFIG['green_min']
        max_duration = self.TIMING_CONFIG['green_max']

        # Normalize demand score to 0-1 range (assuming max score around 50)
        normalized_demand = min(1.0, demand_score / 50.0)

        # Linear interpolation between min and max duration
        duration = min_duration + (max_duration - min_duration) * normalized_demand

        return duration

    def force_phase_change(self, target_phase: IntersectionPhase) -> bool:
        """Force an immediate phase change (manual override)."""
        if self.state.emergency_mode:
            self.logger.warning("Cannot force phase change during emergency mode")
            return False

        self.logger.info(f"Manual override: forcing phase change to {target_phase.value}")
        self.state.manual_override = True

        # Set the target phase
        self.state.current_phase = target_phase
        self.state.phase_start_time = datetime.now()
        self.state.time_remaining = self._calculate_phase_duration(target_phase)

        self._apply_phase(target_phase)

        # Validate safety
        if not self.validate_safety():
            self.emergency_override()
            return False

        return True

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a comprehensive status summary of the intersection."""
        return {
            'intersection_id': self.intersection_id,
            'current_phase': self.state.current_phase.value,
            'time_remaining': self.state.time_remaining,
            'emergency_mode': self.state.emergency_mode,
            'manual_override': self.state.manual_override,
            'directions': {
                direction: {
                    'light': state.current_light.value,
                    'time_remaining': state.time_remaining,
                    'vehicle_count': state.vehicle_count,
                    'avg_speed_kmh': state.avg_speed_kmh,
                    'waiting_time': state.waiting_time,
                    'priority_score': state.priority_score
                }
                for direction, state in self.state.directions.items()
            },
            'safety_valid': self.validate_safety(),
            'next_recommended_phase': self._determine_next_phase().value
        }
