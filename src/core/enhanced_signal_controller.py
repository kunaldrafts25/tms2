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
Enhanced Traffic Signal Controller - Phase 2C

This module provides intelligent traffic signal control using advanced RL algorithms
integrated with LSTM traffic predictions and multi-intersection coordination.

Phase 2C Features:
- Advanced RL signal control (Double DQN, Dueling DQN, Actor-Critic)
- LSTM prediction integration for proactive signal timing
- Multi-intersection coordination with shared learning
- Real-time optimization with sub-200ms decision making
- Environmental impact consideration
- Comprehensive performance monitoring
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import threading
from collections import deque

from ..models.rl_agent import (
    RLAgent, DoubleDQNAgent, DuelingDQNAgent, MultiIntersectionRLCoordinator,
    RLState, RLAction, RLPerformanceMetrics
)
from ..models.lstm_model import RealTimeLSTMPredictor, VehicleDetectionData, PredictionOutput
from ..core.traffic_predictor import EnhancedTrafficPredictor
from ..utils.config_manager import get_config
from ..utils.logger import get_logger, performance_monitor
from ..utils.error_handler import SignalControlError, error_handler

@dataclass
class SignalState:
    """Current signal state for an intersection."""
    intersection_id: str
    current_phase: int  # 0: North-South Green, 1: N-S Yellow, 2: East-West Green, 3: E-W Yellow
    time_in_phase: float
    last_change_time: float
    queue_lengths: List[int]  # [North, South, East, West]
    traffic_density: List[float]  # [North, South, East, West]

@dataclass
class SignalAction:
    """Signal control action."""
    intersection_id: str
    action_type: int  # 0: keep current, 1: change phase, 2: emergency override
    target_phase: int
    duration: float
    priority: int = 1

@dataclass
class ControlDecision:
    """Complete control decision with reasoning."""
    intersection_id: str
    action: SignalAction
    confidence: float
    reasoning: str
    predicted_improvement: float
    environmental_impact: float
    coordination_factor: float

class EnhancedSignalController:
    """
    Enhanced traffic signal controller with RL and LSTM integration.

    Features:
    - Advanced RL algorithms for signal optimization
    - LSTM prediction integration for proactive control
    - Multi-intersection coordination
    - Real-time performance optimization
    - Environmental impact consideration
    """

    def __init__(self, intersection_ids: List[str], config_path: Optional[str] = None):
        """
        Initialize enhanced signal controller.

        Args:
            intersection_ids: List of intersection identifiers
            config_path: Path to configuration file
        """
        self.config = get_config()
        self.logger = get_logger("EnhancedSignalController")

        # Configuration
        self.intersection_ids = intersection_ids
        self.control_enabled = True
        self.real_time_optimization = True

        # Core components
        self.rl_coordinator = MultiIntersectionRLCoordinator(intersection_ids, config_path)
        self.traffic_predictor = EnhancedTrafficPredictor(config_path)

        # Signal state management
        self.signal_states: Dict[str, SignalState] = {}
        self.control_history: Dict[str, List[ControlDecision]] = {}

        # Performance tracking
        self.control_metrics = {
            'decisions_made': 0,
            'average_decision_time': 0.0,
            'total_decision_time': 0.0,
            'coordination_effectiveness': 0.0,
            'environmental_improvement': 0.0
        }

        # Real-time processing
        self._lock = threading.Lock()
        self._control_active = False

        # Initialize system
        self._initialize_signal_states()
        self._integrate_components()

        self.logger.info(f"Enhanced signal controller initialized for {len(intersection_ids)} intersections")

    def _initialize_signal_states(self) -> None:
        """Initialize signal states for all intersections."""
        try:
            current_time = time.time()

            for intersection_id in self.intersection_ids:
                self.signal_states[intersection_id] = SignalState(
                    intersection_id=intersection_id,
                    current_phase=0,  # Start with North-South green
                    time_in_phase=0.0,
                    last_change_time=current_time,
                    queue_lengths=[0, 0, 0, 0],
                    traffic_density=[0.0, 0.0, 0.0, 0.0]
                )

                self.control_history[intersection_id] = []

            self.logger.info("Signal states initialized for all intersections")

        except Exception as e:
            self.logger.error(f"Failed to initialize signal states: {e}")
            raise SignalControlError(f"Signal state initialization failed: {e}")

    def _integrate_components(self) -> None:
        """Integrate RL coordinator with LSTM predictor."""
        try:
            # Integrate LSTM predictor with RL coordinator
            self.rl_coordinator.integrate_lstm_predictor(self.traffic_predictor)

            # Setup multi-intersection coordination
            self.rl_coordinator.setup_multi_intersection_coordination(self.intersection_ids)

            self.logger.info("Component integration completed")

        except Exception as e:
            self.logger.error(f"Component integration failed: {e}")

    def process_detection_data(self, detection_data: VehicleDetectionData) -> None:
        """
        Process vehicle detection data and update signal states.

        Args:
            detection_data: Vehicle detection data from YOLOv8
        """
        try:
            intersection_id = detection_data.intersection_id

            if intersection_id in self.signal_states:
                # Update traffic density based on detection
                signal_state = self.signal_states[intersection_id]

                # Simple mapping of vehicle count to density (can be enhanced)
                total_vehicles = detection_data.vehicle_count
                density_per_direction = total_vehicles / 4.0  # Distribute across 4 directions

                signal_state.traffic_density = [density_per_direction] * 4
                signal_state.queue_lengths = [int(density_per_direction * 5)] * 4  # Estimate queue lengths

                # Process with traffic predictor for LSTM predictions
                prediction = self.traffic_predictor.process_yolo_detection(
                    detection_data, intersection_id
                )

                # Make control decision if needed
                if self.control_enabled:
                    self._evaluate_control_decision(intersection_id, prediction)

        except Exception as e:
            self.logger.error(f"Failed to process detection data: {e}")

    @performance_monitor("SignalControl")
    def _evaluate_control_decision(self, intersection_id: str,
                                 prediction: Optional[PredictionOutput] = None) -> Optional[ControlDecision]:
        """
        Evaluate and make control decision for an intersection.

        Args:
            intersection_id: Intersection to evaluate
            prediction: LSTM prediction output

        Returns:
            Control decision if action is needed
        """
        start_time = time.time()

        try:
            with self._lock:
                signal_state = self.signal_states[intersection_id]
                current_time = time.time()

                # Create RL state with LSTM predictions
                rl_state = self._create_rl_state(intersection_id, prediction)

                # Get all intersection states for coordination
                all_states = {}
                for iid in self.intersection_ids:
                    if iid in self.signal_states:
                        pred = None  # Would get prediction for each intersection
                        all_states[iid] = self._create_rl_state(iid, pred)

                # Get coordinated actions
                coordinated_actions = self.rl_coordinator.get_coordinated_actions(all_states)
                action = coordinated_actions.get(intersection_id, 0)

                # Create control decision
                decision = self._create_control_decision(
                    intersection_id, action, rl_state, prediction
                )

                # Execute decision if significant
                if decision and decision.confidence > 0.6:
                    self._execute_control_decision(decision)

                    # Store in history
                    self.control_history[intersection_id].append(decision)

                    # Keep only recent history
                    if len(self.control_history[intersection_id]) > 100:
                        self.control_history[intersection_id] = self.control_history[intersection_id][-100:]

                # Update metrics
                decision_time = time.time() - start_time
                self.control_metrics['decisions_made'] += 1
                self.control_metrics['total_decision_time'] += decision_time
                self.control_metrics['average_decision_time'] = (
                    self.control_metrics['total_decision_time'] /
                    self.control_metrics['decisions_made']
                )

                return decision

        except Exception as e:
            self.logger.error(f"Control decision evaluation failed: {e}")
            return None

    def _create_rl_state(self, intersection_id: str,
                        prediction: Optional[PredictionOutput] = None) -> RLState:
        """Create RL state from current signal state and predictions."""
        signal_state = self.signal_states[intersection_id]
        current_time = time.time()

        # Extract time features
        hour = time.localtime(current_time).tm_hour
        day_of_week = time.localtime(current_time).tm_wday

        # Time since last change
        time_since_change = current_time - signal_state.last_change_time

        # Create RL state with LSTM predictions
        rl_state = RLState(
            traffic_density=signal_state.traffic_density,
            queue_lengths=signal_state.queue_lengths,
            current_signal_states=[signal_state.current_phase],
            time_since_last_change=[time_since_change],
            time_of_day=hour,
            day_of_week=day_of_week
        )

        # Add LSTM predictions if available
        if prediction is not None:
            rl_state.predicted_traffic = prediction.predicted_values.tolist()
            rl_state.prediction_confidence = [prediction.model_confidence]
            rl_state.environmental_impact = 0.5  # Default, can be calculated

            # Get coordination signals from other intersections
            coordination_signals = {}
            for other_id in self.intersection_ids:
                if other_id != intersection_id:
                    # Simple coordination signal based on traffic density difference
                    other_state = self.signal_states.get(other_id)
                    if other_state:
                        density_diff = (sum(signal_state.traffic_density) -
                                      sum(other_state.traffic_density))
                        coordination_signals[other_id] = max(0.0, min(1.0, density_diff + 0.5))

            rl_state.coordination_signals = coordination_signals

        return rl_state

    def _create_control_decision(self, intersection_id: str, action: int,
                               rl_state: RLState, prediction: Optional[PredictionOutput]) -> ControlDecision:
        """Create control decision from RL action."""
        signal_state = self.signal_states[intersection_id]

        # Map RL action to signal action
        if action == 0:  # Keep current
            signal_action = SignalAction(
                intersection_id=intersection_id,
                action_type=0,
                target_phase=signal_state.current_phase,
                duration=30.0  # Default duration
            )
            reasoning = "Maintaining current signal phase for traffic flow stability"
            confidence = 0.7

        elif action == 1:  # Change phase
            next_phase = (signal_state.current_phase + 1) % 4
            signal_action = SignalAction(
                intersection_id=intersection_id,
                action_type=1,
                target_phase=next_phase,
                duration=25.0
            )
            reasoning = "Changing signal phase to optimize traffic flow"
            confidence = 0.8

        else:  # Emergency or coordination action
            signal_action = SignalAction(
                intersection_id=intersection_id,
                action_type=2,
                target_phase=0,  # Default to North-South green
                duration=20.0,
                priority=2
            )
            reasoning = "Emergency or coordination-driven signal change"
            confidence = 0.9

        # Calculate predicted improvement and environmental impact
        predicted_improvement = 0.1  # Simplified calculation
        environmental_impact = 0.05  # Simplified calculation
        coordination_factor = 0.0

        if rl_state.coordination_signals:
            coordination_factor = sum(rl_state.coordination_signals.values()) / len(rl_state.coordination_signals)

        return ControlDecision(
            intersection_id=intersection_id,
            action=signal_action,
            confidence=confidence,
            reasoning=reasoning,
            predicted_improvement=predicted_improvement,
            environmental_impact=environmental_impact,
            coordination_factor=coordination_factor
        )

    def _execute_control_decision(self, decision: ControlDecision) -> None:
        """Execute the control decision."""
        try:
            intersection_id = decision.intersection_id
            signal_state = self.signal_states[intersection_id]
            action = decision.action

            if action.action_type == 1:  # Phase change
                signal_state.current_phase = action.target_phase
                signal_state.last_change_time = time.time()
                signal_state.time_in_phase = 0.0

                self.logger.info(f"Signal changed for {intersection_id}: Phase {action.target_phase}")

            # In a real system, this would send commands to traffic signal hardware

        except Exception as e:
            self.logger.error(f"Failed to execute control decision: {e}")

    def get_control_performance(self) -> Dict[str, Any]:
        """Get comprehensive control performance metrics."""
        try:
            # Get RL coordination performance
            rl_performance = self.rl_coordinator.get_coordination_performance()

            # Get traffic prediction performance
            prediction_performance = self.traffic_predictor.get_enhanced_performance_stats()

            # Combine with control metrics
            performance = {
                'control_metrics': self.control_metrics,
                'rl_performance': rl_performance,
                'prediction_performance': prediction_performance,
                'signal_states': {
                    iid: {
                        'current_phase': state.current_phase,
                        'time_in_phase': state.time_in_phase,
                        'traffic_density': state.traffic_density,
                        'queue_lengths': state.queue_lengths
                    }
                    for iid, state in self.signal_states.items()
                },
                'recent_decisions': {
                    iid: [
                        {
                            'action_type': decision.action.action_type,
                            'confidence': decision.confidence,
                            'reasoning': decision.reasoning,
                            'coordination_factor': decision.coordination_factor
                        }
                        for decision in history[-5:]  # Last 5 decisions
                    ]
                    for iid, history in self.control_history.items()
                }
            }

            return performance

        except Exception as e:
            self.logger.error(f"Failed to get control performance: {e}")
            return {'error': str(e)}

    def start_real_time_control(self) -> None:
        """Start real-time signal control."""
        self._control_active = True
        self.logger.info("Real-time signal control started")

    def stop_real_time_control(self) -> None:
        """Stop real-time signal control."""
        self._control_active = False
        self.logger.info("Real-time signal control stopped")

    def cleanup(self) -> None:
        """Cleanup controller resources."""
        try:
            self.stop_real_time_control()
            self.rl_coordinator.cleanup()
            self.traffic_predictor.cleanup()

            with self._lock:
                self.signal_states.clear()
                self.control_history.clear()

            self.logger.info("Enhanced signal controller cleaned up")

        except Exception as e:
            self.logger.error(f"Controller cleanup failed: {e}")
