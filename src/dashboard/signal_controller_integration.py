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
Signal Controller Integration for TMS2 Dashboard

This module integrates the EnhancedSignalController with the dashboard
to provide real-time signal control monitoring and demonstration.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
import time
import queue
import json

# Import handling with fallbacks
EnhancedSignalController = None
SignalPhase = None
TrafficSignalSimulator = None
RLDecisionInfo = None

try:
    from src.core.enhanced_signal_controller import (
        EnhancedSignalController, SignalState, SignalAction, ControlDecision
    )
    from src.models.rl_agent import RLState, RLAction
    from src.models.lstm_model import VehicleDetectionData
except ImportError:
    pass

try:
    from src.dashboard.traffic_signal_display import (
        TrafficSignalSimulator, SignalPhase, RLDecisionInfo,
        TrafficSignalState
    )
except ImportError:
    # Fallback for direct execution
    try:
        from traffic_signal_display import (
            TrafficSignalSimulator, SignalPhase, RLDecisionInfo,
            TrafficSignalState
        )
    except ImportError:
        # Create minimal fallback classes
        from enum import Enum

        class SignalPhase(Enum):
            NORTH_SOUTH_GREEN = 0
            NORTH_SOUTH_YELLOW = 1
            EAST_WEST_GREEN = 2
            EAST_WEST_YELLOW = 3
            ALL_RED = 4
            EMERGENCY = 5

        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
            from traffic_signal_display import TrafficSignalSimulator, RLDecisionInfo
        except ImportError:
            st.warning("Signal controller components not available. Using simulation mode.")


class SignalControllerDashboardIntegration:
    """
    Integration layer between EnhancedSignalController and dashboard visualization.

    Features:
    - Real-time signal state monitoring
    - RL decision tracking and visualization
    - Performance metrics collection
    - Manual override capabilities
    - Emergency mode management
    """

    def __init__(self, intersection_ids: List[str]):
        """Initialize the integration layer."""
        self.intersection_ids = intersection_ids
        self.signal_simulator = TrafficSignalSimulator()

        # Enhanced signal controller (if available)
        self.enhanced_controller = None
        self.controller_available = False

        # Data storage
        self.control_decisions_history = []
        self.performance_metrics = {}
        self.real_time_data = queue.Queue(maxsize=1000)

        # Threading for real-time updates
        self._running = False
        self._update_thread = None

        # Initialize controller if available
        self._initialize_controller()

    def _initialize_controller(self):
        """Initialize the enhanced signal controller with comprehensive error handling."""
        try:
            if EnhancedSignalController:
                # Add specific error handling for component integration issues
                try:
                    self.enhanced_controller = EnhancedSignalController(self.intersection_ids)
                    self.controller_available = True
                    st.success("✅ Enhanced Signal Controller initialized successfully!")
                except AttributeError as e:
                    if "setup_multi_intersection_coordination" in str(e):
                        st.warning("⚠️ Multi-intersection coordination method missing. Using basic coordination.")
                        # Try to initialize with basic coordination
                        self._initialize_basic_controller()
                    else:
                        st.error(f"❌ Controller initialization failed: {e}")
                        self._initialize_simulation_mode()
                except Exception as e:
                    st.warning(f"⚠️ Enhanced controller initialization failed: {e}")
                    st.info("🔄 Falling back to simulation mode...")
                    self._initialize_simulation_mode()
            else:
                st.info("ℹ️ Enhanced Signal Controller not available. Using simulation mode.")
                self._initialize_simulation_mode()
        except Exception as e:
            st.error(f"❌ Critical error during controller initialization: {e}")
            self._initialize_simulation_mode()

    def _initialize_basic_controller(self):
        """Initialize basic controller without advanced coordination features."""
        try:
            # Create a simplified version without problematic components
            from src.core.enhanced_signal_controller import EnhancedSignalController

            # Monkey patch the missing method temporarily
            def dummy_setup_coordination(self, intersection_ids):
                self.logger.info(f"Basic coordination setup for {len(intersection_ids)} intersections")
                return True

            # Apply the patch
            EnhancedSignalController.setup_multi_intersection_coordination = dummy_setup_coordination

            self.enhanced_controller = EnhancedSignalController(self.intersection_ids)
            self.controller_available = True
            st.success("✅ Basic Signal Controller initialized successfully!")

        except Exception as e:
            st.warning(f"⚠️ Basic controller initialization failed: {e}")
            self._initialize_simulation_mode()

    def _initialize_simulation_mode(self):
        """Initialize simulation mode when real controller is not available."""
        self.controller_available = False

        for intersection_id in self.intersection_ids:
            phase = np.random.choice([
                SignalPhase.NORTH_SOUTH_GREEN,
                SignalPhase.EAST_WEST_GREEN
            ])

            self.signal_simulator.update_signal_state(
                intersection_id, phase, rl_confidence=np.random.uniform(0.7, 0.9)
            )

    def start_monitoring(self):
        """Start real-time monitoring of signal controller."""
        self._running = True
        self._update_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._update_thread.start()

    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=1.0)

    def _monitoring_loop(self):
        """Main monitoring loop for signal controller."""
        while self._running:
            try:
                if self.controller_available and self.enhanced_controller:
                    self._update_from_real_controller()
                else:
                    self._update_simulation()

                time.sleep(1.0)  # Update every second

            except Exception as e:
                st.error(f"Monitoring error: {e}")
                time.sleep(5.0)

    def _update_from_real_controller(self):
        """Update data from real enhanced signal controller."""
        try:
            for intersection_id in self.intersection_ids:
                if intersection_id in self.enhanced_controller.signal_states:
                    controller_state = self.enhanced_controller.signal_states[intersection_id]

                    # Convert to simulator format
                    phase = self._convert_phase_from_controller(controller_state.current_phase)

                    self.signal_simulator.update_signal_state(
                        intersection_id,
                        phase,
                        rl_confidence=0.8,  # Default confidence
                        manual=False
                    )

                    # Update timing information
                    sim_state = self.signal_simulator.signal_states[intersection_id]
                    sim_state.time_in_phase = controller_state.time_in_phase
                    sim_state.time_remaining = max(0, 30 - controller_state.time_in_phase)

            for intersection_id in self.intersection_ids:
                if intersection_id in self.enhanced_controller.control_history:
                    recent_decisions = self.enhanced_controller.control_history[intersection_id]

                    for decision in recent_decisions[-5:]:  # Last 5 decisions
                        self._add_rl_decision_from_controller(decision)

            # Update performance metrics
            self.performance_metrics = self.enhanced_controller.control_metrics.copy()

        except Exception as e:
            st.error(f"Error updating from real controller: {e}")

    def _update_simulation(self):
        """Update simulation data when real controller is not available."""
        # Simulate RL decisions periodically
        if np.random.random() < 0.1:  # 10% chance per second
            intersection_id = np.random.choice(self.intersection_ids)
            self._simulate_rl_decision(intersection_id)

        # Update signal timings
        for intersection_id in self.intersection_ids:
            state = self.signal_simulator.signal_states[intersection_id]

            # Decrease time remaining
            state.time_remaining = max(0, state.time_remaining - 1)
            state.time_in_phase += 1

            # Auto-advance phases when time expires
            if state.time_remaining <= 0 and not state.manual_override:
                self._advance_to_next_phase(intersection_id)

        # Update performance metrics
        self.performance_metrics = {
            'decisions_made': len(self.control_decisions_history),
            'average_decision_time': np.random.uniform(0.05, 0.15),
            'total_decision_time': len(self.control_decisions_history) * 0.1,
            'coordination_effectiveness': np.random.uniform(0.7, 0.9),
            'environmental_improvement': np.random.uniform(0.1, 0.3)
        }

    def _convert_phase_from_controller(self, controller_phase: int) -> SignalPhase:
        """Convert controller phase to simulator phase."""
        phase_mapping = {
            0: SignalPhase.NORTH_SOUTH_GREEN,
            1: SignalPhase.NORTH_SOUTH_YELLOW,
            2: SignalPhase.EAST_WEST_GREEN,
            3: SignalPhase.EAST_WEST_YELLOW,
            4: SignalPhase.ALL_RED
        }
        return phase_mapping.get(controller_phase, SignalPhase.NORTH_SOUTH_GREEN)

    def _add_rl_decision_from_controller(self, decision: 'ControlDecision'):
        """Add RL decision from controller to simulator."""
        # Convert controller decision to simulator format
        rl_decision = RLDecisionInfo(
            intersection_id=decision.intersection_id,
            timestamp=datetime.now(),
            current_state={
                'action_type': decision.action.action_type,
                'target_phase': decision.action.target_phase,
                'duration': decision.action.duration
            },
            action_taken=decision.action.action_type,
            action_description=self._get_action_description(decision.action.action_type),
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            predicted_improvement=decision.predicted_improvement,
            q_values=[0.5, 0.7, 0.3],  # Simulated Q-values
            reward=np.random.uniform(-2, 10),
            environmental_impact=decision.environmental_impact
        )

        self.signal_simulator.add_rl_decision(rl_decision)
        self.control_decisions_history.append(decision)

    def _simulate_rl_decision(self, intersection_id: str):
        """Simulate an RL decision for demonstration."""
        from .traffic_signal_display import simulate_rl_decision
        simulate_rl_decision(self.signal_simulator, intersection_id)

    def _advance_to_next_phase(self, intersection_id: str):
        """Advance signal to next phase automatically."""
        current_state = self.signal_simulator.signal_states[intersection_id]
        current_phase = current_state.current_phase

        # Define phase progression
        next_phase_map = {
            SignalPhase.NORTH_SOUTH_GREEN: SignalPhase.NORTH_SOUTH_YELLOW,
            SignalPhase.NORTH_SOUTH_YELLOW: SignalPhase.EAST_WEST_GREEN,
            SignalPhase.EAST_WEST_GREEN: SignalPhase.EAST_WEST_YELLOW,
            SignalPhase.EAST_WEST_YELLOW: SignalPhase.NORTH_SOUTH_GREEN,
            SignalPhase.ALL_RED: SignalPhase.NORTH_SOUTH_GREEN
        }

        next_phase = next_phase_map.get(current_phase, SignalPhase.NORTH_SOUTH_GREEN)

        self.signal_simulator.update_signal_state(
            intersection_id,
            next_phase,
            rl_confidence=np.random.uniform(0.7, 0.9)
        )

    def _get_action_description(self, action_type: int) -> str:
        """Get human-readable action description."""
        action_descriptions = {
            0: "Maintain current signal phase",
            1: "Change to next phase",
            2: "Emergency override activated"
        }
        return action_descriptions.get(action_type, "Unknown action")

    def process_vehicle_detection(self, detection_data: Dict[str, Any]):
        """Process vehicle detection data and update signal controller."""
        try:
            if self.controller_available and self.enhanced_controller:
                # Extract confidence scores and calculate average
                confidence_scores = detection_data.get('confidence_scores', [])
                avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.8

                # Estimate vehicle types (simplified for now)
                vehicle_count = detection_data.get('vehicle_count', 0)
                vehicle_types = {
                    'car': max(0, vehicle_count - 2),  # Most vehicles are cars
                    'truck': min(2, vehicle_count // 3),  # Some trucks
                    'bus': min(1, vehicle_count // 5)   # Occasional buses
                }

                # Calculate traffic density (vehicles per unit area)
                frame_area = 640 * 480  # Default frame size
                traffic_density = vehicle_count / (frame_area / 10000)  # per 100x100 area

                # Convert to VehicleDetectionData format (matching LSTM model structure)
                vehicle_data = VehicleDetectionData(
                    timestamp=detection_data.get('timestamp', time.time()),
                    vehicle_count=vehicle_count,
                    vehicle_types=vehicle_types,
                    avg_confidence=avg_confidence,
                    traffic_density=traffic_density,
                    frame_id=detection_data.get('frame_id', 0),
                    intersection_id=detection_data.get('intersection_id', 'main')
                )

                # Process with enhanced controller
                self.enhanced_controller.process_detection_data(vehicle_data)

        except Exception as e:
            st.error(f"Error processing vehicle detection: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")

    def manual_override(self, intersection_id: str, phase: SignalPhase):
        """Apply manual override to signal."""
        try:
            if self.controller_available and self.enhanced_controller:
                # This would need to be implemented in the enhanced controller
                pass

            # Always update simulator
            self.signal_simulator.update_signal_state(
                intersection_id, phase, manual=True, rl_confidence=0.0
            )

        except Exception as e:
            st.error(f"Error applying manual override: {e}")

    def emergency_mode(self, intersection_id: str, activate: bool):
        """Activate or deactivate emergency mode."""
        try:
            if activate:
                phase = SignalPhase.EMERGENCY
                self.signal_simulator.update_signal_state(
                    intersection_id, phase, manual=True, rl_confidence=0.0
                )
                self.signal_simulator.signal_states[intersection_id].emergency_mode = True
            else:
                phase = SignalPhase.NORTH_SOUTH_GREEN
                self.signal_simulator.update_signal_state(
                    intersection_id, phase, manual=False, rl_confidence=0.8
                )
                self.signal_simulator.signal_states[intersection_id].emergency_mode = False
                self.signal_simulator.signal_states[intersection_id].manual_override = False

        except Exception as e:
            st.error(f"Error managing emergency mode: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for dashboard display."""
        return {
            'total_intersections': len(self.intersection_ids),
            'active_intersections': len([s for s in self.signal_simulator.signal_states.values()
                                       if not s.emergency_mode]),
            'manual_overrides': len([s for s in self.signal_simulator.signal_states.values()
                                   if s.manual_override]),
            'emergency_modes': len([s for s in self.signal_simulator.signal_states.values()
                                  if s.emergency_mode]),
            'recent_decisions': len(self.signal_simulator.rl_decisions),
            'controller_available': self.controller_available,
            'performance_metrics': self.performance_metrics
        }

    def get_recent_decisions_df(self, minutes: int = 10) -> pd.DataFrame:
        """Get recent RL decisions as DataFrame."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        recent_decisions = [
            {
                'timestamp': d.timestamp,
                'intersection_id': d.intersection_id,
                'action': d.action_description,
                'confidence': d.confidence,
                'reward': d.reward,
                'improvement': d.predicted_improvement,
                'environmental_impact': d.environmental_impact
            }
            for d in self.signal_simulator.rl_decisions
            if d.timestamp >= cutoff_time
        ]

        return pd.DataFrame(recent_decisions)

    def cleanup(self):
        """Cleanup resources."""
        self.stop_monitoring()

        if self.enhanced_controller:
            try:
                self.enhanced_controller.cleanup()
            except:
                pass
