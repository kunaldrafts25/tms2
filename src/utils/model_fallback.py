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

Model Fallback System for Streamlit Cloud Deployment

Provides fallback functionality when trained models are not available,
creating realistic simulation data for demonstration purposes.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import random

@dataclass
class SimulatedTrafficData:
    """Simulated traffic data for demonstration."""
    vehicle_count: int
    avg_speed: float
    traffic_density: float
    congestion_level: str
    timestamp: datetime
    intersection_id: str = "main"

class ModelFallbackManager:
    """
    Manages fallback functionality when trained models are unavailable.
    Provides realistic simulation data for dashboard demonstration.
    """
    
    def __init__(self):
        self.simulation_start_time = time.time()
        self.base_traffic_patterns = self._generate_base_patterns()
        self.lstm_simulation_data = self._generate_lstm_simulation()
        self.rl_simulation_data = self._generate_rl_simulation()
        
    def _generate_base_patterns(self) -> Dict[str, List[float]]:
        """Generate base traffic patterns for different times of day."""
        # Simulate realistic traffic patterns
        hours = list(range(24))
        
        # Rush hour patterns (morning and evening peaks)
        morning_peak = [0.3, 0.2, 0.1, 0.1, 0.2, 0.4, 0.8, 0.9, 0.7, 0.5]  # 6-10 AM
        midday = [0.4, 0.4, 0.3, 0.3, 0.4]  # 10 AM - 3 PM
        evening_peak = [0.5, 0.7, 0.9, 0.8, 0.6, 0.4]  # 3-9 PM
        night = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2]  # 9 PM - 6 AM
        
        daily_pattern = night[-3:] + morning_peak + midday + evening_peak + night[:-3]
        
        return {
            'hourly_density': daily_pattern,
            'speed_multiplier': [1.2 - d for d in daily_pattern],  # Inverse relationship
            'congestion_probability': [d * 0.8 for d in daily_pattern]
        }
    
    def _generate_lstm_simulation(self) -> Dict[str, Any]:
        """Generate LSTM model simulation data."""
        return {
            'model_type': 'LSTM Transformer',
            'training_sessions': '20250531_015149',
            'training_scale': {
                'videos': 254,
                'frames': 13317,
                'training_time': '1h 24min',
                'final_loss': 2.24e-12,
                'validation_accuracy': 0.9847
            },
            'prediction_horizon': 5,  # minutes
            'confidence_threshold': 0.85,
            'last_update': datetime.now()
        }
    
    def _generate_rl_simulation(self) -> Dict[str, Any]:
        """Generate RL agent simulation data."""
        return {
            'agent_type': 'Multi-Intersection DQN',
            'training_episodes': 1000,
            'final_reward': 49.92,
            'exploration_rate': 0.01,
            'learning_rate': 0.001,
            'experience_buffer_size': 10000,
            'last_update': datetime.now()
        }
    
    def get_current_traffic_simulation(self, intersection_id: str = "main") -> SimulatedTrafficData:
        """
        Get current simulated traffic data.
        
        Args:
            intersection_id: Intersection identifier
            
        Returns:
            Simulated traffic data
        """
        current_time = datetime.now()
        hour = current_time.hour
        
        # Get base pattern for current hour
        base_density = self.base_traffic_patterns['hourly_density'][hour]
        base_speed = self.base_traffic_patterns['speed_multiplier'][hour]
        
        # Add some randomness and time-based variation
        time_factor = np.sin(time.time() * 0.1) * 0.1  # Slow oscillation
        random_factor = (random.random() - 0.5) * 0.2  # Random variation
        
        # Calculate current values
        traffic_density = max(0.1, min(1.0, base_density + time_factor + random_factor))
        avg_speed = max(10, min(80, 50 * base_speed + random_factor * 10))
        vehicle_count = int(traffic_density * 25 + random.randint(-3, 3))
        
        # Determine congestion level
        if traffic_density < 0.3:
            congestion_level = "Low"
        elif traffic_density < 0.7:
            congestion_level = "Medium"
        else:
            congestion_level = "High"
        
        return SimulatedTrafficData(
            vehicle_count=vehicle_count,
            avg_speed=avg_speed,
            traffic_density=traffic_density,
            congestion_level=congestion_level,
            timestamp=current_time,
            intersection_id=intersection_id
        )
    
    def get_lstm_prediction(self, horizon_minutes: int = 5) -> Dict[str, Any]:
        """
        Simulate LSTM traffic prediction.
        
        Args:
            horizon_minutes: Prediction horizon in minutes
            
        Returns:
            Simulated prediction data
        """
        current_data = self.get_current_traffic_simulation()
        
        # Simulate prediction with some trend
        predictions = []
        for i in range(horizon_minutes):
            # Add trend and noise
            trend = np.sin((time.time() + i * 60) * 0.05) * 0.1
            noise = (random.random() - 0.5) * 0.05
            
            predicted_density = max(0.1, min(1.0, 
                current_data.traffic_density + trend + noise))
            
            predictions.append({
                'time_offset': i + 1,
                'predicted_density': predicted_density,
                'confidence': 0.85 + random.random() * 0.1,
                'timestamp': current_data.timestamp + timedelta(minutes=i+1)
            })
        
        return {
            'model_info': self.lstm_simulation_data,
            'current_state': current_data,
            'predictions': predictions,
            'model_confidence': 0.89,
            'prediction_accuracy': 0.92
        }
    
    def get_rl_decision(self, traffic_data: Optional[SimulatedTrafficData] = None) -> Dict[str, Any]:
        """
        Simulate RL agent signal control decision.
        
        Args:
            traffic_data: Current traffic data (optional)
            
        Returns:
            Simulated RL decision
        """
        if traffic_data is None:
            traffic_data = self.get_current_traffic_simulation()
        
        # Simulate RL decision based on traffic density
        if traffic_data.traffic_density < 0.3:
            recommended_phase = "NORTH_SOUTH_GREEN"
            phase_duration = 45
            confidence = 0.92
        elif traffic_data.traffic_density < 0.7:
            recommended_phase = "EAST_WEST_GREEN"
            phase_duration = 35
            confidence = 0.87
        else:
            recommended_phase = "ADAPTIVE_TIMING"
            phase_duration = 25
            confidence = 0.94
        
        return {
            'agent_info': self.rl_simulation_data,
            'recommended_action': {
                'phase': recommended_phase,
                'duration': phase_duration,
                'priority_direction': 'north_south' if traffic_data.traffic_density > 0.6 else 'east_west'
            },
            'decision_confidence': confidence,
            'q_values': {
                'north_south_green': 0.85 + random.random() * 0.1,
                'east_west_green': 0.78 + random.random() * 0.1,
                'all_red': 0.45 + random.random() * 0.1
            },
            'reasoning': f"High traffic density ({traffic_data.traffic_density:.2f}) detected, optimizing for flow"
        }
    
    def get_training_metrics_simulation(self) -> Dict[str, Any]:
        """Get simulated training metrics for display."""
        return {
            'lstm': {
                'training_results': {
                    'loss': [0.5, 0.3, 0.15, 0.08, 0.04, 2.24e-12],
                    'val_loss': [0.52, 0.31, 0.16, 0.09, 0.045, 2.8e-12],
                    'epochs': list(range(1, 7))
                },
                'final_metrics': {
                    'mse': 2.24e-12,
                    'mae': 1.5e-6,
                    'mape': 0.003
                }
            },
            'rl': {
                'training_results': {
                    'rewards': [10, 15, 25, 35, 42, 49.92],
                    'episodes': [100, 200, 400, 600, 800, 1000]
                },
                'final_avg_reward': 49.92,
                'exploration_decay': 0.995
            },
            'session_info': {
                'session_id': '20250531_015149',
                'total_videos': 254,
                'total_frames': 13317,
                'training_duration': '1h 24min'
            }
        }
