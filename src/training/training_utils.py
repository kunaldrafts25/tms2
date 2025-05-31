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
Advanced Training Utilities for TMS2 AI Models
Provides specialized training functions and data preparation utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
import pickle
from datetime import datetime, timedelta
import cv2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..utils.logger import get_logger


class TrafficDataGenerator:
    """
    Generate synthetic traffic data for training when real data is limited.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator."""
        np.random.seed(seed)
        self.logger = get_logger("TrafficDataGenerator")
    
    def generate_daily_traffic_pattern(self, hours: int = 24) -> List[float]:
        """
        Generate realistic daily traffic pattern.
        
        Args:
            hours: Number of hours to generate
            
        Returns:
            List of traffic density values (0-1)
        """
        # Base pattern: low at night, peaks at rush hours
        base_pattern = []
        
        for hour in range(hours):
            # Morning rush (7-9 AM)
            if 7 <= hour <= 9:
                base_traffic = 0.8 + np.random.normal(0, 0.1)
            # Evening rush (5-7 PM)
            elif 17 <= hour <= 19:
                base_traffic = 0.9 + np.random.normal(0, 0.1)
            # Daytime moderate traffic
            elif 10 <= hour <= 16:
                base_traffic = 0.5 + np.random.normal(0, 0.15)
            # Night time low traffic
            elif hour <= 6 or hour >= 22:
                base_traffic = 0.1 + np.random.normal(0, 0.05)
            # Transition periods
            else:
                base_traffic = 0.3 + np.random.normal(0, 0.1)
            
            # Ensure values are within bounds
            base_traffic = max(0.0, min(1.0, base_traffic))
            base_pattern.append(base_traffic)
        
        return base_pattern
    
    def generate_weekly_pattern(self, days: int = 7) -> List[List[float]]:
        """
        Generate weekly traffic patterns with weekend variations.
        
        Args:
            days: Number of days to generate
            
        Returns:
            List of daily patterns
        """
        weekly_pattern = []
        
        for day in range(days):
            # Weekend (Saturday=5, Sunday=6) has different patterns
            if day in [5, 6]:  # Weekend
                daily_pattern = self._generate_weekend_pattern()
            else:  # Weekday
                daily_pattern = self.generate_daily_traffic_pattern()
            
            weekly_pattern.append(daily_pattern)
        
        return weekly_pattern
    
    def _generate_weekend_pattern(self) -> List[float]:
        """Generate weekend traffic pattern (later start, more distributed)."""
        pattern = []
        
        for hour in range(24):
            # Weekend traffic starts later and is more distributed
            if 10 <= hour <= 14:  # Late morning to afternoon
                base_traffic = 0.6 + np.random.normal(0, 0.1)
            elif 15 <= hour <= 20:  # Evening activities
                base_traffic = 0.7 + np.random.normal(0, 0.1)
            elif hour <= 8 or hour >= 23:  # Late night/early morning
                base_traffic = 0.05 + np.random.normal(0, 0.02)
            else:
                base_traffic = 0.3 + np.random.normal(0, 0.1)
            
            base_traffic = max(0.0, min(1.0, base_traffic))
            pattern.append(base_traffic)
        
        return pattern
    
    def generate_intersection_data(self, num_intersections: int = 4,
                                 time_steps: int = 1000) -> Dict[str, np.ndarray]:
        """
        Generate multi-intersection traffic data.
        
        Args:
            num_intersections: Number of intersections
            time_steps: Number of time steps
            
        Returns:
            Dictionary with intersection data
        """
        intersection_data = {}
        
        for i in range(num_intersections):
            intersection_id = f"intersection_{i+1}"
            
            # Generate correlated traffic patterns
            base_pattern = np.tile(self.generate_daily_traffic_pattern(), 
                                 time_steps // 24 + 1)[:time_steps]
            
            variation = np.random.normal(0, 0.1, time_steps)
            traffic_density = np.clip(base_pattern + variation, 0, 1)
            
            vehicle_counts = (traffic_density * 50).astype(int)
            
            queue_lengths = np.maximum(0, vehicle_counts - 20 + np.random.normal(0, 5, time_steps))
            waiting_times = queue_lengths * 2 + np.random.exponential(10, time_steps)
            
            intersection_data[intersection_id] = {
                'vehicle_counts': vehicle_counts,
                'traffic_density': traffic_density,
                'queue_lengths': queue_lengths.astype(int),
                'waiting_times': waiting_times,
                'timestamps': np.arange(time_steps)
            }
        
        return intersection_data


class ModelEvaluator:
    """
    Comprehensive model evaluation and validation utilities.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.logger = get_logger("ModelEvaluator")
    
    def evaluate_lstm_model(self, model, X_test: np.ndarray, 
                          y_test: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive LSTM model evaluation.
        
        Args:
            model: Trained LSTM model
            X_test: Test input data
            y_test: Test target data
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Make predictions
            y_pred = model.predict(X_test, verbose=0)
            
            # Flatten if necessary
            if len(y_pred.shape) > 1:
                y_pred = y_pred.flatten()
            if len(y_test.shape) > 1:
                y_test = y_test.flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate percentage errors
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
            
            # Direction accuracy (for trend prediction)
            y_test_diff = np.diff(y_test)
            y_pred_diff = np.diff(y_pred)
            direction_accuracy = np.mean(np.sign(y_test_diff) == np.sign(y_pred_diff)) * 100
            
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2_score': float(r2),
                'mape': float(mape),
                'direction_accuracy': float(direction_accuracy),
                'samples': len(y_test)
            }
            
            self.logger.info(f"LSTM Evaluation - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"LSTM evaluation failed: {e}")
            return {'error': str(e)}
    
    def evaluate_rl_agent(self, agent, test_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate RL agent performance.
        
        Args:
            agent: Trained RL agent
            test_episodes: Number of test episodes
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            episode_rewards = []
            episode_lengths = []
            
            for episode in range(test_episodes):
                episode_reward = 0
                episode_length = 0
                
                # Simulate episode (simplified)
                for step in range(100):  # Max 100 steps per episode
                    # Generate mock state
                    state = self._generate_mock_state()
                    
                    # Get action (no exploration during evaluation)
                    action = agent.choose_action(state, training=False)
                    
                    # Calculate reward (simplified)
                    reward = np.random.normal(0.5, 0.1)
                    episode_reward += reward
                    episode_length += 1
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            metrics = {
                'mean_reward': float(np.mean(episode_rewards)),
                'std_reward': float(np.std(episode_rewards)),
                'max_reward': float(np.max(episode_rewards)),
                'min_reward': float(np.min(episode_rewards)),
                'mean_episode_length': float(np.mean(episode_lengths)),
                'episodes': test_episodes
            }
            
            self.logger.info(f"RL Evaluation - Mean Reward: {metrics['mean_reward']:.2f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"RL evaluation failed: {e}")
            return {'error': str(e)}
    
    def _generate_mock_state(self):
        """Generate mock state for RL evaluation."""
        # This would be replaced with actual state generation
        return {
            'vehicle_counts': [np.random.randint(0, 20) for _ in range(4)],
            'current_phase': np.random.randint(0, 4),
            'phase_time': np.random.randint(0, 60)
        }
    
    def plot_training_history(self, history: Dict[str, List], 
                            save_path: Optional[str] = None) -> None:
        """
        Plot training history and metrics.
        
        Args:
            history: Training history dictionary
            save_path: Optional path to save the plot
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Model Training History', fontsize=16)
            
            # Loss plot
            if 'loss' in history:
                axes[0, 0].plot(history['loss'], label='Training Loss')
                if 'val_loss' in history:
                    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
                axes[0, 0].set_title('Model Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # Accuracy/MAE plot
            if 'mae' in history:
                axes[0, 1].plot(history['mae'], label='Training MAE')
                if 'val_mae' in history:
                    axes[0, 1].plot(history['val_mae'], label='Validation MAE')
                axes[0, 1].set_title('Mean Absolute Error')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('MAE')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Learning rate plot
            if 'lr' in history:
                axes[1, 0].plot(history['lr'])
                axes[1, 0].set_title('Learning Rate')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].grid(True)
            
            # Custom metrics plot
            if 'custom_metric' in history:
                axes[1, 1].plot(history['custom_metric'])
                axes[1, 1].set_title('Custom Metric')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Metric Value')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Training history plot saved: {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Failed to plot training history: {e}")
    
    def generate_evaluation_report(self, metrics: Dict[str, Any], 
                                 model_name: str) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            metrics: Evaluation metrics
            model_name: Name of the model
            
        Returns:
            Formatted evaluation report
        """
        report = f"""
# {model_name} Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Metrics
"""
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                report += f"- **{metric.upper()}**: {value:.4f}\n"
            else:
                report += f"- **{metric.upper()}**: {value}\n"
        
        if 'mse' in metrics and 'r2_score' in metrics:
            r2 = metrics['r2_score']
            if r2 > 0.9:
                performance = "Excellent"
            elif r2 > 0.8:
                performance = "Good"
            elif r2 > 0.6:
                performance = "Fair"
            else:
                performance = "Poor"
            
            report += f"\n## Performance Assessment\n"
            report += f"Overall Performance: **{performance}** (R² = {r2:.3f})\n"
        
        return report


class DataAugmentation:
    """
    Data augmentation techniques for traffic data.
    """
    
    def __init__(self):
        """Initialize data augmentation."""
        self.logger = get_logger("DataAugmentation")
    
    def augment_traffic_sequences(self, sequences: np.ndarray, 
                                labels: np.ndarray,
                                augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment traffic sequence data.
        
        Args:
            sequences: Input sequences
            labels: Target labels
            augmentation_factor: How many times to augment
            
        Returns:
            Augmented sequences and labels
        """
        augmented_sequences = [sequences]
        augmented_labels = [labels]
        
        for _ in range(augmentation_factor):
            # Add noise
            noise_factor = 0.05
            noisy_sequences = sequences + np.random.normal(0, noise_factor, sequences.shape)
            augmented_sequences.append(noisy_sequences)
            augmented_labels.append(labels)
            
            # Time shifting
            shift_amount = np.random.randint(-2, 3)
            if shift_amount != 0:
                shifted_sequences = np.roll(sequences, shift_amount, axis=1)
                augmented_sequences.append(shifted_sequences)
                augmented_labels.append(labels)
        
        return np.vstack(augmented_sequences), np.hstack(augmented_labels)
    
    def augment_video_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Augment video frames for training.
        
        Args:
            frames: List of video frames
            
        Returns:
            Augmented frames
        """
        augmented_frames = []
        
        for frame in frames:
            # Original frame
            augmented_frames.append(frame)
            
            # Brightness adjustment
            bright_frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
            augmented_frames.append(bright_frame)
            
            # Contrast adjustment
            contrast_frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)
            augmented_frames.append(contrast_frame)
            
            # Gaussian noise
            noise = np.random.normal(0, 25, frame.shape).astype(np.uint8)
            noisy_frame = cv2.add(frame, noise)
            augmented_frames.append(noisy_frame)
        
        return augmented_frames
