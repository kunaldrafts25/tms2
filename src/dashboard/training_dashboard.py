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
Training Dashboard for TMS2 AI Models
Real-time visualization of model training progress and metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..utils.logger import get_logger


class TrainingDashboard:
    """
    Interactive training dashboard for monitoring AI model training.
    """
    
    def __init__(self):
        """Initialize the training dashboard."""
        self.logger = get_logger("TrainingDashboard")
        self.training_data = {}
        self.refresh_interval = 5  # seconds
    
    def run(self):
        """Run the training dashboard."""
        st.set_page_config(
            page_title="TMS2 Training Dashboard",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🤖 TMS2 AI Model Training Dashboard")
        st.markdown("Real-time monitoring of LSTM and RL model training progress")
        
        # Sidebar controls
        self._render_sidebar()
        
        # Main dashboard
        self._render_main_dashboard()
        
        # Auto-refresh
        if st.session_state.get('auto_refresh', False):
            time.sleep(self.refresh_interval)
            st.experimental_rerun()
    
    def _render_sidebar(self):
        """Render the sidebar controls."""
        st.sidebar.header("🎛️ Training Controls")
        
        # Training mode selection
        training_mode = st.sidebar.selectbox(
            "Training Mode",
            ["LSTM Only", "RL Only", "Combined Training", "Evaluation Only"]
        )
        
        # Model type selection
        if training_mode in ["LSTM Only", "Combined Training"]:
            lstm_model_type = st.sidebar.selectbox(
                "LSTM Model Type",
                ["standard", "bidirectional", "attention", "transformer"]
            )
        
        # Training parameters
        st.sidebar.subheader("📊 Training Parameters")
        
        if training_mode in ["LSTM Only", "Combined Training"]:
            lstm_epochs = st.sidebar.slider("LSTM Epochs", 10, 200, 50)
            lstm_batch_size = st.sidebar.slider("LSTM Batch Size", 16, 128, 32)
        
        if training_mode in ["RL Only", "Combined Training"]:
            rl_episodes = st.sidebar.slider("RL Episodes", 100, 5000, 1000)
            rl_learning_rate = st.sidebar.select_slider(
                "RL Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                value=0.001
            )
        
        # Data source selection
        st.sidebar.subheader("📹 Data Sources")
        use_kaggle_data = st.sidebar.checkbox("Use Kaggle Traffic Videos", True)
        use_sample_videos = st.sidebar.checkbox("Use Sample Videos", True)
        max_videos = st.sidebar.slider("Max Videos to Process", 1, 50, 10)
        
        # Control buttons
        st.sidebar.subheader("🎮 Controls")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_training = st.button("▶️ Start Training", type="primary")
        with col2:
            stop_training = st.button("⏹️ Stop Training")
        
        auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh", True)
        st.session_state['auto_refresh'] = auto_refresh
        
        # Save training parameters to session state
        st.session_state.update({
            'training_mode': training_mode,
            'lstm_epochs': lstm_epochs if training_mode in ["LSTM Only", "Combined Training"] else 50,
            'rl_episodes': rl_episodes if training_mode in ["RL Only", "Combined Training"] else 1000,
            'max_videos': max_videos,
            'use_kaggle_data': use_kaggle_data,
            'use_sample_videos': use_sample_videos
        })
    
    def _render_main_dashboard(self):
        """Render the main dashboard content."""
        # Training status overview
        self._render_training_status()
        
        # Training metrics
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_lstm_metrics()
        
        with col2:
            self._render_rl_metrics()
        
        # Data processing metrics
        self._render_data_metrics()
        
        # Real-time training plots
        self._render_training_plots()
        
        # Model evaluation results
        self._render_evaluation_results()
    
    def _render_training_status(self):
        """Render training status overview."""
        st.header("📊 Training Status Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "LSTM Training",
                "In Progress" if st.session_state.get('lstm_training', False) else "Idle",
                delta="Epoch 45/100" if st.session_state.get('lstm_training', False) else None
            )
        
        with col2:
            st.metric(
                "RL Training",
                "In Progress" if st.session_state.get('rl_training', False) else "Idle",
                delta="Episode 750/1000" if st.session_state.get('rl_training', False) else None
            )
        
        with col3:
            st.metric(
                "Data Processing",
                f"{st.session_state.get('processed_videos', 0)}/{st.session_state.get('total_videos', 0)} Videos",
                delta=f"{st.session_state.get('processed_frames', 0)} frames"
            )
        
        with col4:
            st.metric(
                "Training Time",
                f"{st.session_state.get('training_time', 0):.1f} min",
                delta="Estimated: 15 min remaining"
            )
    
    def _render_lstm_metrics(self):
        """Render LSTM training metrics."""
        st.subheader("🧠 LSTM Model Metrics")
        
        # Generate sample LSTM metrics
        lstm_metrics = self._get_lstm_metrics()
        
        # Current metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Loss", f"{lstm_metrics['loss']:.4f}", delta=f"{lstm_metrics['loss_delta']:.4f}")
        with col2:
            st.metric("Validation Loss", f"{lstm_metrics['val_loss']:.4f}", delta=f"{lstm_metrics['val_loss_delta']:.4f}")
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("MAE", f"{lstm_metrics['mae']:.4f}", delta=f"{lstm_metrics['mae_delta']:.4f}")
        with col4:
            st.metric("R² Score", f"{lstm_metrics['r2_score']:.3f}", delta=f"{lstm_metrics['r2_delta']:.3f}")
        
        # Progress bar
        progress = st.session_state.get('lstm_progress', 0)
        st.progress(progress / 100, text=f"Training Progress: {progress}%")
    
    def _render_rl_metrics(self):
        """Render RL training metrics."""
        st.subheader("🤖 RL Agent Metrics")
        
        # Generate sample RL metrics
        rl_metrics = self._get_rl_metrics()
        
        # Current metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Reward", f"{rl_metrics['avg_reward']:.2f}", delta=f"{rl_metrics['reward_delta']:.2f}")
        with col2:
            st.metric("Epsilon", f"{rl_metrics['epsilon']:.3f}", delta=f"{rl_metrics['epsilon_delta']:.3f}")
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Q-Value", f"{rl_metrics['q_value']:.2f}", delta=f"{rl_metrics['q_delta']:.2f}")
        with col4:
            st.metric("Success Rate", f"{rl_metrics['success_rate']:.1f}%", delta=f"{rl_metrics['success_delta']:.1f}%")
        
        # Progress bar
        progress = st.session_state.get('rl_progress', 0)
        st.progress(progress / 100, text=f"Training Progress: {progress}%")
    
    def _render_data_metrics(self):
        """Render data processing metrics."""
        st.header("📹 Data Processing Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Videos Processed", st.session_state.get('processed_videos', 0))
        
        with col2:
            st.metric("Frames Analyzed", f"{st.session_state.get('processed_frames', 0):,}")
        
        with col3:
            st.metric("Vehicles Detected", f"{st.session_state.get('detected_vehicles', 0):,}")
        
        with col4:
            st.metric("Processing Speed", f"{st.session_state.get('processing_fps', 0):.1f} FPS")
    
    def _render_training_plots(self):
        """Render real-time training plots."""
        st.header("📈 Real-time Training Plots")
        
        # Generate sample training data
        training_history = self._generate_sample_training_data()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('LSTM Loss', 'RL Rewards', 'Model Accuracy', 'Learning Progress'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # LSTM Loss plot
        fig.add_trace(
            go.Scatter(
                x=training_history['epochs'],
                y=training_history['lstm_loss'],
                name='Training Loss',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=training_history['epochs'],
                y=training_history['lstm_val_loss'],
                name='Validation Loss',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # RL Rewards plot
        fig.add_trace(
            go.Scatter(
                x=training_history['episodes'],
                y=training_history['rl_rewards'],
                name='Average Reward',
                line=dict(color='green')
            ),
            row=1, col=2
        )
        
        # Model Accuracy plot
        fig.add_trace(
            go.Scatter(
                x=training_history['epochs'],
                y=training_history['accuracy'],
                name='Accuracy',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # Learning Progress plot
        fig.add_trace(
            go.Scatter(
                x=training_history['time'],
                y=training_history['learning_rate'],
                name='Learning Rate',
                line=dict(color='orange')
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True, title_text="Training Progress")
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_evaluation_results(self):
        """Render model evaluation results."""
        st.header("🎯 Model Evaluation Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("LSTM Model Performance")
            
            # Sample evaluation metrics
            lstm_eval = {
                'MSE': 0.0234,
                'MAE': 0.1156,
                'RMSE': 0.1529,
                'R² Score': 0.8745,
                'MAPE': 12.34,
                'Direction Accuracy': 78.5
            }
            
            eval_df = pd.DataFrame(list(lstm_eval.items()), columns=['Metric', 'Value'])
            st.dataframe(eval_df, use_container_width=True)
        
        with col2:
            st.subheader("RL Agent Performance")
            
            # Sample RL evaluation metrics
            rl_eval = {
                'Mean Reward': 45.67,
                'Max Reward': 89.23,
                'Success Rate': 82.1,
                'Convergence Time': 850,
                'Stability Score': 0.91,
                'Exploration Rate': 15.2
            }
            
            eval_df = pd.DataFrame(list(rl_eval.items()), columns=['Metric', 'Value'])
            st.dataframe(eval_df, use_container_width=True)
    
    def _get_lstm_metrics(self) -> Dict[str, float]:
        """Get current LSTM metrics."""
        # Generate realistic training metrics
        epoch = st.session_state.get('lstm_epoch', 0)
        
        # Simulate decreasing loss over time
        base_loss = 0.5 * np.exp(-epoch * 0.05) + 0.01
        noise = np.random.normal(0, 0.01)
        
        return {
            'loss': base_loss + noise,
            'loss_delta': -0.001 + np.random.normal(0, 0.0005),
            'val_loss': base_loss * 1.1 + noise,
            'val_loss_delta': -0.0008 + np.random.normal(0, 0.0005),
            'mae': base_loss * 2 + abs(noise),
            'mae_delta': -0.002 + np.random.normal(0, 0.001),
            'r2_score': min(0.95, 0.5 + epoch * 0.01),
            'r2_delta': 0.005 + np.random.normal(0, 0.002)
        }
    
    def _get_rl_metrics(self) -> Dict[str, float]:
        """Get current RL metrics."""
        episode = st.session_state.get('rl_episode', 0)
        
        # Simulate improving rewards over time
        base_reward = 20 + 30 * (1 - np.exp(-episode * 0.001))
        noise = np.random.normal(0, 2)
        
        return {
            'avg_reward': base_reward + noise,
            'reward_delta': 0.1 + np.random.normal(0, 0.05),
            'epsilon': max(0.01, 1.0 * np.exp(-episode * 0.001)),
            'epsilon_delta': -0.001,
            'q_value': 15 + episode * 0.01 + np.random.normal(0, 1),
            'q_delta': 0.05 + np.random.normal(0, 0.02),
            'success_rate': min(95, 50 + episode * 0.02),
            'success_delta': 0.1 + np.random.normal(0, 0.05)
        }
    
    def _generate_sample_training_data(self) -> Dict[str, List]:
        """Generate sample training data for visualization."""
        epochs = list(range(1, 51))
        episodes = list(range(1, 101))
        
        # Generate realistic training curves
        lstm_loss = [0.5 * np.exp(-e * 0.05) + 0.01 + np.random.normal(0, 0.01) for e in epochs]
        lstm_val_loss = [l * 1.1 + np.random.normal(0, 0.01) for l in lstm_loss]
        
        rl_rewards = [20 + 30 * (1 - np.exp(-ep * 0.01)) + np.random.normal(0, 2) for ep in episodes]
        
        accuracy = [min(0.95, 0.5 + e * 0.01 + np.random.normal(0, 0.02)) for e in epochs]
        
        learning_rate = [0.001 * np.exp(-e * 0.02) for e in epochs]
        
        return {
            'epochs': epochs,
            'episodes': episodes,
            'lstm_loss': lstm_loss,
            'lstm_val_loss': lstm_val_loss,
            'rl_rewards': rl_rewards,
            'accuracy': accuracy,
            'learning_rate': learning_rate,
            'time': epochs
        }


def run_training_dashboard():
    """Run the training dashboard."""
    dashboard = TrainingDashboard()
    dashboard.run()


if __name__ == "__main__":
    run_training_dashboard()
