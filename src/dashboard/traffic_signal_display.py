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
Traffic Signal Display Components for TMS2 Dashboard

This module provides visual traffic signal simulation components including:
- Real-time traffic light visualization
- Signal state management
- RL decision reasoning display
- Interactive controls for manual override
- Emergency mode activation
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import json

# Pune Street Names Mapping
PUNE_INTERSECTION_NAMES = {
    'fc_jm_junction': 'FC Road & JM Road',
    'shivajinagar_deccan': 'Shivajinagar & Deccan',
    'baner_aundh_crossing': 'Baner & Aundh Road',
    'karve_senapati_junction': 'Karve & Senapati Bapat',
    'camp_mg_intersection': 'Camp & MG Road',
    # Legacy fallbacks
    'main': 'FC Road & JM Road',
    'north_main': 'Shivajinagar & Deccan',
    'south_main': 'Baner & Aundh Road',
    'east_main': 'Karve & Senapati Bapat',
    'intersection_1': 'FC Road & JM Road',
    'intersection_2': 'Shivajinagar & Deccan'
}

def get_pune_intersection_display_name(intersection_id: str) -> str:
    """Get the display name for a Pune intersection."""
    return PUNE_INTERSECTION_NAMES.get(intersection_id, intersection_id.replace('_', ' ').title())

# Signal state definitions
class SignalPhase(Enum):
    """Traffic signal phases."""
    NORTH_SOUTH_GREEN = 0
    NORTH_SOUTH_YELLOW = 1
    EAST_WEST_GREEN = 2
    EAST_WEST_YELLOW = 3
    ALL_RED = 4
    EMERGENCY = 5

@dataclass
class TrafficSignalState:
    """Current state of a traffic signal."""
    intersection_id: str
    current_phase: SignalPhase
    time_in_phase: float
    time_remaining: float
    last_change_time: datetime
    manual_override: bool = False
    emergency_mode: bool = False
    rl_confidence: float = 0.0
    next_recommended_phase: Optional[SignalPhase] = None

@dataclass
class RLDecisionInfo:
    """Information about RL agent decision."""
    intersection_id: str
    timestamp: datetime
    current_state: Dict[str, Any]
    action_taken: int
    action_description: str
    confidence: float
    reasoning: str
    predicted_improvement: float
    q_values: List[float]
    reward: float
    environmental_impact: float

class TrafficSignalSimulator:
    """
    Visual traffic signal simulator with RL integration.
    
    Features:
    - Real-time signal state visualization
    - RL decision monitoring
    - Manual override controls
    - Emergency mode activation
    - Signal timing analysis
    """
    
    def __init__(self):
        """Initialize the traffic signal simulator."""
        self.signal_states: Dict[str, TrafficSignalState] = {}
        self.rl_decisions: List[RLDecisionInfo] = []
        self.signal_history: List[Dict[str, Any]] = []
        
        # Default signal timings (in seconds)
        self.default_timings = {
            SignalPhase.NORTH_SOUTH_GREEN: 30,
            SignalPhase.NORTH_SOUTH_YELLOW: 5,
            SignalPhase.EAST_WEST_GREEN: 25,
            SignalPhase.EAST_WEST_YELLOW: 5,
            SignalPhase.ALL_RED: 2,
            SignalPhase.EMERGENCY: 60
        }
        
        # Initialize default intersections
        self._initialize_intersections()
    
    def _initialize_intersections(self):
        """Initialize Pune intersection states with authentic street names."""
        # Pune street intersections with proper IDs
        pune_intersections = [
            'fc_jm_junction',           # FC Road & JM Road Junction
            'shivajinagar_deccan',      # Shivajinagar & Deccan Gymkhana
            'baner_aundh_crossing',     # Baner Road & Aundh Road Crossing
            'karve_senapati_junction',  # Karve Road & Senapati Bapat Road
            'camp_mg_intersection'      # Camp & MG Road Intersection
        ]

        for intersection_id in pune_intersections:
            self.signal_states[intersection_id] = TrafficSignalState(
                intersection_id=intersection_id,
                current_phase=SignalPhase.NORTH_SOUTH_GREEN,
                time_in_phase=0.0,
                time_remaining=self.default_timings[SignalPhase.NORTH_SOUTH_GREEN],
                last_change_time=datetime.now(),
                rl_confidence=0.8
            )
    
    def update_signal_state(self, intersection_id: str, new_phase: SignalPhase, 
                          rl_confidence: float = 0.0, manual: bool = False):
        """Update signal state for an intersection."""
        if intersection_id in self.signal_states:
            state = self.signal_states[intersection_id]
            
            # Record state change
            if state.current_phase != new_phase:
                self.signal_history.append({
                    'timestamp': datetime.now(),
                    'intersection_id': intersection_id,
                    'from_phase': state.current_phase.name,
                    'to_phase': new_phase.name,
                    'manual_override': manual,
                    'rl_confidence': rl_confidence
                })
            
            # Update state
            state.current_phase = new_phase
            state.time_in_phase = 0.0
            state.time_remaining = self.default_timings.get(new_phase, 30)
            state.last_change_time = datetime.now()
            state.manual_override = manual
            state.rl_confidence = rl_confidence
    
    def add_rl_decision(self, decision_info: RLDecisionInfo):
        """Add RL decision information."""
        self.rl_decisions.append(decision_info)
        
        # Keep only recent decisions (last 100)
        if len(self.rl_decisions) > 100:
            self.rl_decisions = self.rl_decisions[-100:]
    
    def get_signal_color(self, phase: SignalPhase, direction: str) -> str:
        """Get signal color for visualization."""
        if phase == SignalPhase.EMERGENCY:
            return 'red'
        elif phase == SignalPhase.ALL_RED:
            return 'red'
        elif direction in ['north', 'south']:
            if phase == SignalPhase.NORTH_SOUTH_GREEN:
                return 'green'
            elif phase == SignalPhase.NORTH_SOUTH_YELLOW:
                return 'yellow'
            else:
                return 'red'
        else:  # east, west
            if phase == SignalPhase.EAST_WEST_GREEN:
                return 'green'
            elif phase == SignalPhase.EAST_WEST_YELLOW:
                return 'yellow'
            else:
                return 'red'


def create_traffic_signal_visualization(simulator: TrafficSignalSimulator, 
                                      intersection_id: str) -> go.Figure:
    """Create visual traffic signal display."""
    if intersection_id not in simulator.signal_states:
        return go.Figure().add_annotation(
            text=f"Intersection {intersection_id} not found",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    state = simulator.signal_states[intersection_id]
    
    fig = go.Figure()
    
    # Draw intersection roads
    # Horizontal road (East-West)
    fig.add_shape(type="rect", x0=-2, y0=-0.5, x1=2, y1=0.5,
                  fillcolor="gray", line=dict(color="black", width=2))
    
    # Vertical road (North-South)
    fig.add_shape(type="rect", x0=-0.5, y0=-2, x1=0.5, y1=2,
                  fillcolor="gray", line=dict(color="black", width=2))
    
    # Traffic signals for each direction
    directions = {
        'north': (0, 1.5, 0),
        'south': (0, -1.5, 180),
        'east': (1.5, 0, 270),
        'west': (-1.5, 0, 90)
    }
    
    for direction, (x, y, rotation) in directions.items():
        color = simulator.get_signal_color(state.current_phase, direction)
        
        # Signal light
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers',
            marker=dict(
                size=30,
                color=color,
                symbol='circle',
                line=dict(color='black', width=2)
            ),
            name=f'{direction.title()} Signal',
            showlegend=False
        ))
        
        # Direction label
        fig.add_annotation(
            x=x, y=y-0.3,
            text=direction.title(),
            showarrow=False,
            font=dict(size=10)
        )
    
    # Intersection ID with Pune street names
    display_name = get_pune_intersection_display_name(intersection_id)
    fig.add_annotation(
        x=0, y=0,
        text=display_name,
        showarrow=False,
        font=dict(size=12, color='white'),
        bgcolor='black',
        bordercolor='white',
        borderwidth=1
    )
    
    # Current phase info
    phase_text = f"Phase: {state.current_phase.name.replace('_', ' ')}<br>"
    phase_text += f"Time Remaining: {state.time_remaining:.1f}s<br>"
    phase_text += f"RL Confidence: {state.rl_confidence:.2f}"
    
    if state.manual_override:
        phase_text += "<br><span style='color:orange'>MANUAL OVERRIDE</span>"
    if state.emergency_mode:
        phase_text += "<br><span style='color:red'>EMERGENCY MODE</span>"
    
    fig.add_annotation(
        x=0, y=-2.5,
        text=phase_text,
        showarrow=False,
        font=dict(size=10),
        bgcolor='lightgray',
        bordercolor='black',
        borderwidth=1
    )
    
    fig.update_layout(
        title=f"Traffic Signal - {intersection_id.replace('_', ' ').title()}",
        xaxis=dict(range=[-3, 3], showgrid=False, showticklabels=False),
        yaxis=dict(range=[-3, 3], showgrid=False, showticklabels=False),
        width=400,
        height=400,
        template="plotly_white",
        showlegend=False
    )
    
    return fig


def create_rl_decision_display(simulator: TrafficSignalSimulator, 
                             intersection_id: str) -> Dict[str, Any]:
    """Create RL decision reasoning display."""
    recent_decisions = [d for d in simulator.rl_decisions 
                       if d.intersection_id == intersection_id]
    
    if not recent_decisions:
        return {
            'latest_decision': None,
            'decision_history': pd.DataFrame(),
            'q_values': [],
            'reasoning': "No recent RL decisions available"
        }
    
    latest = recent_decisions[-1]
    
    # Create decision history DataFrame
    history_data = []
    for decision in recent_decisions[-10:]:  # Last 10 decisions
        history_data.append({
            'timestamp': decision.timestamp,
            'action': decision.action_description,
            'confidence': decision.confidence,
            'reward': decision.reward,
            'improvement': decision.predicted_improvement
        })
    
    return {
        'latest_decision': latest,
        'decision_history': pd.DataFrame(history_data),
        'q_values': latest.q_values,
        'reasoning': latest.reasoning
    }


def create_signal_timing_chart(simulator: TrafficSignalSimulator, 
                             intersection_id: str) -> go.Figure:
    """Create signal timing analysis chart."""
    history = [h for h in simulator.signal_history 
              if h['intersection_id'] == intersection_id]
    
    if not history:
        return go.Figure().add_annotation(
            text="No signal timing data available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    fig = go.Figure()
    
    # Phase colors
    phase_colors = {
        'NORTH_SOUTH_GREEN': 'green',
        'NORTH_SOUTH_YELLOW': 'yellow',
        'EAST_WEST_GREEN': 'blue',
        'EAST_WEST_YELLOW': 'orange',
        'ALL_RED': 'red',
        'EMERGENCY': 'darkred'
    }
    
    for i, event in enumerate(history[-20:]):  # Last 20 events
        color = phase_colors.get(event['to_phase'], 'gray')
        
        fig.add_trace(go.Scatter(
            x=[event['timestamp']],
            y=[event['to_phase']],
            mode='markers',
            marker=dict(
                size=15,
                color=color,
                symbol='circle' if not event['manual_override'] else 'diamond'
            ),
            name=event['to_phase'],
            showlegend=False,
            hovertemplate=f"<b>{event['to_phase']}</b><br>" +
                         f"Time: {event['timestamp'].strftime('%H:%M:%S')}<br>" +
                         f"Manual: {event['manual_override']}<br>" +
                         f"RL Confidence: {event['rl_confidence']:.2f}<extra></extra>"
        ))
    
    fig.update_layout(
        title=f"Signal Timing History - {intersection_id.replace('_', ' ').title()}",
        xaxis_title="Time",
        yaxis_title="Signal Phase",
        template="plotly_white",
        height=300
    )
    
    return fig


def create_q_values_chart(q_values: List[float]) -> go.Figure:
    """Create Q-values visualization for RL decisions."""
    if not q_values:
        return go.Figure().add_annotation(
            text="No Q-values available",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    actions = ['Keep Current', 'Change Phase', 'Emergency Override'][:len(q_values)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=actions,
            y=q_values,
            marker_color=['green' if i == np.argmax(q_values) else 'lightblue' 
                         for i in range(len(q_values))],
            text=[f'{v:.3f}' for v in q_values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="RL Agent Q-Values",
        xaxis_title="Actions",
        yaxis_title="Q-Value",
        template="plotly_white",
        height=300
    )
    
    return fig


def create_signal_control_interface(simulator: TrafficSignalSimulator, 
                                  intersection_id: str) -> None:
    """Create interactive signal control interface."""
    st.subheader(f"🎛️ Signal Control - {intersection_id.replace('_', ' ').title()}")
    
    if intersection_id not in simulator.signal_states:
        st.error(f"Intersection {intersection_id} not found")
        return
    
    state = simulator.signal_states[intersection_id]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Manual Override**")
        
        # Phase selection
        phase_options = [
            ("North-South Green", SignalPhase.NORTH_SOUTH_GREEN),
            ("North-South Yellow", SignalPhase.NORTH_SOUTH_YELLOW),
            ("East-West Green", SignalPhase.EAST_WEST_GREEN),
            ("East-West Yellow", SignalPhase.EAST_WEST_YELLOW),
            ("All Red", SignalPhase.ALL_RED)
        ]
        
        selected_phase_name = st.selectbox(
            "Select Phase",
            [name for name, _ in phase_options],
            key=f"phase_select_{intersection_id}"
        )
        
        selected_phase = next(phase for name, phase in phase_options 
                            if name == selected_phase_name)
        
        if st.button("Apply Manual Override", key=f"manual_{intersection_id}"):
            simulator.update_signal_state(intersection_id, selected_phase, 
                                        manual=True, rl_confidence=0.0)
            st.success(f"Manual override applied: {selected_phase_name}")
    
    with col2:
        st.write("**Emergency Controls**")
        
        if st.button("🚨 Emergency Mode", key=f"emergency_{intersection_id}"):
            simulator.update_signal_state(intersection_id, SignalPhase.EMERGENCY, 
                                        manual=True, rl_confidence=0.0)
            simulator.signal_states[intersection_id].emergency_mode = True
            st.error("Emergency mode activated!")
        
        if st.button("🔄 Resume Normal", key=f"resume_{intersection_id}"):
            simulator.update_signal_state(intersection_id, SignalPhase.NORTH_SOUTH_GREEN, 
                                        manual=False, rl_confidence=0.8)
            simulator.signal_states[intersection_id].emergency_mode = False
            simulator.signal_states[intersection_id].manual_override = False
            st.success("Normal operation resumed")
    
    with col3:
        st.write("**Current Status**")
        st.metric("Current Phase", state.current_phase.name.replace('_', ' '))
        st.metric("Time Remaining", f"{state.time_remaining:.1f}s")
        st.metric("RL Confidence", f"{state.rl_confidence:.2f}")
        
        if state.manual_override:
            st.warning("Manual Override Active")
        if state.emergency_mode:
            st.error("Emergency Mode Active")


def simulate_rl_decision(simulator: TrafficSignalSimulator, intersection_id: str):
    """Simulate an RL decision for demonstration."""
    # Create sample RL decision
    current_time = datetime.now()
    
    # Sample Q-values for different actions
    q_values = [
        np.random.uniform(0.5, 0.9),  # Keep current
        np.random.uniform(0.3, 0.8),  # Change phase
        np.random.uniform(0.1, 0.3)   # Emergency
    ]
    
    best_action = np.argmax(q_values)
    action_descriptions = [
        "Maintain current signal phase",
        "Change to next phase",
        "Emergency override"
    ]
    
    # Generate reasoning based on action
    if best_action == 0:
        reasoning = "Traffic flow is stable. Current phase timing is optimal based on vehicle density predictions."
    elif best_action == 1:
        reasoning = "LSTM model predicts increased traffic from perpendicular direction. Phase change recommended."
    else:
        reasoning = "Emergency vehicle detected or critical traffic congestion. Override required."
    
    decision_info = RLDecisionInfo(
        intersection_id=intersection_id,
        timestamp=current_time,
        current_state={
            'traffic_density': [0.6, 0.4, 0.3, 0.5],
            'queue_lengths': [8, 5, 3, 6],
            'time_in_phase': 15.0
        },
        action_taken=best_action,
        action_description=action_descriptions[best_action],
        confidence=q_values[best_action],
        reasoning=reasoning,
        predicted_improvement=np.random.uniform(0.1, 0.3),
        q_values=q_values,
        reward=np.random.uniform(-2, 10),
        environmental_impact=np.random.uniform(0.05, 0.15)
    )
    
    simulator.add_rl_decision(decision_info)
    
    # Apply the decision if it's a phase change
    if best_action == 1:
        current_phase = simulator.signal_states[intersection_id].current_phase
        if current_phase == SignalPhase.NORTH_SOUTH_GREEN:
            new_phase = SignalPhase.NORTH_SOUTH_YELLOW
        elif current_phase == SignalPhase.NORTH_SOUTH_YELLOW:
            new_phase = SignalPhase.EAST_WEST_GREEN
        elif current_phase == SignalPhase.EAST_WEST_GREEN:
            new_phase = SignalPhase.EAST_WEST_YELLOW
        else:
            new_phase = SignalPhase.NORTH_SOUTH_GREEN
        
        simulator.update_signal_state(intersection_id, new_phase, 
                                    rl_confidence=q_values[best_action])
