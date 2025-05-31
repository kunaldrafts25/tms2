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
Enhanced Signal Performance Analytics for TMS2 Smart Traffic Dashboard
Provides comprehensive signal timing effectiveness metrics and Pune street integration.
"""

import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

# Pune Street Names Integration
PUNE_INTERSECTION_NAMES = {
    'fc_jm_junction': {
        'name': 'FC Road & JM Road Junction',
        'display_name': 'FC Road & JM Road',
        'area': 'Deccan Gymkhana',
        'traffic_volume': 'high'
    },
    'shivajinagar_deccan': {
        'name': 'Shivajinagar & Deccan Gymkhana Intersection',
        'display_name': 'Shivajinagar & Deccan',
        'area': 'Shivajinagar',
        'traffic_volume': 'medium'
    },
    'baner_aundh_crossing': {
        'name': 'Baner Road & Aundh Road Crossing',
        'display_name': 'Baner & Aundh Road',
        'area': 'Baner',
        'traffic_volume': 'very_high'
    },
    'karve_senapati_junction': {
        'name': 'Karve Road & Senapati Bapat Road Junction',
        'display_name': 'Karve & Senapati Bapat',
        'area': 'Erandwane',
        'traffic_volume': 'high'
    },
    'camp_mg_intersection': {
        'name': 'Camp & MG Road Intersection',
        'display_name': 'Camp & MG Road',
        'area': 'Camp',
        'traffic_volume': 'medium'
    }
}

@dataclass
class SignalTimingMetrics:
    """Signal timing performance metrics."""
    intersection_id: str
    cycle_time: float
    green_duration: float
    yellow_duration: float
    red_duration: float
    all_red_duration: float
    wait_time: float
    throughput: float
    efficiency_score: float
    timestamp: float

@dataclass
class RLDecisionData:
    """RL agent decision reasoning data."""
    intersection_id: str
    decision_type: str
    confidence: float
    reasoning: str
    predicted_improvement: float
    actual_improvement: Optional[float]
    timestamp: float

@dataclass
class CoordinationMetrics:
    """Multi-intersection coordination metrics."""
    primary_intersection: str
    coordinated_intersections: List[str]
    coordination_score: float
    sync_efficiency: float
    network_throughput: float
    timestamp: float

class SignalPerformanceAnalytics:
    """Comprehensive signal performance analytics with Pune street integration."""
    
    def __init__(self, max_history_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.max_history_size = max_history_size
        
        # Performance data storage
        self.timing_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.rl_decisions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.coordination_data: deque = deque(maxlen=max_history_size)
        
        # Real-time state tracking
        self.current_states: Dict[str, Dict[str, Any]] = {}
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        self._initialize_pune_baselines()
    
    def _initialize_pune_baselines(self):
        """Initialize performance baselines for Pune intersections."""
        for intersection_id, info in PUNE_INTERSECTION_NAMES.items():
            traffic_volume = info['traffic_volume']
            
            # Set baselines based on traffic volume
            if traffic_volume == 'very_high':
                baseline = {'cycle_time': 120, 'wait_time': 90, 'throughput': 0.8}
            elif traffic_volume == 'high':
                baseline = {'cycle_time': 100, 'wait_time': 75, 'throughput': 0.75}
            else:  # medium
                baseline = {'cycle_time': 80, 'wait_time': 60, 'throughput': 0.7}
            
            self.performance_baselines[intersection_id] = baseline
    
    def record_signal_timing(self, intersection_id: str, cycle_time: float,
                           green_duration: float, yellow_duration: float,
                           red_duration: float, all_red_duration: float,
                           wait_time: float, throughput: float) -> None:
        """Record signal timing metrics."""
        efficiency_score = self._calculate_efficiency_score(
            intersection_id, cycle_time, wait_time, throughput
        )
        
        metrics = SignalTimingMetrics(
            intersection_id=intersection_id,
            cycle_time=cycle_time,
            green_duration=green_duration,
            yellow_duration=yellow_duration,
            red_duration=red_duration,
            all_red_duration=all_red_duration,
            wait_time=wait_time,
            throughput=throughput,
            efficiency_score=efficiency_score,
            timestamp=time.time()
        )
        
        self.timing_metrics[intersection_id].append(metrics)
        self._update_current_state(intersection_id, metrics)
    
    def record_rl_decision(self, intersection_id: str, decision_type: str,
                          confidence: float, reasoning: str,
                          predicted_improvement: float) -> None:
        """Record RL agent decision with reasoning."""
        decision = RLDecisionData(
            intersection_id=intersection_id,
            decision_type=decision_type,
            confidence=confidence,
            reasoning=reasoning,
            predicted_improvement=predicted_improvement,
            actual_improvement=None,  # To be updated later
            timestamp=time.time()
        )
        
        self.rl_decisions[intersection_id].append(decision)
    
    def record_coordination_metrics(self, primary_intersection: str,
                                  coordinated_intersections: List[str],
                                  coordination_score: float,
                                  sync_efficiency: float,
                                  network_throughput: float) -> None:
        """Record multi-intersection coordination metrics."""
        metrics = CoordinationMetrics(
            primary_intersection=primary_intersection,
            coordinated_intersections=coordinated_intersections,
            coordination_score=coordination_score,
            sync_efficiency=sync_efficiency,
            network_throughput=network_throughput,
            timestamp=time.time()
        )
        
        self.coordination_data.append(metrics)
    
    def _calculate_efficiency_score(self, intersection_id: str, cycle_time: float,
                                  wait_time: float, throughput: float) -> float:
        """Calculate signal efficiency score (0-100)."""
        baseline = self.performance_baselines.get(intersection_id, {})
        
        # Cycle time efficiency (closer to optimal = better)
        optimal_cycle = baseline.get('cycle_time', 90)
        cycle_efficiency = max(0, 100 - abs(cycle_time - optimal_cycle) * 2)
        
        # Wait time efficiency (lower = better)
        max_wait = baseline.get('wait_time', 120)
        wait_efficiency = max(0, 100 - (wait_time / max_wait) * 100)
        
        # Throughput efficiency (higher = better)
        target_throughput = baseline.get('throughput', 0.7)
        throughput_efficiency = min(100, (throughput / target_throughput) * 100)
        
        # Weighted average
        efficiency_score = (cycle_efficiency * 0.3 + wait_efficiency * 0.4 + throughput_efficiency * 0.3)
        return round(efficiency_score, 1)
    
    def _update_current_state(self, intersection_id: str, metrics: SignalTimingMetrics) -> None:
        """Update current state tracking."""
        self.current_states[intersection_id] = {
            'cycle_time': metrics.cycle_time,
            'efficiency_score': metrics.efficiency_score,
            'wait_time': metrics.wait_time,
            'throughput': metrics.throughput,
            'last_update': metrics.timestamp
        }
    
    def get_intersection_performance(self, intersection_id: str,
                                   time_window_minutes: int = 30) -> Dict[str, Any]:
        """Get comprehensive performance analysis for an intersection."""
        if intersection_id not in self.timing_metrics:
            return {'error': f'No data available for intersection {intersection_id}'}
        
        # Get recent data within time window
        cutoff_time = time.time() - (time_window_minutes * 60)
        recent_metrics = [m for m in self.timing_metrics[intersection_id] if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {'error': f'No recent data for intersection {intersection_id}'}
        
        # Calculate performance statistics
        cycle_times = [m.cycle_time for m in recent_metrics]
        wait_times = [m.wait_time for m in recent_metrics]
        throughputs = [m.throughput for m in recent_metrics]
        efficiency_scores = [m.efficiency_score for m in recent_metrics]
        
        # Get Pune street information
        pune_info = PUNE_INTERSECTION_NAMES.get(intersection_id, {})
        
        return {
            'intersection_info': {
                'id': intersection_id,
                'name': pune_info.get('name', intersection_id),
                'display_name': pune_info.get('display_name', intersection_id),
                'area': pune_info.get('area', 'Unknown'),
                'traffic_volume': pune_info.get('traffic_volume', 'medium')
            },
            'timing_analysis': {
                'average_cycle_time': round(np.mean(cycle_times), 1),
                'cycle_time_variance': round(np.var(cycle_times), 2),
                'optimal_cycle_range': [60, 120],
                'cycle_time_trend': self._calculate_trend(cycle_times)
            },
            'wait_time_analysis': {
                'average_wait_time': round(np.mean(wait_times), 1),
                'max_wait_time': round(max(wait_times), 1),
                'wait_time_reduction': self._calculate_wait_time_reduction(intersection_id),
                'wait_time_trend': self._calculate_trend(wait_times)
            },
            'throughput_analysis': {
                'average_throughput': round(np.mean(throughputs), 3),
                'throughput_improvement': self._calculate_throughput_improvement(intersection_id),
                'throughput_trend': self._calculate_trend(throughputs)
            },
            'efficiency_metrics': {
                'current_efficiency_score': round(np.mean(efficiency_scores), 1),
                'efficiency_trend': self._calculate_trend(efficiency_scores),
                'performance_status': self._get_performance_status(np.mean(efficiency_scores))
            },
            'rl_optimization': self._get_rl_optimization_summary(intersection_id),
            'data_points': len(recent_metrics),
            'time_window_minutes': time_window_minutes
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return 'improving'
        elif slope < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_wait_time_reduction(self, intersection_id: str) -> float:
        """Calculate wait time reduction percentage from RL optimization."""
        if intersection_id not in self.timing_metrics:
            return 0.0
        
        metrics = list(self.timing_metrics[intersection_id])
        if len(metrics) < 10:
            return 0.0
        
        # Compare first 25% vs last 25% of data
        quarter_size = len(metrics) // 4
        early_wait_times = [m.wait_time for m in metrics[:quarter_size]]
        recent_wait_times = [m.wait_time for m in metrics[-quarter_size:]]
        
        if not early_wait_times or not recent_wait_times:
            return 0.0
        
        early_avg = np.mean(early_wait_times)
        recent_avg = np.mean(recent_wait_times)
        
        if early_avg > 0:
            reduction = ((early_avg - recent_avg) / early_avg) * 100
            return round(max(0, reduction), 1)
        
        return 0.0
    
    def _calculate_throughput_improvement(self, intersection_id: str) -> float:
        """Calculate throughput improvement percentage from RL optimization."""
        if intersection_id not in self.timing_metrics:
            return 0.0
        
        metrics = list(self.timing_metrics[intersection_id])
        if len(metrics) < 10:
            return 0.0
        
        # Compare first 25% vs last 25% of data
        quarter_size = len(metrics) // 4
        early_throughputs = [m.throughput for m in metrics[:quarter_size]]
        recent_throughputs = [m.throughput for m in metrics[-quarter_size:]]
        
        if not early_throughputs or not recent_throughputs:
            return 0.0
        
        early_avg = np.mean(early_throughputs)
        recent_avg = np.mean(recent_throughputs)
        
        if early_avg > 0:
            improvement = ((recent_avg - early_avg) / early_avg) * 100
            return round(max(0, improvement), 1)
        
        return 0.0
    
    def _get_performance_status(self, efficiency_score: float) -> str:
        """Get performance status based on efficiency score."""
        if efficiency_score >= 85:
            return 'excellent'
        elif efficiency_score >= 70:
            return 'good'
        elif efficiency_score >= 55:
            return 'fair'
        else:
            return 'poor'
    
    def _get_rl_optimization_summary(self, intersection_id: str) -> Dict[str, Any]:
        """Get RL optimization summary for an intersection."""
        if intersection_id not in self.rl_decisions:
            return {'status': 'No RL decision data available'}
        
        recent_decisions = list(self.rl_decisions[intersection_id])[-20:]  # Last 20 decisions
        
        if not recent_decisions:
            return {'status': 'No recent RL decisions'}
        
        confidences = [d.confidence for d in recent_decisions]
        improvements = [d.predicted_improvement for d in recent_decisions if d.predicted_improvement is not None]
        
        return {
            'total_decisions': len(recent_decisions),
            'average_confidence': round(np.mean(confidences), 3),
            'high_confidence_decisions': len([c for c in confidences if c >= 0.8]),
            'average_predicted_improvement': round(np.mean(improvements), 2) if improvements else 0,
            'latest_reasoning': recent_decisions[-1].reasoning if recent_decisions else 'No recent decisions'
        }
    
    def get_coordination_efficiency(self) -> Dict[str, Any]:
        """Get multi-intersection coordination efficiency metrics."""
        if not self.coordination_data:
            return {'status': 'No coordination data available'}
        
        recent_data = list(self.coordination_data)[-10:]  # Last 10 coordination events
        
        coordination_scores = [d.coordination_score for d in recent_data]
        sync_efficiencies = [d.sync_efficiency for d in recent_data]
        network_throughputs = [d.network_throughput for d in recent_data]
        
        return {
            'average_coordination_score': round(np.mean(coordination_scores), 3),
            'average_sync_efficiency': round(np.mean(sync_efficiencies), 3),
            'average_network_throughput': round(np.mean(network_throughputs), 3),
            'coordination_events': len(recent_data),
            'coordination_status': 'excellent' if np.mean(coordination_scores) >= 0.8 else 'good' if np.mean(coordination_scores) >= 0.6 else 'fair'
        }
    
    def get_pune_intersections_overview(self) -> Dict[str, Any]:
        """Get overview of all Pune intersections performance."""
        overview = {}
        
        for intersection_id in PUNE_INTERSECTION_NAMES.keys():
            if intersection_id in self.current_states:
                state = self.current_states[intersection_id]
                pune_info = PUNE_INTERSECTION_NAMES[intersection_id]
                
                overview[intersection_id] = {
                    'display_name': pune_info['display_name'],
                    'area': pune_info['area'],
                    'efficiency_score': state.get('efficiency_score', 0),
                    'wait_time': state.get('wait_time', 0),
                    'throughput': state.get('throughput', 0),
                    'status': self._get_performance_status(state.get('efficiency_score', 0))
                }
        
        return overview
