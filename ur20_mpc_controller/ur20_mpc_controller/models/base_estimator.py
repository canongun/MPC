#!/usr/bin/env python3

import numpy as np
from typing import Dict

class BaseMotionEstimator:
    def __init__(self):
        self.history_size = 10
        self.position_history = []
        self.velocity_history = []
        self.dt = 0.1  # 10Hz
        
    def update(self, current_base_state: Dict) -> Dict:
        """Update estimator with new base state measurement"""
        # Add current position to history
        self.position_history.append(current_base_state['position'])
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
            
        # Compute velocity using finite difference
        if len(self.position_history) >= 2:
            current_vel = (self.position_history[-1] - self.position_history[-2]) / self.dt
            self.velocity_history.append(current_vel)
            if len(self.velocity_history) > self.history_size:
                self.velocity_history.pop(0)
        
        # Return state with all required fields
        return {
            'position': current_base_state['position'],
            'orientation': np.zeros(3),  # Not used in current implementation
            'linear_velocity': self._estimate_velocity(),
            'angular_velocity': np.zeros(3)  # Not used in current implementation
        }
    
    def _estimate_velocity(self) -> np.ndarray:
        """Estimate current velocity based on position history"""
        if len(self.position_history) < 2:
            return np.zeros(3)
            
        # Use finite difference for velocity
        velocity = (self.position_history[-1] - self.position_history[-2]) / self.dt
        
        # Optional: apply smoothing if velocity history exists
        if self.velocity_history:
            alpha = 0.7  # Smoothing factor
            velocity = alpha * velocity + (1 - alpha) * self.velocity_history[-1]
            
        return velocity