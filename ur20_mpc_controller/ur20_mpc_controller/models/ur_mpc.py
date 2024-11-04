#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, List

from ur20_mpc_controller.models.base_estimator import BaseMotionEstimator

class URMPC:
    def __init__(self):
        # MPC parameters
        self.horizon = 10  # Prediction horizon
        self.dt = 0.1      # Time step (seconds)
        
        # Adjust weights for smoother response
        self.w_ee_pos = 2.0     # Increase position tracking weight
        self.w_ee_ori = 0.5     # Keep orientation weight
        self.w_control = 0.3    # Increase control effort penalty
        
        # Adjust velocity limits for smoother motion
        self.joint_vel_limits = np.array([-2.0, 2.0])  # Reduced from ±3.0
        
        # State and control limits
        self.joint_pos_limits = np.array([  # [min, max] for each joint
            [-2*np.pi, 2*np.pi],  # Joint 1
            [-2*np.pi, 2*np.pi],  # Joint 2
            [-2*np.pi, 2*np.pi],  # Joint 3
            [-2*np.pi, 2*np.pi],  # Joint 4
            [-2*np.pi, 2*np.pi],  # Joint 5
            [-2*np.pi, 2*np.pi]   # Joint 6
        ])
        
        self.base_estimator = BaseMotionEstimator()
        
    def compute_control(self, 
                       current_joint_state: Dict,
                       current_ee_pose: Dict,
                       target_ee_pose: Dict,
                       base_state: Dict) -> np.ndarray:
        """Compute optimal joint velocities using MPC"""
        # Initial guess - set to non-zero when base is moving
        x0 = np.zeros(12)
        x0[:6] = current_joint_state['position']
        
        # Set initial velocities proportional to base motion
        base_vel_magnitude = np.linalg.norm(base_state['linear_velocity'][:2])
        x0[6:] = 0.1 * base_vel_magnitude * np.ones(6)
        
        try:
            result = minimize(
                fun=lambda x: self._cost_function(x, current_ee_pose, target_ee_pose, base_state),
                x0=x0,
                method='SLSQP',
                bounds=self._get_bounds(x0),
                constraints=self._get_constraints(x0, current_ee_pose, base_state),
                options={
                    'ftol': 1e-6,
                    'maxiter': 200,
                    'disp': True
                }
            )
            
            if not result.success:
                rospy.logwarn(f"MPC optimization failed: {result.message}")
                # Return the initial guess instead of zeros
                return x0[6:]
                
            return result.x[6:]
            
        except Exception as e:
            rospy.logerr(f"Optimization error: {str(e)}")
            return x0[6:]  # Return initial guess instead of zeros
        
    def _cost_function(self, 
                      x: np.ndarray, 
                      current_ee_pose: Dict,
                      target_ee_pose: Dict,
                      base_state: Dict) -> float:
        """Compute cost for optimization with end-effector stabilization"""
        joint_velocities = x[6:]
        
        # Forward kinematics to predict EE motion
        predicted_ee_pos = self._predict_ee_position(x[:6], joint_velocities, current_ee_pose)
        
        # Base motion compensation term
        base_vel = base_state['linear_velocity'][:2]
        base_vel_magnitude = np.linalg.norm(base_vel)
        
        # Scale desired velocities differently for each joint
        joint_scales = np.array([0.3, 0.4, 0.5, 0.7, 0.8, 1.0])
        desired_vel = base_vel_magnitude * joint_scales
        
        # Cost terms
        ee_pos_error = np.linalg.norm(predicted_ee_pos - target_ee_pose['position'])
        velocity_tracking = np.sum(joint_scales * (joint_velocities - desired_vel)**2)
        smoothness = np.sum(np.diff(joint_velocities)**2)
        proximal_motion = np.sum(joint_velocities[:3]**2)
        
        cost = (
            5.0 * ee_pos_error +           # End-effector position stability
            1.0 * velocity_tracking +      # Base motion compensation
            0.1 * smoothness +            # Smooth motion
            0.3 * proximal_motion         # Minimize proximal joint motion
        )
        
        return cost
        
    def _predict_ee_position(self, 
                            joint_positions: np.ndarray, 
                            joint_velocities: np.ndarray,
                            current_ee_pose: Dict) -> np.ndarray:
        """Predict end-effector position after applying velocities"""
        # Simple forward prediction
        predicted_joints = joint_positions + joint_velocities * self.dt
        
        # TODO: Add actual forward kinematics calculation
        # For now, return current position
        return current_ee_pose['position']
        
    def _get_bounds(self, x0: np.ndarray) -> List[Tuple[float, float]]:
        """Get bounds for optimization variables"""
        bounds = []
        # Position bounds
        for i in range(6):
            bounds.append((self.joint_pos_limits[i][0], 
                         self.joint_pos_limits[i][1]))
        # Velocity bounds
        for i in range(6):
            bounds.append((self.joint_vel_limits[0], 
                         self.joint_vel_limits[1]))
        return bounds
        
    def _get_constraints(self, x0: np.ndarray, current_ee_pose: Dict, base_state: Dict) -> List[Dict]:
        """Get constraints for optimization problem with base compensation"""
        # Remove equality constraints that might be causing issues
        constraints = []
        
        # Only add inequality constraint for minimum motion when base is moving
        base_vel = base_state['linear_velocity']
        base_vel_magnitude = np.linalg.norm(base_vel[:2])
        
        if base_vel_magnitude > 0.01:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: np.sum(x[6:]**2) - (0.1 * base_vel_magnitude)**2
            })
        
        # End-effector position constraint
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.linalg.norm(
                self._predict_ee_position(x[:6], x[6:], current_ee_pose) - 
                current_ee_pose['position']
            ) - base_vel_magnitude * self.dt
        })
        
        return constraints

def main():
    """Test the MPC controller"""
    rospy.init_node('test_ur_mpc')
    
    # Initialize MPC
    controller = URMPC()
    
    # Test with dummy data
    current_joint_state = {
        'position': np.zeros(6),
        'velocity': np.zeros(6)
    }
    
    current_ee_pose = {
        'position': np.array([0.5, 0, 0.5]),
        'orientation': np.zeros(3)
    }
    
    target_ee_pose = {
        'position': np.array([0.6, 0, 0.5]),
        'orientation': np.zeros(3)
    }
    
    base_state = {
        'position': np.zeros(3),
        'orientation': np.zeros(3),
        'linear_velocity': np.zeros(3),
        'angular_velocity': np.zeros(3)
    }
    
    # Compute control
    control = controller.compute_control(
        current_joint_state,
        current_ee_pose,
        target_ee_pose,
        base_state
    )
    
    rospy.loginfo(f"Computed control: {control}")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass