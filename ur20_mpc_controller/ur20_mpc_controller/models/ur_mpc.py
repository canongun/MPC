#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.optimize import minimize
import moveit_commander
from typing import Dict, Tuple, List

from ur20_mpc_controller.models.base_estimator import BaseMotionEstimator

class URMPC:
    def __init__(self):
        # Initialize MoveIt
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander()
        self.move_group = moveit_commander.MoveGroupCommander("arm")  # Your group name
        
        # MPC parameters
        self.horizon = 10  # Prediction horizon
        self.dt = 0.1      # Time step (seconds)
        
        # Adjust weights for better stability
        self.w_ee_pos = 5.0     # Increase position tracking weight
        self.w_ee_ori = 0.5     # Keep orientation weight low
        self.w_control = 0.1    # Reduce control effort penalty for more aggressive correction
        
        # Reduce velocity limits for smoother motion
        self.joint_vel_limits = [-1.5, 1.5]  # rad/s (reduced from ±2.0)
        
        # State and control limits
        self.joint_pos_limits = np.array([  # [min, max] for each joint
            [-2*np.pi, 2*np.pi],  # Joint 1
            [-2*np.pi, 2*np.pi],  # Joint 2
            [-2*np.pi, 2*np.pi],  # Joint 3
            [-2*np.pi, 2*np.pi],  # Joint 4
            [-2*np.pi, 2*np.pi],  # Joint 5
            [-2*np.pi, 2*np.pi]   # Joint 6
        ])
        
        # Add joint velocity limits
        self.joint_vel_limits = {
            'shoulder_pan_joint': 2.094395102393195,
            'shoulder_lift_joint': 2.094395102393195,
            'elbow_joint': 2.617993877991494,
            'wrist_1_joint': 3.665191429188092,
            'wrist_2_joint': 3.665191429188092,
            'wrist_3_joint': 3.665191429188092
        }
        
        # Adjust joint-specific scaling factors to prefer certain joints
        self.joint_scales = np.array([1.0, 0.8, 0.6, 0.4, 0.3, 0.2])  # Prefer base joints
        
        self.base_estimator = BaseMotionEstimator()
        
        # Add transformation between mobile base and arm base
        self.arm_base_offset = {
            'x': 0.06,    # meters
            'y': 0.1,     # meters
            'z': 0.09,    # meters
            'roll': 0.0,  # radians
            'pitch': 0.0, # radians
            'yaw': np.pi  # 180 degrees in radians
        }
        
    def compute_control(self, 
                    current_joint_state: Dict,
                    current_ee_pose: Dict,
                    target_ee_pose: Dict,
                    base_state: Dict) -> np.ndarray:
        """Compute optimal joint velocities using MPC"""
        # Transform base motion to arm base frame
        transformed_base_state = self._transform_base_motion(base_state)
        
        # Initial guess
        x0 = np.zeros(12)
        x0[:6] = current_joint_state['position']
        
        # Get base velocity in arm frame
        base_vel = transformed_base_state['linear_velocity'][:3]  # Only take first 3 components
        
        # Get Jacobian for current configuration
        J = self._get_jacobian(current_joint_state['position'])
        J_pos = J[:3, :]  # Take only position part of Jacobian
        
        try:
            # Compute initial velocities using pseudoinverse
            x0[6:] = -np.linalg.pinv(J_pos) @ base_vel
        except Exception as e:
            rospy.logwarn(f"Failed to compute initial guess using Jacobian: {str(e)}")
            # Fallback to scaled velocities
            base_vel_magnitude = np.linalg.norm(base_vel)
            x0[6:] = self.joint_scales * base_vel_magnitude
        
        try:
            result = minimize(
                fun=lambda x: self._cost_function(
                    x, 
                    current_joint_state,
                    current_ee_pose,
                    target_ee_pose,
                    transformed_base_state
                ),
                x0=x0,
                method='SLSQP',
                bounds=self._get_bounds(x0),
                constraints=self._get_constraints(x0, current_ee_pose, transformed_base_state),
                options={
                    'ftol': 1e-4,
                    'maxiter': 50,
                    'disp': True
                }
            )
            
            if not result.success:
                rospy.logwarn(f"MPC optimization failed: {result.message}")
                return x0[6:]
                
            return result.x[6:]
            
        except Exception as e:
            rospy.logerr(f"Optimization error: {str(e)}")
            return x0[6:]
        
    def _cost_function(self, x: np.ndarray, current_joint_state: Dict,
                      current_ee_pose: Dict, target_ee_pose: Dict,
                      base_state: Dict) -> float:
        """Cost function focusing on active compensation directions"""
        joint_velocities = x[6:]
        
        # Get Jacobian
        J = self._get_jacobian(current_joint_state['position'])
        J_pos = J[:3, :]  # Position Jacobian
        
        # End-effector velocity
        ee_vel = J_pos @ joint_velocities
        base_vel = base_state['linear_velocity'][:3]
        compensation_error = ee_vel + base_vel
        
        # Detect active motion directions
        active_dirs = np.abs(base_vel) > 0.01
        
        # Weight compensation error more heavily in active directions
        dir_weights = np.where(active_dirs, 10.0, 1.0)
        position_cost = np.sum((dir_weights * compensation_error)**2)
        
        # Add regularization terms
        damping_cost = np.sum(joint_velocities**2) * 0.1
        
        return position_cost + damping_cost
        
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
    
    def _get_current_ee_pose(self, joint_positions: np.ndarray) -> Dict:
        """Get current end-effector pose using MoveIt"""
        # Set the joint positions
        self.move_group.set_joint_value_target(joint_positions)
        
        # Get the pose
        current_pose = self.move_group.get_current_pose(end_effector_link = "gripper_end_tool_link").pose
        
        return {
            'position': np.array([
                current_pose.position.x,
                current_pose.position.y,
                current_pose.position.z
            ]),
            'orientation': np.array([
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w
            ])
        }

    def _get_constraints(self, x0: np.ndarray, current_ee_pose: Dict, base_state: Dict) -> List[Dict]:
        """Update constraints for better compensation in all directions"""
        constraints = []
        
        # Velocity constraints
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: self.joint_vel_limits[1] - np.abs(x[6:])
        })
        
        def ee_motion_constraint(x):
            J = self._get_jacobian(x[:6])
            ee_vel = J @ x[6:]
            
            # Split velocities
            pos_vel = ee_vel[:3]
            ori_vel = ee_vel[3:]
            
            # Compensation constraint: ee_vel + base_vel should be near zero
            compensation_error = pos_vel + base_state['linear_velocity'][:3]
            
            # Tighter bounds on compensation error in primary motion direction
            motion_mask = np.abs(base_state['linear_velocity'][:3]) > 0.01
            bounds = np.where(motion_mask, 0.005, 0.02)  # Tighter bounds in motion direction
            
            # Combine constraints
            return np.concatenate([
                bounds - np.abs(compensation_error),  # Position compensation
                0.01 - np.abs(ori_vel)               # Orientation stability
            ])
        
        constraints.append({
            'type': 'ineq',
            'fun': ee_motion_constraint
        })
        
        return constraints

    def _transform_base_motion(self, base_state: Dict) -> Dict:
        """Transform base motion from mobile base frame to arm base frame"""
        base_vel = base_state['linear_velocity']
        base_pos = base_state['position']
        
        # Create full rotation matrix with proper handling of orientation
        yaw = self.arm_base_offset['yaw']
        R = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Transform velocities and positions
        arm_base_vel = R @ base_vel
        arm_base_pos = R @ base_pos + np.array([
            self.arm_base_offset['x'],
            self.arm_base_offset['y'],
            self.arm_base_offset['z']  # Include Z offset
        ])
        
        transformed_state = base_state.copy()
        transformed_state['linear_velocity'] = arm_base_vel
        transformed_state['position'] = arm_base_pos
        
        return transformed_state

    def _get_jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        """Get the Jacobian matrix for the current configuration"""
        try:
            # Ensure joint_positions is a list (not numpy array)
            if isinstance(joint_positions, np.ndarray):
                joint_positions = joint_positions.tolist()
            
            # Get Jacobian directly using joint positions
            J = self.move_group.get_jacobian_matrix(joint_positions)
            
            # Convert to numpy array and ensure correct shape (6x6)
            J = np.array(J)
            if J.shape != (6, 6):
                rospy.logwarn(f"Unexpected Jacobian shape: {J.shape}, using fallback")
                return np.eye(6)
                
            return J
            
        except Exception as e:
            rospy.logerr(f"Failed to get Jacobian: {str(e)}")
            return np.eye(6)

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