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
        
        # Add joint velocity limits
        self.joint_vel_limits = [-2.0, 2.0]  # rad/s
        
        # Add joint-specific scaling factors
        self.joint_scales = np.array([0.3, 0.4, 0.5, 0.7, 0.8, 1.0])
        
        self.base_estimator = BaseMotionEstimator()
        
        # Add transformation between mobile base and arm base
        self.arm_base_offset = {
            'x': 0.06,
            'y': 0.1,
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
        
        # Set initial velocities based on transformed base motion
        base_vel = transformed_base_state['linear_velocity'][:2]
        base_vel_magnitude = np.linalg.norm(base_vel)
        
        if base_vel_magnitude > 0.01:
            # Scale velocities based on joint position in chain
            x0[6:] = self.joint_scales * base_vel_magnitude
        
        try:
            result = minimize(
                fun=lambda x: self._cost_function(x, current_ee_pose, target_ee_pose, transformed_base_state),
                x0=x0,
                method='SLSQP',
                bounds=self._get_bounds(x0),
                constraints=self._get_constraints(x0, current_ee_pose, transformed_base_state),
                options={
                    'ftol': 1e-6,
                    'maxiter': 100,
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
        
    def _cost_function(self, 
                      x: np.ndarray, 
                      current_ee_pose: Dict,
                      target_ee_pose: Dict,
                      base_state: Dict) -> float:
        """Compute cost with improved base motion compensation"""
        joint_positions = x[:6]
        joint_velocities = x[6:]
        
        # Get transformed base motion
        base_vel = base_state['linear_velocity'][:2]  # Already transformed
        base_vel_magnitude = np.linalg.norm(base_vel)
        
        # Predict end-effector position
        predicted_ee_pos = self._predict_ee_position(
            joint_positions, 
            joint_velocities,
            current_ee_pose
        )
        
        # Separate XY and Z errors
        xy_error = np.linalg.norm(predicted_ee_pos[:2] - target_ee_pose['position'][:2])
        z_error = abs(predicted_ee_pos[2] - target_ee_pose['position'][2])
        
        # Compute compensation velocities with transformed motion
        compensation_vel = -base_vel_magnitude * self.joint_scales
        
        # Combined cost with separate weights for XY and Z
        cost = (
            10.0 * xy_error +            # XY position maintenance
            20.0 * z_error +            # Strict Z position maintenance
            2.0 * np.sum((joint_velocities - compensation_vel)**2) +  # Track compensation velocities
            0.1 * np.sum(joint_velocities**2)  # Minimize overall velocity
        )
        
        return cost
        
    def _predict_ee_position(self, 
                            joint_positions: np.ndarray, 
                            joint_velocities: np.ndarray,
                            current_ee_pose: Dict) -> np.ndarray:
        """Predict end-effector position using MoveIt forward kinematics"""
        try:
            # Predict next joint positions
            predicted_joints = joint_positions + joint_velocities * self.dt
            
            # Check joint limits
            for i, pos in enumerate(predicted_joints):
                if pos < self.joint_pos_limits[i][0] or pos > self.joint_pos_limits[i][1]:
                    rospy.logwarn(f"Joint {i} position {pos} exceeds limits")
                    return current_ee_pose['position']
            
            # Use MoveIt to compute forward kinematics
            robot_state = self.move_group.get_current_state()
            robot_state.joint_state.position = predicted_joints.tolist()
            fk = self.move_group.get_robot_state_fk(robot_state)
            
            if fk is None:
                rospy.logwarn("Forward kinematics failed")
                return current_ee_pose['position']
                
            return np.array([
                fk.pose.position.x,
                fk.pose.position.y,
                fk.pose.position.z
            ])
            
        except Exception as e:
            rospy.logwarn(f"Forward kinematics error: {str(e)}")
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
    
    def _get_current_ee_pose(self, joint_positions: np.ndarray) -> Dict:
        """Get current end-effector pose using MoveIt"""
        # Set the joint positions
        self.move_group.set_joint_value_target(joint_positions)
        
        # Get the pose
        current_pose = self.move_group.get_current_pose().pose
        
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
        """Simplified constraints"""
        constraints = []
        
        # Only constrain maximum velocity
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: self.joint_vel_limits[1] - np.abs(x[6:])
        })
        
        return constraints

    def _transform_base_motion(self, base_state: Dict) -> Dict:
        """Transform base motion from mobile base frame to arm base frame"""
        # Get base motion in mobile base frame
        base_vel = base_state['linear_velocity'][:2]
        base_pos = base_state['position'][:2]
        
        # Rotation matrix for 180 degree rotation
        R = np.array([
            [-1, 0],
            [0, -1]
        ])
        
        # Transform velocities to arm base frame
        arm_base_vel = R @ base_vel
        
        # Transform position to arm base frame
        arm_base_pos = R @ base_pos + np.array([self.arm_base_offset['x'], 
                                              self.arm_base_offset['y']])
        
        # Create transformed base state
        transformed_state = base_state.copy()
        transformed_state['linear_velocity'][:2] = arm_base_vel
        transformed_state['position'][:2] = arm_base_pos
        
        return transformed_state

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