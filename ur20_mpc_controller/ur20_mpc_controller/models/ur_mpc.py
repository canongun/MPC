#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.optimize import minimize
import moveit_commander
from typing import Dict, Tuple, List

from ur20_mpc_controller.models.base_estimator import BaseMotionEstimator

class URMPC:
    def __init__(self):
        # Initialize ROS node
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander()
        self.move_group = moveit_commander.MoveGroupCommander("arm")
        
        # Load parameters from config
        self.horizon = rospy.get_param('~mpc_controller/horizon', 15)
        self.dt = rospy.get_param('~mpc_controller/dt', 0.05)
        
        # Load weights
        self.w_ee_pos = rospy.get_param('~mpc_controller/weights/position', 10.0)
        self.w_ee_ori = rospy.get_param('~mpc_controller/weights/orientation', 2.0)
        self.w_control = rospy.get_param('~mpc_controller/weights/control', 0.05)
        
        # Load thresholds
        self.lin_threshold = rospy.get_param('~mpc_controller/thresholds/linear', 0.005)
        self.ang_threshold = rospy.get_param('~mpc_controller/thresholds/angular', 0.005)
        
        # Load active weights
        self.lin_weight_active = rospy.get_param('~mpc_controller/active_weights/linear', 15.0)
        self.lin_weight_idle = rospy.get_param('~mpc_controller/active_weights/linear_idle', 1.0)
        self.ang_weight_active = rospy.get_param('~mpc_controller/active_weights/angular', 5.0)
        self.ang_weight_idle = rospy.get_param('~mpc_controller/active_weights/angular_idle', 0.5)
        
        # Load bounds
        self.pos_bound = rospy.get_param('~mpc_controller/bounds/position', 0.005)
        self.ori_bound = rospy.get_param('~mpc_controller/bounds/orientation', 0.005)
        
        # Load joint scales
        self.joint_scales = np.array(rospy.get_param('~mpc_controller/joint_scales', 
            [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]))
        
        # Load robot configuration
        self.arm_base_offset = {
            'x': rospy.get_param('~mpc_controller/arm_base_offset/x', 0.06),
            'y': rospy.get_param('~mpc_controller/arm_base_offset/y', -0.1),
            'z': rospy.get_param('~mpc_controller/arm_base_offset/z', 0.09),
            'yaw': rospy.get_param('~mpc_controller/arm_base_offset/yaw', np.pi)
        }
        
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
        
        self.base_estimator = BaseMotionEstimator()
        
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
        
        # Get full Jacobian
        J = self._get_jacobian(current_joint_state['position'])
        J_pos = J[:3, :]
        J_ori = J[3:, :]
        
        # Get base velocities in arm frame (invert linear velocity)
        base_lin_vel = -transformed_base_state['linear_velocity'][:3]
        base_ang_vel = -transformed_base_state['angular_velocity']
        
        try:
            # Compute initial velocities using pseudoinverse for both position and orientation
            J_full = np.vstack([J_pos, J_ori])
            base_vel_full = np.concatenate([base_lin_vel, base_ang_vel])
            x0[6:] = np.linalg.pinv(J_full) @ base_vel_full
        except Exception as e:
            rospy.logwarn(f"Failed to compute initial guess using Jacobian: {str(e)}")
            base_vel_magnitude = np.linalg.norm(np.concatenate([base_lin_vel, base_ang_vel]))
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
        
        # Get full Jacobian (including orientation)
        J = self._get_jacobian(current_joint_state['position'])
        J_pos = J[:3, :]  # Position Jacobian
        J_ori = J[3:, :]  # Orientation Jacobian
        
        # End-effector velocities (both linear and angular)
        ee_lin_vel = J_pos @ joint_velocities
        ee_ang_vel = J_ori @ joint_velocities
        
        # Base velocities (invert both linear and angular velocities)
        base_lin_vel = -base_state['linear_velocity'][:3]    # Negative for position compensation
        base_ang_vel = -base_state['angular_velocity']       # Negative for orientation compensation
        
        # Compensation errors
        pos_compensation_error = ee_lin_vel - base_lin_vel
        ori_compensation_error = ee_ang_vel - base_ang_vel
        
        # Detect active motion directions
        active_lin_dirs = np.abs(base_lin_vel) > 0.005
        active_ang_dirs = np.abs(base_ang_vel) > 0.005
        
        # Weight compensation errors
        lin_weights = np.where(active_lin_dirs, 15.0, 1.0)
        ang_weights = np.where(active_ang_dirs, 5.0, 0.5)
        
        # Compute costs
        position_cost = np.sum((lin_weights * pos_compensation_error)**2)
        orientation_cost = np.sum((ang_weights * ori_compensation_error)**2)
        damping_cost = np.sum(joint_velocities**2) * 0.1
        
        return position_cost + self.w_ee_ori * orientation_cost + damping_cost
        
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
            
            pos_vel = ee_vel[:3]
            ori_vel = ee_vel[3:]
            
            # Invert both linear and angular velocities for proper compensation
            base_lin_vel = -base_state['linear_velocity'][:3]    # Negative for position compensation
            base_ang_vel = -base_state['angular_velocity']       # Negative for orientation compensation
            
            pos_error = pos_vel + base_lin_vel
            ori_error = ori_vel + base_ang_vel
            
            # Stricter bounds for orientation
            pos_bounds = 0.005 * np.ones(3)  # 5mm position error
            ori_bounds = 0.005 * np.ones(3)  # ~0.3 degrees orientation error
            
            return np.concatenate([
                pos_bounds - np.abs(pos_error),
                ori_bounds - np.abs(ori_error)
            ])
        
        constraints.append({
            'type': 'ineq',
            'fun': ee_motion_constraint
        })
        
        return constraints

    def _transform_base_motion(self, base_state: Dict) -> Dict:
        """Transform base motion from mobile base frame to arm base frame"""
        # Get platform velocities
        platform_vel = base_state['linear_velocity']
        platform_ang_vel = base_state['angular_velocity']
        
        # Get offset vector from rotation center to arm base
        offset = np.array([
            self.arm_base_offset['x'],
            self.arm_base_offset['y'],
            self.arm_base_offset['z']
        ])
        
        # For rotation around z-axis
        omega = platform_ang_vel[2]  # positive is counter-clockwise
        
        # Calculate tangential velocity for counter-rotation
        # When platform rotates clockwise (negative omega):
        # We need to move the end-effector in the opposite direction in world frame
        tangential_vel = -np.array([
            omega * offset[1],     # Changed sign: -ω * y
            -omega * offset[0],    # Changed sign: ω * x
            0.0
        ])
        
        # Platform linear velocity compensation
        platform_comp_vel = platform_vel
        
        # Total velocity combines both compensations
        total_vel = platform_comp_vel + tangential_vel
        
        # Debug information
        rospy.loginfo("=== Motion Transform Debug ===")
        rospy.loginfo(f"Platform angular velocity: {platform_ang_vel}")
        rospy.loginfo(f"Offset from rotation center: {offset}")
        rospy.loginfo(f"Calculated tangential velocity: {tangential_vel}")
        rospy.loginfo(f"Total compensation velocity: {total_vel}")
        rospy.loginfo("============================")
        
        transformed_state = base_state.copy()
        transformed_state['linear_velocity'] = total_vel
        transformed_state['position'] = base_state['position'] + offset
        transformed_state['angular_velocity'] = platform_ang_vel  # Keep for orientation compensation
        
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