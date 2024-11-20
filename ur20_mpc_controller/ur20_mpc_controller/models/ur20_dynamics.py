#!/usr/bin/env python3

import numpy as np
import rospy
from typing import Dict, List, Tuple
import moveit_commander
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class UR20Dynamics:
    def __init__(self):
        """Initialize UR20 robot dynamics model"""
        # Initialize MoveIt for kinematics
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander()
        self.move_group = moveit_commander.MoveGroupCommander("arm")
        
        # Robot parameters
        self.n_joints = 6
        self.dt = rospy.get_param('~mpc_controller/dt', 0.05)
        
        # State dimensions
        self.n_q = self.n_joints          # Joint positions
        self.n_dq = self.n_joints         # Joint velocities
        self.n_ee = 6                     # End-effector pose (position + orientation)
        self.n_states = self.n_q + self.n_dq + self.n_ee
        
        # Control dimensions
        self.n_controls = self.n_joints   # Joint velocities/accelerations as control
        
        # Load joint limits
        self.joint_pos_limits = np.array([  # [min, max] for each joint
            [-2*np.pi, 2*np.pi],  # Joint 1
            [-2*np.pi, 2*np.pi],  # Joint 2
            [-2*np.pi, 2*np.pi],  # Joint 3
            [-2*np.pi, 2*np.pi],  # Joint 4
            [-2*np.pi, 2*np.pi],  # Joint 5
            [-2*np.pi, 2*np.pi]   # Joint 6
        ])
        
        self.joint_vel_limits = {
            'shoulder_pan_joint': 2.094395102393195,
            'shoulder_lift_joint': 2.094395102393195,
            'elbow_joint': 2.617993877991494,
            'wrist_1_joint': 3.665191429188092,
            'wrist_2_joint': 3.665191429188092,
            'wrist_3_joint': 3.665191429188092
        }

    def get_state_vector(self, joint_state: Dict, ee_pose: Dict) -> np.ndarray:
        """Convert joint state and EE pose to state vector"""
        q = joint_state['position']
        dq = joint_state['velocity']
        ee_pos = ee_pose['position']
        ee_ori = ee_pose['orientation']
        
        return np.concatenate([q, dq, ee_pos, ee_ori])

    def forward_kinematics(self, q: np.ndarray) -> Dict:
        """Compute forward kinematics for given joint positions"""
        # Set joint positions in MoveIt
        self.move_group.set_joint_value_target(q)
        
        # Get end-effector pose
        pose = self.move_group.get_current_pose(end_effector_link="gripper_end_tool_link").pose
        
        return {
            'position': np.array([pose.position.x, pose.position.y, pose.position.z]),
            'orientation': np.array(euler_from_quaternion([
                pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w
            ]))
        }

    def get_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Get system Jacobian at given joint positions"""
        try:
            # Convert to list if numpy array
            if isinstance(q, np.ndarray):
                q = q.tolist()
            
            # Get Jacobian from MoveIt
            J = self.move_group.get_jacobian_matrix(q)
            return np.array(J)
            
        except Exception as e:
            rospy.logerr(f"Failed to get Jacobian: {str(e)}")
            return np.eye(6)

    def predict_next_state(self, 
                          current_state: np.ndarray, 
                          control: np.ndarray,
                          dt: float = None) -> np.ndarray:
        """
        Predict next state given current state and control input
        x[k+1] = f(x[k], u[k])
        """
        if dt is None:
            dt = self.dt
            
        # Extract current state components
        q = current_state[:self.n_q]
        dq = current_state[self.n_q:self.n_q + self.n_dq]
        
        # Predict next joint positions and velocities
        q_next = q + control * dt  # Update using control input
        dq_next = control  # Direct velocity control
        
        # Get Jacobian at current configuration
        J = self.get_jacobian(q)
        
        # Compute end-effector velocity
        ee_vel = J @ control
        
        # Extract current ee pose
        ee_pos = current_state[2*self.n_q:2*self.n_q + 3]
        ee_ori = current_state[2*self.n_q + 3:2*self.n_q + 6]
        
        # Update end-effector pose
        ee_pos_next = ee_pos + ee_vel[:3] * dt
        ee_ori_next = ee_ori + ee_vel[3:] * dt
        
        # Combine into next state
        next_state = np.concatenate([
            q_next,
            dq_next,
            ee_pos_next,
            ee_ori_next
        ])
        
        return next_state

    def predict_trajectory(self, 
                         initial_state: np.ndarray,
                         control_sequence: np.ndarray,
                         horizon: int) -> np.ndarray:
        """Predict state trajectory for a sequence of controls"""
        # Initialize trajectory array
        trajectory = np.zeros((horizon + 1, self.n_states))
        trajectory[0] = initial_state
        
        # Propagate dynamics
        for k in range(horizon):
            trajectory[k + 1] = self.predict_next_state(
                trajectory[k],
                control_sequence[k]
            )
            
        return trajectory

    def check_limits(self, state: np.ndarray) -> bool:
        """Check if state is within limits"""
        # Extract joint positions and velocities
        q = state[:self.n_q]
        dq = state[self.n_q:self.n_q + self.n_dq]
        
        # Check position limits
        pos_within = np.all(q >= self.joint_pos_limits[:, 0]) and \
                    np.all(q <= self.joint_pos_limits[:, 1])
                    
        # Check velocity limits
        vel_within = np.all(np.abs(dq) <= list(self.joint_vel_limits.values()))
        
        return pos_within and vel_within

def main():
    """Test the dynamics model"""
    rospy.init_node('test_ur20_dynamics')
    
    # Initialize dynamics model
    dynamics = UR20Dynamics()
    
    # Test with dummy data
    initial_state = np.zeros(dynamics.n_states)
    control = np.zeros(dynamics.n_controls)
    
    # Predict next state
    next_state = dynamics.predict_next_state(initial_state, control)
    
    rospy.loginfo(f"Initial state: {initial_state}")
    rospy.loginfo(f"Next state: {next_state}")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass