#!/usr/bin/env python3

import rospy
import numpy as np
import matplotlib.pyplot as plt
import actionlib
import moveit_commander
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from ur20_mpc_controller.models.ur_mpc import URMPC

def test_base_compensation():
    """Test the MPC controller with base motion compensation"""
    rospy.init_node('test_base_compensation')
    
    # Initialize MoveIt
    moveit_commander.roscpp_initialize([])
    robot = moveit_commander.RobotCommander()
    move_group = moveit_commander.MoveGroupCommander("arm")
    
    # Get current joint positions instead of zeros
    current_joint_positions = np.array(move_group.get_current_joint_values())
    
    # Initialize controller
    controller = URMPC()
    
    # Initialize action client
    trajectory_client = actionlib.SimpleActionClient(
        '/arm/scaled_pos_traj_controller/follow_joint_trajectory',
        FollowJointTrajectoryAction
    )
    
    rospy.loginfo("Waiting for action server...")
    trajectory_client.wait_for_server()
    rospy.loginfo("Action server connected!")
    
    rate = rospy.Rate(10)  # 10 Hz
    
    # Get initial end-effector pose
    initial_pose = move_group.get_current_pose().pose
    current_ee_pose = {
        'position': np.array([
            initial_pose.position.x,
            initial_pose.position.y,
            initial_pose.position.z
        ]),
        'orientation': np.zeros(3)
    }
    target_ee_pose = current_ee_pose.copy()  # Try to maintain initial pose
    
    # Initialize storage for plotting
    base_positions = []
    ee_positions = []
    joint_velocities = []
    
    # Initial states
    current_joint_state = {
        'position': current_joint_positions,
        'velocity': np.zeros(6)
    }
    
    times = np.arange(0, 10.0, 0.1)
    
    for t in times:
        if rospy.is_shutdown():
            break
            
        # Simulate base motion (circular path)
        base_x = 0.5 * np.cos(2 * np.pi * 0.2 * t)
        base_y = 0.5 * np.sin(2 * np.pi * 0.2 * t)
        base_vx = -2 * np.pi * 0.2 * 0.5 * np.sin(2 * np.pi * 0.2 * t)
        base_vy = 2 * np.pi * 0.2 * 0.5 * np.cos(2 * np.pi * 0.2 * t)
        
        base_state = {
            'position': np.array([base_x, base_y, 0.0]),
            'orientation': np.zeros(3),
            'linear_velocity': np.array([base_vx, base_vy, 0.0]),
            'angular_velocity': np.zeros(3)
        }
        
        # Update current joint state
        current_joint_state = {
            'position': current_joint_positions,
            'velocity': np.zeros(6)  # Use actual velocities if available
        }
        
        # Compute control
        joint_velocities_cmd = controller.compute_control(
            current_joint_state,
            current_ee_pose,
            target_ee_pose,
            base_state
        )
        
        # Predict next joint positions
        next_joint_positions = current_joint_positions + joint_velocities_cmd * 0.1
        
        # Create trajectory goal
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Add current and next positions
        current_point = JointTrajectoryPoint()
        current_point.positions = current_joint_positions.tolist()
        current_point.velocities = [0.0] * 6
        current_point.time_from_start = rospy.Duration(0.0)
        
        next_point = JointTrajectoryPoint()
        next_point.positions = next_joint_positions.tolist()
        next_point.velocities = joint_velocities_cmd.tolist()
        next_point.time_from_start = rospy.Duration(0.1)
        
        goal.trajectory.points = [current_point, next_point]
        
        # Send goal
        trajectory_client.send_goal(goal)
        
        # Store results
        base_positions.append(base_state['position'])
        ee_positions.append(current_ee_pose['position'])
        joint_velocities.append(joint_velocities_cmd)
        
        # Update current joint positions
        current_joint_positions = next_joint_positions
        current_joint_state['position'] = current_joint_positions
        current_joint_state['velocity'] = joint_velocities_cmd
        
        # Log results
        rospy.loginfo("\nMPC Test Results:")
        rospy.loginfo(f"Current EE Position: {current_ee_pose['position']}")
        rospy.loginfo(f"Target EE Position: {target_ee_pose['position']}")
        rospy.loginfo(f"Base Position: {base_state['position']}")
        rospy.loginfo(f"Computed Joint Velocities: {joint_velocities_cmd}")
        rospy.loginfo("------------------------")
        
        rate.sleep()

    # Convert to numpy arrays for plotting
    base_positions = np.array(base_positions)
    ee_positions = np.array(ee_positions)
    joint_velocities = np.array(joint_velocities)

    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Base and EE trajectories
    plt.subplot(211)
    plt.plot(base_positions[:, 0], base_positions[:, 1], 'b-', label='Base')
    plt.plot(ee_positions[:, 0], ee_positions[:, 1], 'r-', label='End-effector')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend()
    plt.title('Base and End-effector Trajectories')
    plt.grid(True)
    
    # Joint velocities
    plt.subplot(212)
    for i in range(6):
        plt.plot(times, joint_velocities[:, i], label=f'Joint {i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [rad/s]')
    plt.legend()
    plt.title('Joint Velocities')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    try:
        test_base_compensation()
    except rospy.ROSInterruptException:
        pass