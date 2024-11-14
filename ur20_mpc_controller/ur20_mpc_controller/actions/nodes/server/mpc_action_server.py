#!/usr/bin/env python3

import rospy
import actionlib
import numpy as np
from ur20_mpc_controller.msg import MPCAction, MPCFeedback, MPCResult
from ur20_mpc_controller.models.ur_mpc import URMPC
from ur20_mpc_controller.models.base_observer import BaseObserver
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
import moveit_commander
from tf.transformations import euler_from_quaternion

SIM = True

class MPCActionServer:
    def __init__(self):
        # Initialize the action server
        self.server = actionlib.SimpleActionServer(
            'mpc_controller',
            MPCAction,
            execute_cb=self.execute_cb,
            auto_start=False
        )
        
        # Initialize MoveIt and controllers
        moveit_commander.roscpp_initialize([])
        self.move_group = moveit_commander.MoveGroupCommander("arm")
        self.mpc = URMPC()
        self.base_observer = BaseObserver()
        
        # Initialize trajectory action client
        if SIM:
            self.trajectory_client = actionlib.SimpleActionClient(
                '/arm/scaled_pos_traj_controller/follow_joint_trajectory',
                FollowJointTrajectoryAction
            )
        else:
            self.trajectory_client = actionlib.SimpleActionClient(
                '/scaled_pos_joint_traj_controller/follow_joint_trajectory',
                FollowJointTrajectoryAction
            )
            
        rospy.loginfo("Waiting for trajectory action server...")
        self.trajectory_client.wait_for_server()
        rospy.loginfo("Trajectory action server connected!")
        
        # Start the server
        self.server.start()
        rospy.loginfo("MPC Action Server is ready")
        
    def execute_cb(self, goal):
        rate = rospy.Rate(10)  # 10 Hz
        
        # Get initial poses and joint positions
        initial_pose = self.move_group.get_current_pose(
            end_effector_link="gripper_end_tool_link"
        ).pose
        
        current_joint_positions = self.move_group.get_current_joint_values()
        
        current_ee_pose = {
            'position': np.array([
                initial_pose.position.x,
                initial_pose.position.y,
                initial_pose.position.z
            ]),
            'orientation': np.array(euler_from_quaternion([
                initial_pose.orientation.x,
                initial_pose.orientation.y,
                initial_pose.orientation.z,
                initial_pose.orientation.w
            ]))
        }
        
        target_ee_pose = current_ee_pose.copy()
        
        # Main control loop
        while not rospy.is_shutdown() and not self.server.is_preempt_requested():
            current_joint_state = {
                'position': current_joint_positions,
                'velocity': np.zeros(6)
            }
            
            # Get base state from observer
            base_state = self.base_observer.get_base_state()
            
            # Compute control
            control = self.mpc.compute_control(
                current_joint_state,
                current_ee_pose,
                target_ee_pose,
                base_state
            )
            
            # Predict next joint positions
            next_joint_positions = current_joint_positions + control * 0.1
            
            # Create and send trajectory
            goal = FollowJointTrajectoryGoal()
            goal.trajectory.joint_names = [
                'shoulder_pan_joint',
                'shoulder_lift_joint',
                'elbow_joint',
                'wrist_1_joint',
                'wrist_2_joint',
                'wrist_3_joint'
            ]
            
            # Add current and next points
            current_point = JointTrajectoryPoint()
            current_point.positions = current_joint_positions
            current_point.velocities = [0.0] * 6
            current_point.time_from_start = rospy.Duration(0.0)
            
            next_point = JointTrajectoryPoint()
            next_point.positions = next_joint_positions.tolist()
            next_point.velocities = control.tolist()
            next_point.time_from_start = rospy.Duration(0.1)
            
            goal.trajectory.points = [current_point, next_point]
            
            # Send trajectory goal
            self.trajectory_client.send_goal(goal)
            
            # Send feedback
            feedback = MPCFeedback()
            feedback.current_position = current_ee_pose['position'].tolist()
            feedback.current_orientation = current_ee_pose['orientation'].tolist()
            feedback.joint_velocities = control.tolist()
            self.server.publish_feedback(feedback)
            
            # Update states
            current_joint_positions = next_joint_positions
            current_pose = self.move_group.get_current_pose(
                end_effector_link="gripper_end_tool_link"
            ).pose
            current_ee_pose['position'] = np.array([
                current_pose.position.x,
                current_pose.position.y,
                current_pose.position.z
            ])
            current_ee_pose['orientation'] = np.array(euler_from_quaternion([
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w
            ]))
            
            rate.sleep()
        
        # Set result
        result = MPCResult()
        result.success = True
        self.server.set_succeeded(result)

if __name__ == '__main__':
    rospy.init_node('mpc_action_server')
    server = MPCActionServer()
    rospy.spin()