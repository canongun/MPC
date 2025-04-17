#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import JointState
import moveit_commander
import geometry_msgs.msg
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class UR20Interface:
    def __init__(self):
        # Initialize MoveIt
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "arm"  # Your MoveIt group name
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        
        # Joint states subscriber
        self.joint_states = None
        self.joint_states_sub = rospy.Subscriber(
            "/joint_states", 
            JointState, 
            self.joint_states_callback
        )
        
        # Wait for first joint state message
        self.wait_for_joint_states()
        
    def joint_states_callback(self, msg):
        """Store latest joint states"""
        self.joint_states = msg
        
    def wait_for_joint_states(self):
        """Wait until joint states are available"""
        rospy.loginfo("Waiting for joint states...")
        while self.joint_states is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Received first joint states!")
        
    def get_current_joint_states(self):
        """Get current joint positions and velocities"""
        return {
            'position': np.array(self.joint_states.position),
            'velocity': np.array(self.joint_states.velocity)
        }
        
    def get_current_ee_pose(self):
        """Get current end-effector pose"""
        current_pose = self.move_group.get_current_pose().pose
        
        # Convert to position and euler angles
        position = np.array([
            current_pose.position.x,
            current_pose.position.y,
            current_pose.position.z
        ])
        
        orientation = euler_from_quaternion([
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
            current_pose.orientation.w
        ])
        
        return {
            'position': position,
            'orientation': np.array(orientation)
        }

def main():
    """Test the UR20 interface"""
    rospy.init_node('test_ur20_interface')
    rate = rospy.Rate(10)  # 10 Hz
    
    # Initialize interface
    ur20 = UR20Interface()
    
    # Main loop
    while not rospy.is_shutdown():
        # Get current states
        joint_states = ur20.get_current_joint_states()
        ee_pose = ur20.get_current_ee_pose()
        
        # Print information
        rospy.loginfo("Joint States:")
        rospy.loginfo(f"Positions: {joint_states['position']}")
        rospy.loginfo(f"Velocities: {joint_states['velocity']}")
        rospy.loginfo("\nEnd-Effector Pose:")
        rospy.loginfo(f"Position: {ee_pose['position']}")
        rospy.loginfo(f"Orientation: {ee_pose['orientation']}")
        rospy.loginfo("------------------------")
        
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass