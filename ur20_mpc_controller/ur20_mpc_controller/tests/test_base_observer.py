#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class BaseObserver:
    def __init__(self):
        # Initialize subscriber
        self.odom_sub = rospy.Subscriber('/mobile_base/odom', Odometry, self.odom_callback)
        
        # State variables
        self.base_position = np.zeros(3)  # [x, y, z]
        self.base_orientation = np.zeros(3)  # [roll, pitch, yaw]
        self.base_linear_vel = np.zeros(3)  # [vx, vy, vz]
        self.base_angular_vel = np.zeros(3)  # [wx, wy, wz]
        
        # Status flag
        self.has_received_odom = False

    def odom_callback(self, msg):
        # Extract position
        self.base_position[0] = msg.pose.pose.position.x
        self.base_position[1] = msg.pose.pose.position.y
        self.base_position[2] = msg.pose.pose.position.z

        # Extract orientation and convert to numpy array
        quaternion = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        self.base_orientation = np.array(euler_from_quaternion(quaternion))  # Store as numpy array

        # Extract velocities
        self.base_linear_vel[0] = msg.twist.twist.linear.x
        self.base_linear_vel[1] = msg.twist.twist.linear.y
        self.base_linear_vel[2] = msg.twist.twist.linear.z
        
        self.base_angular_vel[0] = msg.twist.twist.angular.x
        self.base_angular_vel[1] = msg.twist.twist.angular.y
        self.base_angular_vel[2] = msg.twist.twist.angular.z

        self.has_received_odom = True

    def get_base_state(self):
        """Returns the current base state as a dict"""
        return {
            'position': self.base_position.copy(),
            'orientation': np.array(self.base_orientation).copy(),
            'linear_velocity': self.base_linear_vel.copy(),
            'angular_velocity': self.base_angular_vel.copy()
        }

    def is_ready(self):
        """Check if we have received odometry data"""
        return self.has_received_odom

def main():
    rospy.init_node('test_base_observer')
    rate = rospy.Rate(10)  # 10 Hz for testing

    # Initialize base observer
    observer = BaseObserver()
    
    # Wait for first odometry message
    rospy.loginfo("Waiting for odometry data...")
    while not observer.is_ready() and not rospy.is_shutdown():
        rate.sleep()
    
    rospy.loginfo("Received first odometry message!")

    # Main loop
    while not rospy.is_shutdown():
        # Get current base state
        state = observer.get_base_state()
        
        # Print state information
        rospy.loginfo("Base State:")
        rospy.loginfo(f"Position (x,y,z): {state['position']}")
        rospy.loginfo(f"Orientation (r,p,y): {state['orientation']}")
        rospy.loginfo(f"Linear Velocity: {state['linear_velocity']}")
        rospy.loginfo(f"Angular Velocity: {state['angular_velocity']}")
        rospy.loginfo("------------------------")
        
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass