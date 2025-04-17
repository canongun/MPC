#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from tf.transformations import quaternion_from_euler

def simulate_base_motion():
    rospy.init_node('base_motion_simulator')
    odom_pub = rospy.Publisher('/mobile_base/odom', Odometry, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    # Motion parameters
    amplitude = 0.2  # meters
    frequency = 0.1  # Hz
    
    # Initialize odometry message
    odom = Odometry()
    odom.header.frame_id = "odom"
    odom.child_frame_id = "base_link"

    t = 0.0
    while not rospy.is_shutdown():
        # Current time
        current_time = rospy.Time.now()
        
        # Compute position (sinusoidal motion in X direction)
        x = amplitude * np.sin(2 * np.pi * frequency * t)
        
        # Compute velocity
        dx = amplitude * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * t)
        
        # Create quaternion for orientation (no rotation)
        quat = quaternion_from_euler(0, 0, 0)
        
        # Set position
        odom.pose.pose = Pose(
            Point(x, 0.0, 0.0),
            Quaternion(*quat)
        )
        
        # Set velocity
        odom.twist.twist = Twist(
            Vector3(dx, 0.0, 0.0),
            Vector3(0.0, 0.0, 0.0)
        )
        
        # Set header timestamp
        odom.header.stamp = current_time
        
        # Publish odometry message
        odom_pub.publish(odom)
        
        t += 0.1  # Time increment (matches rate)
        rate.sleep()

if __name__ == '__main__':
    try:
        simulate_base_motion()
    except rospy.ROSInterruptException:
        pass