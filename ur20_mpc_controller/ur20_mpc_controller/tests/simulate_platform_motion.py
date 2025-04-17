#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64 # Use Float64 for position command

def simulate_platform_x_motion():
    rospy.init_node('platform_x_motion_simulator')

    # Publish to the JointPositionController's command topic
    command_pub = rospy.Publisher(
        '/platform_x_joint_position_controller/command',
        Float64,
        queue_size=10
    )
    rate = rospy.Rate(10)  # 10 Hz

    # Motion parameters
    amplitude = 0.2  # meters (gives 0.4m total range: -0.2m to +0.2m)
    frequency = 0.1  # Hz

    # Create a Float64 message object once
    command_msg = Float64()

    t = 0.0
    rospy.loginfo("Starting platform X motion simulation...")
    while not rospy.is_shutdown():
        # Compute desired position (sinusoidal motion in X direction)
        x_position_command = amplitude * np.sin(2 * np.pi * frequency * t)

        # Set the command value in the message
        command_msg.data = x_position_command

        # Publish the command message
        command_pub.publish(command_msg)
        rospy.logdebug(f"Publishing platform_x command: {x_position_command:.3f}")

        t += 0.1  # Time increment (matches rate)
        rate.sleep()

if __name__ == '__main__':
    try:
        simulate_platform_x_motion()
    except rospy.ROSInterruptException:
        rospy.loginfo("Platform motion simulation stopped.")
