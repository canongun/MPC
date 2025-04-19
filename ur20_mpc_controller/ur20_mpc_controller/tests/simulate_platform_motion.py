#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64 # Use Float64 for velocity command as well

# Renamed function
def simulate_platform_prismatic_velocity_motion():
    rospy.init_node('platform_prismatic_velocity_motion_simulator') # Renamed node

    # Publish to the JointVelocityController's command topic
    command_pub = rospy.Publisher(
        '/platform_prismatic_joint_velocity_controller/command', # Updated topic name
        Float64,
        queue_size=10
    )
    rate = rospy.Rate(10)  # 10 Hz

    # Motion parameters
    amplitude = 0.2  # meters (position amplitude)
    frequency = 0.1  # Hz
    omega = 2 * np.pi * frequency # Angular frequency

    # Create a Float64 message object once
    command_msg = Float64()

    start_time = rospy.get_time()
    rospy.loginfo("Starting platform prismatic velocity motion simulation...")
    while not rospy.is_shutdown():
        current_time = rospy.get_time()
        t = current_time - start_time

        # Compute desired velocity (derivative of A*sin(wt) is A*w*cos(wt))
        x_velocity_command = amplitude * omega * np.cos(omega * t)

        # Set the command value in the message
        command_msg.data = x_velocity_command

        # Publish the command message
        command_pub.publish(command_msg)
        rospy.logdebug(f"Publishing platform prismatic velocity command: {x_velocity_command:.3f}")

        rate.sleep()

if __name__ == '__main__':
    try:
        simulate_platform_prismatic_velocity_motion() # Renamed function call
    except rospy.ROSInterruptException:
        rospy.loginfo("Platform motion simulation stopped.")
