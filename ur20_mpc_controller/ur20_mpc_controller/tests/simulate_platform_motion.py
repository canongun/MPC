#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64

def simulate_platform_velocity_motion():
    rospy.init_node('platform_velocity_motion_simulator')

    # Get motion type from parameter server (default to linear)
    motion_type = rospy.get_param("~motion_type", "linear") # Added parameter reading

    # --- Conditional Setup based on motion_type ---
    if motion_type == "angular":
        controller_name = "platform_revolute_joint_velocity_controller"
        # Motion parameters for angular
        amplitude = np.radians(20.0) # Amplitude in radians (e.g., 30 degrees)
        unit = "rad/s"
        motion_description = "angular (revolute)"
    elif motion_type == "linear":
        controller_name = "platform_prismatic_joint_velocity_controller"
        # Motion parameters for linear
        amplitude = 0.2  # Amplitude in meters
        unit = "m/s"
        motion_description = "linear (prismatic)"
    else:
        rospy.logerr(f"Invalid motion_type '{motion_type}'. Use 'linear' or 'angular'.")
        return
    # --- End Conditional Setup ---

    topic_name = f"/{controller_name}/command"

    # Publish to the determined JointVelocityController's command topic
    command_pub = rospy.Publisher(
        topic_name,
        Float64,
        queue_size=10
    )
    rate = rospy.Rate(10)  # 10 Hz

    # Common motion parameters
    frequency = 0.1  # Hz
    omega = 2 * np.pi * frequency # Angular frequency

    # Create a Float64 message object once
    command_msg = Float64()

    start_time = rospy.get_time()
    rospy.loginfo(f"Starting platform {motion_description} velocity motion simulation...")
    rospy.loginfo(f"Publishing to topic: {topic_name}")
    rospy.loginfo(f"Amplitude: {amplitude:.3f} {'rad' if motion_type == 'angular' else 'm'}, Frequency: {frequency} Hz")

    while not rospy.is_shutdown():
        current_time = rospy.get_time()
        t = current_time - start_time

        # Compute desired velocity (derivative of A*sin(wt) is A*w*cos(wt))
        # Calculation is the same, interpretation depends on amplitude unit
        velocity_command = amplitude * omega * np.cos(omega * t)

        # Set the command value in the message
        command_msg.data = velocity_command

        # Publish the command message
        command_pub.publish(command_msg)
        rospy.logdebug(f"Publishing platform {motion_description} velocity command: {velocity_command:.3f} {unit}")

        rate.sleep()

if __name__ == '__main__':
    try:
        simulate_platform_velocity_motion() # Renamed function call
    except rospy.ROSInterruptException:
        rospy.loginfo("Platform motion simulation stopped.")
    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")
