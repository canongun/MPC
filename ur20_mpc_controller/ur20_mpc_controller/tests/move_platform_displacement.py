#!/usr/bin/env python3

import rospy
import numpy as np
import argparse
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class PlatformMover:
    def __init__(self, motion_type, target_velocity, target_displacement):
        """
        Initializes the PlatformMover node.

        Args:
            motion_type (str): 'linear' or 'angular'.
            target_velocity (float): Target linear (m/s) or angular (deg/s) velocity.
            target_displacement (float): Target linear distance (m) or angular displacement (deg).
        """
        rospy.init_node('platform_displacement_mover', anonymous=True)

        # Validate inputs
        if motion_type not in ['linear', 'angular']:
            rospy.logerr("Invalid motion_type. Must be 'linear' or 'angular'.")
            raise ValueError("Invalid motion_type")
        if target_velocity <= 0:
            rospy.logerr("Target velocity must be positive.")
            raise ValueError("Target velocity must be positive")
        if target_displacement <= 0:
             rospy.logerr("Target displacement must be positive.")
             raise ValueError("Target displacement must be positive")

        self.motion_type = motion_type
        self.target_velocity_user = target_velocity # Store user-provided velocity
        self.target_displacement_user = target_displacement # Store user-provided displacement

        # Convert angular inputs if necessary
        if self.motion_type == 'angular':
            self.target_velocity_rad = math.radians(self.target_velocity_user)
            self.target_displacement_rad = math.radians(self.target_displacement_user)
            self.unit_vel = "rad/s"
            self.unit_dist = "rad"
        else:
            self.target_velocity_rad = self.target_velocity_user # Linear is already m/s
            self.target_displacement_rad = self.target_displacement_user # Linear is already m
            self.unit_vel = "m/s"
            self.unit_dist = "m"


        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # Assuming '/odometry/filtered' provides the necessary feedback
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)

        self.current_pose = None
        self.start_pose = None
        self.odom_received = False
        self.total_displacement = 0.0

        rospy.loginfo(f"PlatformMover initialized for {self.motion_type} motion.")
        rospy.loginfo(f"Target Velocity: {self.target_velocity_user:.3f} {'deg/s' if self.motion_type == 'angular' else 'm/s'}")
        rospy.loginfo(f"Target Displacement: {self.target_displacement_user:.3f} {'deg' if self.motion_type == 'angular' else 'm'}")
        rospy.loginfo(f"Publishing to /cmd_vel, Subscribing to /odometry/filtered")

    def odom_callback(self, msg):
        """Stores the latest odometry data."""
        self.current_pose = msg.pose.pose
        if not self.odom_received:
            self.start_pose = self.current_pose
            self.odom_received = True
            rospy.loginfo("Odometry received, starting pose recorded.")

    def get_current_yaw(self, pose):
        """Extracts yaw angle from pose orientation."""
        quaternion = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        )
        euler = euler_from_quaternion(quaternion)
        return euler[2] # Yaw is the third element

    def calculate_displacement(self):
        """Calculates displacement from the start pose based on motion type."""
        if not self.start_pose or not self.current_pose:
            return 0.0

        if self.motion_type == 'linear':
            start_pos = np.array([self.start_pose.position.x, self.start_pose.position.y, self.start_pose.position.z])
            current_pos = np.array([self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z])
            return np.linalg.norm(current_pos - start_pos)

        elif self.motion_type == 'angular':
            start_yaw = self.get_current_yaw(self.start_pose)
            current_yaw = self.get_current_yaw(self.current_pose)
            # Calculate the shortest angle difference, handling wrapping
            delta_yaw = current_yaw - start_yaw
            delta_yaw = (delta_yaw + math.pi) % (2 * math.pi) - math.pi # Normalize to [-pi, pi]
            return abs(delta_yaw) # Return the absolute angle rotated

        return 0.0

    def move(self):
        """Executes the movement command."""
        rate = rospy.Rate(20) # Control loop frequency (e.g., 20 Hz)
        twist_msg = Twist()

        # Wait for the first odometry message
        rospy.loginfo("Waiting for odometry...")
        while not self.odom_received and not rospy.is_shutdown():
            rate.sleep()
        if rospy.is_shutdown(): return # Exit if ROS shuts down

        rospy.loginfo("Starting movement...")
        start_time = rospy.get_time()

        while not rospy.is_shutdown():
            if not self.odom_received:
                # Lost odometry? Hold position.
                rospy.logwarn_throttle(5.0, "Odometry not available, stopping motion.")
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(twist_msg)
                rate.sleep()
                continue

            # Calculate current displacement
            self.total_displacement = self.calculate_displacement()

            rospy.logdebug(f"Current Displacement: {self.total_displacement:.4f} {self.unit_dist} "
                          f"(Target: {self.target_displacement_rad:.4f} {self.unit_dist})")

            # Check if target displacement is reached
            if self.total_displacement >= self.target_displacement_rad:
                rospy.loginfo(f"Target displacement reached ({self.total_displacement:.4f} >= {self.target_displacement_rad:.4f} {self.unit_dist}). Stopping.")
                break # Exit the loop

            # Set the velocity in the Twist message
            if self.motion_type == 'linear':
                twist_msg.linear.x = self.target_velocity_rad
                twist_msg.angular.z = 0.0
            elif self.motion_type == 'angular':
                twist_msg.linear.x = 0.0
                # Ensure angular velocity direction matches displacement sign if needed,
                # but for simple target distance, magnitude is enough.
                # Use copysign if directional control is needed:
                # twist_msg.angular.z = math.copysign(self.target_velocity_rad, self.target_displacement_rad)
                twist_msg.angular.z = self.target_velocity_rad # Assuming positive velocity for positive displacement goal

            # Publish the command
            self.cmd_vel_pub.publish(twist_msg)
            rate.sleep()

        # Ensure the robot stops by publishing zero velocity
        rospy.loginfo("Sending zero velocity command.")
        stop_twist = Twist() # Default Twist has all zeros
        # Publish multiple times to ensure it's received
        for _ in range(5):
            self.cmd_vel_pub.publish(stop_twist)
            rate.sleep()

        end_time = rospy.get_time()
        rospy.loginfo(f"Movement finished in {end_time - start_time:.2f} seconds.")
        rospy.loginfo(f"Final displacement: {self.total_displacement:.4f} {self.unit_dist}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Move the robot base by a specific displacement (distance or angle).")
    parser.add_argument("motion_type", choices=['linear', 'angular'], help="Type of motion ('linear' or 'angular').")
    parser.add_argument("target_velocity", type=float, help="Target velocity (m/s for linear, deg/s for angular). Must be positive.")
    parser.add_argument("target_displacement", type=float, help="Target displacement (meters for linear, degrees for angular). Must be positive.")

    args = parser.parse_args()

    try:
        mover = PlatformMover(args.motion_type, args.target_velocity, args.target_displacement)
        mover.move()
    except ValueError as e:
         rospy.logerr(f"Initialization failed: {e}")
    except rospy.ROSInterruptException:
        rospy.loginfo("Movement interrupted.")
        # Attempt to send a stop command on interrupt
        try:
             stop_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
             stop_twist = Twist()
             stop_pub.publish(stop_twist)
             rospy.loginfo("Sent stop command on interrupt.")
        except Exception as stop_e:
             rospy.logerr(f"Could not send stop command on interrupt: {stop_e}")
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
         # Attempt to send a stop command on error
        try:
             stop_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
             stop_twist = Twist()
             stop_pub.publish(stop_twist)
             rospy.loginfo("Sent stop command after error.")
        except Exception as stop_e:
             rospy.logerr(f"Could not send stop command after error: {stop_e}")
