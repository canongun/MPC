#!/usr/bin/env python3

import rospy
import numpy as np
import argparse
import math
import sys # Import sys to check arguments for mixed mode
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class PlatformMover:
    def __init__(self, args):
        """
        Initializes the PlatformMover node based on parsed arguments.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        rospy.init_node('platform_displacement_mover', anonymous=True)

        self.motion_type = args.motion_type

        # Target variables initialization
        self.target_velocity_linear = 0.0
        self.target_displacement_linear = 0.0
        self.target_velocity_angular_rad = 0.0
        self.target_displacement_angular_rad = 0.0

        log_vel = ""
        log_dist = ""

        if self.motion_type == 'linear':
            if args.target_velocity <= 0 or args.target_displacement <= 0:
                raise ValueError("Target velocity and displacement must be positive for linear motion.")
            self.target_velocity_linear = args.target_velocity
            self.target_displacement_linear = args.target_displacement
            log_vel = f"{self.target_velocity_linear:.3f} m/s"
            log_dist = f"{self.target_displacement_linear:.3f} m"

        elif self.motion_type == 'angular':
            if args.target_velocity <= 0 or args.target_displacement <= 0:
                 raise ValueError("Target velocity and displacement must be positive for angular motion.")
            # User provides deg/s and deg
            self.target_velocity_angular_rad = math.radians(args.target_velocity)
            self.target_displacement_angular_rad = math.radians(args.target_displacement)
            log_vel = f"{args.target_velocity:.3f} deg/s"
            log_dist = f"{args.target_displacement:.3f} deg"

        elif self.motion_type == 'mixed':
             # Expecting 4 additional args: lin_vel, lin_dist, ang_vel, ang_dist
            if len(args.targets) != 4:
                raise ValueError("Mixed motion requires exactly 4 target values: lin_vel lin_dist ang_vel ang_dist")
            lin_vel, lin_dist, ang_vel_deg, ang_dist_deg = args.targets
            if lin_vel <= 0 or lin_dist <= 0 or ang_vel_deg <= 0 or ang_dist_deg <= 0:
                 raise ValueError("All target velocities and displacements must be positive for mixed motion.")

            self.target_velocity_linear = lin_vel
            self.target_displacement_linear = lin_dist
            self.target_velocity_angular_rad = math.radians(ang_vel_deg)
            self.target_displacement_angular_rad = math.radians(ang_dist_deg)
            log_vel = f"Linear: {self.target_velocity_linear:.3f} m/s, Angular: {ang_vel_deg:.3f} deg/s"
            log_dist = f"Linear: {self.target_displacement_linear:.3f} m, Angular: {ang_dist_deg:.3f} deg"

        else:
            # This case should not be reached due to argparse choices
            raise ValueError(f"Unsupported motion_type: {self.motion_type}")


        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)

        self.current_pose = None
        self.start_pose = None
        self.odom_received = False
        self.current_linear_displacement = 0.0
        self.current_angular_displacement = 0.0 # Store angular displacement separately

        rospy.loginfo(f"PlatformMover initialized for {self.motion_type} motion.")
        rospy.loginfo(f"Target Velocity: {log_vel}")
        rospy.loginfo(f"Target Displacement: {log_dist}")
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
        if not pose: return 0.0
        quaternion = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        )
        euler = euler_from_quaternion(quaternion)
        return euler[2] # Yaw is the third element

    def update_displacements(self):
        """Calculates linear and angular displacement from the start pose."""
        if not self.start_pose or not self.current_pose:
            return

        # Calculate linear displacement
        start_pos = np.array([self.start_pose.position.x, self.start_pose.position.y, self.start_pose.position.z])
        current_pos = np.array([self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z])
        self.current_linear_displacement = np.linalg.norm(current_pos - start_pos)

        # Calculate angular displacement (absolute yaw change)
        start_yaw = self.get_current_yaw(self.start_pose)
        current_yaw = self.get_current_yaw(self.current_pose)
        # Calculate the shortest angle difference, handling wrapping
        delta_yaw = current_yaw - start_yaw
        delta_yaw = (delta_yaw + math.pi) % (2 * math.pi) - math.pi # Normalize to [-pi, pi]
        self.current_angular_displacement = abs(delta_yaw) # Store the absolute angle rotated


    def move(self):
        """Executes the movement command."""
        rate = rospy.Rate(20) # Control loop frequency
        twist_msg = Twist()

        rospy.loginfo("Waiting for odometry...")
        while not self.odom_received and not rospy.is_shutdown():
            rate.sleep()
        if rospy.is_shutdown(): return

        rospy.loginfo("Starting movement...")
        start_time = rospy.get_time()

        linear_reached = False
        angular_reached = False

        while not rospy.is_shutdown():
            if not self.odom_received:
                rospy.logwarn_throttle(5.0, "Odometry not available, stopping motion.")
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(twist_msg)
                rate.sleep()
                continue

            # Update current displacements
            self.update_displacements()

            # Check if targets are reached individually
            if self.motion_type == 'linear' or self.motion_type == 'mixed':
                linear_reached = (self.current_linear_displacement >= self.target_displacement_linear)
                if linear_reached:
                     twist_msg.linear.x = 0.0 # Stop linear motion if reached
                else:
                     twist_msg.linear.x = self.target_velocity_linear

            if self.motion_type == 'angular' or self.motion_type == 'mixed':
                angular_reached = (self.current_angular_displacement >= self.target_displacement_angular_rad)
                if angular_reached:
                     twist_msg.angular.z = 0.0 # Stop angular motion if reached
                else:
                     twist_msg.angular.z = self.target_velocity_angular_rad # Assuming positive velocity

            # Logging progress
            log_msg = f"Displacement -> Linear: {self.current_linear_displacement:.3f}/{self.target_displacement_linear:.3f} m" \
                      f" | Angular: {math.degrees(self.current_angular_displacement):.2f}/{math.degrees(self.target_displacement_angular_rad):.2f} deg"
            rospy.logdebug(log_msg)


            # Check termination condition based on motion type
            if self.motion_type == 'linear' and linear_reached:
                rospy.loginfo(f"Linear target reached ({self.current_linear_displacement:.4f} >= {self.target_displacement_linear:.4f} m). Stopping.")
                break
            elif self.motion_type == 'angular' and angular_reached:
                rospy.loginfo(f"Angular target reached ({math.degrees(self.current_angular_displacement):.2f} >= {math.degrees(self.target_displacement_angular_rad):.2f} deg). Stopping.")
                break
            elif self.motion_type == 'mixed' and linear_reached and angular_reached:
                rospy.loginfo("Both linear and angular targets reached. Stopping.")
                break

            # Publish the command
            self.cmd_vel_pub.publish(twist_msg)
            rate.sleep()

        # Ensure the robot stops fully
        rospy.loginfo("Sending zero velocity command.")
        stop_twist = Twist()
        for _ in range(10): # Publish longer to ensure stop
            self.cmd_vel_pub.publish(stop_twist)
            rate.sleep()

        end_time = rospy.get_time()
        # Final report
        self.update_displacements() # Get final numbers
        final_log_msg = f"Movement finished in {end_time - start_time:.2f} seconds. " \
                        f"Final Displacement -> Linear: {self.current_linear_displacement:.4f} m" \
                        f" | Angular: {math.degrees(self.current_angular_displacement):.2f} deg"
        rospy.loginfo(final_log_msg)


if __name__ == '__main__':
    # Use subparsers for different motion types for clearer argument handling
    parser = argparse.ArgumentParser(description="Move the robot base by a specific displacement.")
    subparsers = parser.add_subparsers(dest='motion_type', help='Type of motion', required=True)

    # Linear motion parser
    parser_linear = subparsers.add_parser('linear', help='Move linearly')
    parser_linear.add_argument("target_velocity", type=float, help="Target linear velocity (m/s). Must be positive.")
    parser_linear.add_argument("target_displacement", type=float, help="Target linear displacement (m). Must be positive.")

    # Angular motion parser
    parser_angular = subparsers.add_parser('angular', help='Move angularly')
    parser_angular.add_argument("target_velocity", type=float, help="Target angular velocity (deg/s). Must be positive.")
    parser_angular.add_argument("target_displacement", type=float, help="Target angular displacement (deg). Must be positive.")

    # Mixed motion parser
    parser_mixed = subparsers.add_parser('mixed', help='Move linearly and angularly simultaneously')
    parser_mixed.add_argument("target_velocity_linear", type=float, help="Target linear velocity (m/s). Must be positive.")
    parser_mixed.add_argument("target_displacement_linear", type=float, help="Target linear displacement (m). Must be positive.")
    parser_mixed.add_argument("target_velocity_angular", type=float, help="Target angular velocity (deg/s). Must be positive.")
    parser_mixed.add_argument("target_displacement_angular", type=float, help="Target angular displacement (deg). Must be positive.")


    # --- Simplified Argument Parsing (Alternative if subparsers are complex) ---
    # parser = argparse.ArgumentParser(description="Move the robot base by a specific displacement.")
    # parser.add_argument("motion_type", choices=['linear', 'angular', 'mixed'], help="Type of motion.")
    # # Use nargs='+' to capture remaining arguments for mixed mode
    # parser.add_argument("targets", type=float, nargs='+', help="Target values: vel dist (for linear/angular) OR lin_vel lin_dist ang_vel ang_dist (for mixed)")
    # args = parser.parse_args()
    #
    # # Manual validation for mixed mode if not using subparsers
    # if args.motion_type == 'mixed' and len(args.targets) != 4:
    #      parser.error("Mixed motion requires exactly 4 target values: lin_vel lin_dist ang_vel ang_dist")
    # elif args.motion_type != 'mixed' and len(args.targets) != 2:
    #      parser.error("Linear/Angular motion requires exactly 2 target values: target_velocity target_displacement")
    # --------------------------------------------------------------------------

    args = parser.parse_args() # Use this line with subparsers

    try:
        # Pass the correct arguments based on the subparser used
        if args.motion_type == 'mixed':
            # Reconstruct a temporary Namespace or dict to pass to __init__
            # This is a bit clunky, maybe __init__ should take the args directly
             temp_args = argparse.Namespace(
                 motion_type='mixed',
                 targets=[args.target_velocity_linear, args.target_displacement_linear,
                          args.target_velocity_angular, args.target_displacement_angular]
             )
             mover = PlatformMover(temp_args)
        else:
             # Reconstruct Namespace for linear/angular
             temp_args = argparse.Namespace(
                 motion_type=args.motion_type,
                 target_velocity=args.target_velocity,
                 target_displacement=args.target_displacement,
                 targets=[] # Add empty targets list
             )
             mover = PlatformMover(temp_args)

        mover.move()
    except ValueError as e:
         rospy.logerr(f"Initialization or movement failed: {e}")
    except rospy.ROSInterruptException:
        rospy.loginfo("Movement interrupted by user (Ctrl+C).")
        # Stop command sending is handled implicitly by the main loop exit now
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
         # Attempt to send stop command anyway
        try:
             stop_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
             stop_twist = Twist()
             # Publish multiple times to ensure it's received
             for _ in range(5):
                 if rospy.is_shutdown(): break
                 stop_pub.publish(stop_twist)
                 rospy.sleep(0.1)
             rospy.loginfo("Attempted stop command after error.")
        except Exception as stop_e:
             rospy.logerr(f"Could not send stop command after error: {stop_e}")
