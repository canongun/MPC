#!/usr/bin/env python3

import rospy
import numpy as np
import argparse
import math
from geometry_msgs.msg import Twist

class SinusoidalMover:
    """
    Generates and publishes sinusoidal velocity commands for the robot's mobile base.
    This script is designed to execute the specific motion profiles required for
    the hardware validation described in the user's thesis.
    """
    def __init__(self, args):
        """
        Initializes the SinusoidalMover node.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        rospy.init_node('platform_sinusoidal_mover', anonymous=True)

        self.motion_type = args.motion_type
        self.duration = args.duration
        self.amplitude_lin = 0.0
        self.frequency_lin = 0.0
        self.amplitude_ang_rad = 0.0
        self.frequency_ang = 0.0

        # Configure parameters based on the selected motion type
        if self.motion_type in ['linear_x', 'linear_y']:
            self.amplitude_lin = args.amplitude
            self.frequency_lin = args.frequency
            rospy.loginfo(f"Initialized for {self.motion_type} motion: Amp={self.amplitude_lin:.3f} m/s, Freq={self.frequency_lin:.3f} Hz")
        elif self.motion_type == 'angular':
            self.amplitude_ang_rad = math.radians(args.amplitude) # Convert user input from deg/s to rad/s
            self.frequency_ang = args.frequency
            rospy.loginfo(f"Initialized for angular motion: Amp={args.amplitude:.3f} deg/s, Freq={self.frequency_ang:.3f} Hz")
        elif self.motion_type == 'combined':
            self.amplitude_lin = args.amplitude_lin
            self.frequency_lin = args.frequency_lin
            self.amplitude_ang_rad = math.radians(args.amplitude_ang) # Convert deg/s to rad/s
            self.frequency_ang = args.frequency_ang
            rospy.loginfo(f"Initialized for combined motion: Lin Amp={self.amplitude_lin:.3f} m/s, Lin Freq={self.frequency_lin:.3f} Hz | Ang Amp={args.amplitude_ang:.3f} deg/s, Ang Freq={self.frequency_ang:.3f} Hz")
        else:
            raise ValueError(f"Unsupported motion type: {self.motion_type}")

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.loginfo(f"Will publish to /cmd_vel for {self.duration:.1f} seconds.")

    def move(self):
        """
        Executes the sinusoidal movement loop for the specified duration.
        """
        rate = rospy.Rate(50) # Control loop frequency (50 Hz)
        start_time = rospy.get_time()
        rospy.loginfo("Starting sinusoidal movement...")

        while not rospy.is_shutdown() and (rospy.get_time() - start_time) < self.duration:
            elapsed_time = rospy.get_time() - start_time
            twist_msg = Twist()

            # Generate velocity based on the sinusoidal profile v(t) = A * sin(2*pi*f*t)
            # This follows the primary definition in the thesis "Linear Motion Test Cases" section.
            if self.motion_type in ['linear_x', 'linear_y', 'combined']:
                omega_lin = 2 * math.pi * self.frequency_lin
                velocity_lin = self.amplitude_lin * math.sin(omega_lin * elapsed_time)
                if self.motion_type == 'linear_x':
                    twist_msg.linear.x = velocity_lin
                elif self.motion_type == 'linear_y':
                    twist_msg.linear.y = velocity_lin
                elif self.motion_type == 'combined':
                    # Per thesis, combined motion has linear (X) and angular (Z) components.
                    twist_msg.linear.x = velocity_lin

            if self.motion_type in ['angular', 'combined']:
                omega_ang = 2 * math.pi * self.frequency_ang
                velocity_ang = self.amplitude_ang_rad * math.sin(omega_ang * elapsed_time)
                twist_msg.angular.z = velocity_ang

            self.cmd_vel_pub.publish(twist_msg)
            rate.sleep()

        # Command the robot to stop after the duration is reached
        rospy.loginfo(f"Duration of {self.duration:.1f}s reached. Sending stop command.")
        stop_twist = Twist()
        # Publish for a short duration to ensure the stop command is received
        for _ in range(10):
            if rospy.is_shutdown(): break
            self.cmd_vel_pub.publish(stop_twist)
            rate.sleep()
        rospy.loginfo("Movement finished.")

if __name__ == '__main__':
    # Use subparsers for clear command-line interfaces for each motion type
    parser = argparse.ArgumentParser(description="Move the robot base with sinusoidal velocity profiles for hardware testing.")
    subparsers = parser.add_subparsers(dest='motion_type', help='Type of sinusoidal motion', required=True)

    # --- Linear X Motion ---
    parser_lx = subparsers.add_parser('linear_x', help='Sinusoidal motion along the X axis.')
    parser_lx.add_argument("amplitude", type=float, help="Velocity amplitude (m/s).")
    parser_lx.add_argument("frequency", type=float, help="Oscillation frequency (Hz).")
    parser_lx.add_argument("duration", type=float, help="Duration of the movement (s).")

    # --- Linear Y Motion ---
    parser_ly = subparsers.add_parser('linear_y', help='Sinusoidal motion along the Y axis.')
    parser_ly.add_argument("amplitude", type=float, help="Velocity amplitude (m/s).")
    parser_ly.add_argument("frequency", type=float, help="Oscillation frequency (Hz).")
    parser_ly.add_argument("duration", type=float, help="Duration of the movement (s).")

    # --- Angular Z Motion ---
    parser_az = subparsers.add_parser('angular', help='Sinusoidal rotation around the Z axis.')
    parser_az.add_argument("amplitude", type=float, help="Angular velocity amplitude (deg/s).")
    parser_az.add_argument("frequency", type=float, help="Oscillation frequency (Hz).")
    parser_az.add_argument("duration", type=float, help="Duration of the movement (s).")

    # --- Combined Linear X and Angular Z Motion ---
    parser_comb = subparsers.add_parser('combined', help='Combined sinusoidal linear (X) and angular (Z) motion.')
    parser_comb.add_argument("amplitude_lin", type=float, help="Linear velocity amplitude (m/s).")
    parser_comb.add_argument("frequency_lin", type=float, help="Linear oscillation frequency (Hz).")
    parser_comb.add_argument("amplitude_ang", type=float, help="Angular velocity amplitude (deg/s).")
    parser_comb.add_argument("frequency_ang", type=float, help="Angular oscillation frequency (Hz).")
    parser_comb.add_argument("duration", type=float, help="Duration of the movement (s).")

    args = parser.parse_args()

    try:
        mover = SinusoidalMover(args)
        mover.move()
    except rospy.ROSInterruptException:
        rospy.loginfo("Movement interrupted by user (Ctrl+C).")
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() 