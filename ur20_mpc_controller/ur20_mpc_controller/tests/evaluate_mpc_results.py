#!/usr/bin/env python3
import rospy
import rosbag
# Try importing BagException specifically if the direct access failed
try:
    from rosbag import BagException
except ImportError:
    # Fallback if the above doesn't work (may depend on ROS version)
    # In some setups, it might just be available directly.
    # If this still fails, we might need to catch a more general Exception.
    BagException = Exception # Assign a general exception as a fallback

import numpy as np
import argparse
from tf.transformations import quaternion_multiply, quaternion_inverse, euler_from_quaternion
import sys
import os # Import os for path manipulation

# Add matplotlib imports
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive saving (important if running headless)
import matplotlib.pyplot as plt

def quaternion_distance(q1, q2):
    """Calculates the angular distance between two quaternions.
    Result is in range [0, 1], where 0 is identical, 1 is 180 degrees apart.
    """
    q1 = np.array(q1) / np.linalg.norm(q1) # Ensure unit quaternion
    q2 = np.array(q2) / np.linalg.norm(q2) # Ensure unit quaternion
    dot_product = np.abs(np.dot(q1, q2))
    # Clip dot_product to avoid potential numerical issues with acos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    # The error metric used in the thesis is 1 - |<q_e, q_d>|
    return 1.0 - dot_product

def analyze_bag(bag_path, output_dir="mpc_plots", show_plots=False):
    """
    Analyzes a ROS bag file to calculate MPC performance metrics and generate plots.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Generate a base filename for plots from the bag file name
    base_filename = os.path.splitext(os.path.basename(bag_path))[0]

    ground_truth_topic = '/end_effector/ground_truth' # nav_msgs/Odometry
    # Correctly identify the action feedback topic type
    feedback_topic = '/mpc_controller/feedback'      # ur20_mpc_controller/MPCActionFeedback
    timing_topic = '/mpc_controller/computation_time' # std_msgs/Float64

    positions = []
    orientations = [] # Store as quaternions [x, y, z, w]
    control_inputs = []
    computation_times = []
    pose_timestamps_ros = [] # Store ROS Time objects for accurate timing
    control_timestamps_ros = [] # Store ROS Time for control inputs if available
    timing_timestamps_ros = [] # Store ROS Time for computation times

    initial_pos = None
    initial_ori_quat = None # Store initial orientation as quaternion
    first_timestamp_ros = None
    last_timestamp_ros = None

    print(f"Processing bag file: {bag_path}")
    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            # Get time range first
            try:
                 first_timestamp_ros = bag.get_start_time()
                 last_timestamp_ros = bag.get_end_time()
                 if first_timestamp_ros == last_timestamp_ros:
                     print("Warning: Bag file start and end time are identical.")
                 print(f"Bag time range: {first_timestamp_ros} to {last_timestamp_ros} (Duration: {(last_timestamp_ros - first_timestamp_ros):.3f}s)")
            except Exception as e:
                 print(f"Could not get bag time range: {e}")
                 sys.exit(1)

            # Read initial pose (using first ground truth message)
            print(f"Reading initial pose from topic: {ground_truth_topic}")
            initial_pose_found = False
            for topic, msg, t in bag.read_messages(topics=[ground_truth_topic], start_time=rospy.Time.from_sec(first_timestamp_ros)):
                 initial_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
                 initial_ori_quat = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
                 # Adjust first_timestamp_ros to the actual time of the first pose message used
                 first_timestamp_ros = t.to_sec()
                 print(f"  Initial Position (p_d): {initial_pos}")
                 print(f"  Initial Orientation (q_d): {initial_ori_quat}")
                 print(f"  Using first message at time {first_timestamp_ros:.3f} as reference.")
                 initial_pose_found = True
                 break # Got the first one

            if not initial_pose_found:
                 print(f"Error: No messages found on ground truth topic '{ground_truth_topic}' in the bag file.")
                 sys.exit(1)

            # Extract all data
            print("Extracting data...")
            for topic, msg, t in bag.read_messages(topics=[ground_truth_topic, feedback_topic, timing_topic], start_time=rospy.Time.from_sec(first_timestamp_ros)):
                timestamp_sec = t.to_sec()
                if topic == ground_truth_topic:
                    pose_timestamps_ros.append(timestamp_sec)
                    positions.append(np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]))
                    orientations.append(np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]))
                elif topic == feedback_topic:
                    # ***** FIX: Access data within the 'feedback' field *****
                    try:
                        # Ensure the feedback field exists and has the attribute
                        if hasattr(msg, 'feedback') and hasattr(msg.feedback, 'joint_velocities'):
                             control_inputs.append(np.array(msg.feedback.joint_velocities))
                             control_timestamps_ros.append(timestamp_sec) # Record timestamp for control
                        else:
                             rospy.logwarn_throttle(5.0, f"Feedback message at time {timestamp_sec} missing 'joint_velocities'.")
                    except AttributeError as e:
                         # This handles cases where the structure might be unexpected
                         rospy.logwarn_throttle(5.0, f"AttributeError accessing feedback at {timestamp_sec}: {e}")
                elif topic == timing_topic:
                    computation_times.append(msg.data)
                    timing_timestamps_ros.append(timestamp_sec) # Record timestamp for timing

    except BagException as e:
        print(f"Error reading bag file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during bag processing: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for other errors
        sys.exit(1)

    print(f"Data extracted: {len(positions)} poses, {len(control_inputs)} controls, {len(computation_times)} timings.")

    if not positions:
        print("Error: No position data extracted.")
        sys.exit(1)

    # Convert lists to numpy arrays
    positions = np.array(positions)
    orientations = np.array(orientations)
    control_inputs = np.array(control_inputs)
    computation_times = np.array(computation_times)
    # Convert timestamps relative to the start time for plotting
    pose_times = np.array(pose_timestamps_ros) - first_timestamp_ros
    control_times = np.array(control_timestamps_ros) - first_timestamp_ros
    timing_times = np.array(timing_timestamps_ros) - first_timestamp_ros

    # --- Calculate Metrics ---
    print("Calculating metrics...")
    positions = np.array(positions)
    orientations = np.array(orientations)
    control_inputs = np.array(control_inputs)
    computation_times = np.array(computation_times)
    pose_times = np.array(pose_times)
    control_times = np.array(control_times)
    timing_times = np.array(timing_times)

    # Ensure pose timestamps are sorted (usually are, but good practice)
    sort_indices = np.argsort(pose_times)
    pose_times = pose_times[sort_indices]
    positions = positions[sort_indices]
    orientations = orientations[sort_indices]
    # Note: control_inputs and computation_times are not necessarily aligned with poses,
    # their timing depends on when they were published relative to the pose.

    # Calculate Errors relative to the initial pose
    position_errors = np.linalg.norm(positions - initial_pos, axis=1)
    # Calculate individual axis errors (relative to initial world frame pose)
    position_errors_xyz = positions - initial_pos

    orientation_errors = np.array([quaternion_distance(q, initial_ori_quat) for q in orientations])

    # Calculate duration based on actual pose timestamps
    if len(pose_times) > 1:
        duration = pose_times[-1] - pose_times[0] # Duration based on pose data
    elif len(pose_times) == 1:
        duration = 0.0
        print("Warning: Only one pose timestamp found. RMS and Control Effort integral cannot be reliably calculated.")
    else: # No timestamps
         duration = -1 # Mark as invalid
         print("Warning: No pose timestamps found. Cannot calculate metrics.")

    # Initialize metrics
    e_p_max = e_p_rms = e_o_max = e_o_rms = control_effort = t_comp_avg = np.nan

    if duration >= 0: # Calculate only if duration is valid (>=0)
        if duration > 0 and len(positions) > 1:
            e_p_max = np.max(position_errors)
            e_p_rms = np.sqrt(np.trapz(position_errors**2, pose_times) / duration)
            e_o_max = np.max(orientation_errors)
            e_o_rms = np.sqrt(np.trapz(orientation_errors**2, pose_times) / duration)
        elif len(positions) == 1: # Single point case
            e_p_max = position_errors[0]
            e_p_rms = position_errors[0]
            e_o_max = orientation_errors[0]
            e_o_rms = orientation_errors[0]

        # Control Effort Integral - Using actual time intervals if possible
        if len(control_inputs) > 0 and len(control_times) > 1:
            control_norms_sq = np.linalg.norm(control_inputs, axis=1)**2
            control_dt = np.diff(control_times) # Time difference between control messages
            # Sum ||u_i||^2 * dt_i where dt_i is the time until the *next* control command
            control_effort = np.sum(control_norms_sq[:-1] * control_dt)
            print(f"  (Control effort calculated using {len(control_dt)} intervals between control inputs)")
        elif len(control_inputs) > 0:
             print("Warning: Cannot calculate precise control effort integral (need >1 control timestamp).")
             control_effort = np.nan # Cannot calculate integral properly
        else:
             print("Warning: No control input data found.")
             control_effort = np.nan

        # Average Computation Time
        if len(computation_times) > 0:
            t_comp_avg = np.mean(computation_times)
        else:
            print("Warning: No computation time data found.")
            t_comp_avg = np.nan
    else: # Invalid duration
        print("Metrics calculation skipped due to invalid time range.")

    # --- Print Results ---
    print("\n--- Performance Metrics ---")
    if duration >= 0:
        print(f"Total Duration Analyzed: {duration:.3f} seconds")
        print(f"Max Position Error (e_p_max):  {e_p_max:.6f} meters")
        print(f"RMS Position Error (e_p_rms):  {e_p_rms:.6f} meters")
        print(f"Max Orientation Error (e_o_max): {e_o_max:.6f} (quaternion dist [0,1])")
        print(f"RMS Orientation Error (e_o_rms): {e_o_rms:.6f} (quaternion dist [0,1])")
        print(f"Control Effort (E_c):        {control_effort:.6f} (approx sum ||u||^2*dt)")
        print(f"Avg Computation Time (t_comp): {t_comp_avg:.6f} seconds ({len(computation_times)} samples)")
    else:
        print("Could not calculate metrics due to errors or lack of data.")
    print("---------------------------\n")

    # --- Generate Plots ---
    if duration >= 0 and len(positions) > 0: # Only plot if data is valid
        print("Generating plots...")

        # 1. Position Error Plot
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1) # Position error magnitude
        plt.plot(pose_times, position_errors * 1000, label='Total Position Error') # Convert to mm
        plt.title(f'End-Effector Position Error ({base_filename})')
        plt.ylabel('Error (mm)')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2) # Individual axes errors
        plt.plot(pose_times, position_errors_xyz[:, 0] * 1000, label='X Error')
        plt.plot(pose_times, position_errors_xyz[:, 1] * 1000, label='Y Error')
        plt.plot(pose_times, position_errors_xyz[:, 2] * 1000, label='Z Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (mm)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f"{base_filename}_pos_error.png")
        plt.savefig(plot_filename)
        print(f"Saved position error plot to: {plot_filename}")
        if show_plots: plt.show()
        plt.close()

        # 2. Orientation Error Plot
        plt.figure(figsize=(10, 5))
        plt.plot(pose_times, orientation_errors)
        plt.title(f'End-Effector Orientation Error ({base_filename})')
        plt.xlabel('Time (s)')
        plt.ylabel('Quaternion Distance Error')
        plt.grid(True)
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f"{base_filename}_ori_error.png")
        plt.savefig(plot_filename)
        print(f"Saved orientation error plot to: {plot_filename}")
        if show_plots: plt.show()
        plt.close()

        # 3. Control Inputs Plot
        if len(control_inputs) > 0:
            plt.figure(figsize=(12, 8))
            joint_names = [ # Assuming standard UR joint order
                'Shoulder Pan', 'Shoulder Lift', 'Elbow',
                'Wrist 1', 'Wrist 2', 'Wrist 3'
            ]
            # Use control_times if available and matches length, else use index
            time_axis = control_times if len(control_times) == len(control_inputs) else np.arange(len(control_inputs))
            x_label = 'Time (s)' if len(control_times) == len(control_inputs) else 'Control Step Index'

            for i in range(control_inputs.shape[1]): # Iterate through joints (columns)
                plt.plot(time_axis, control_inputs[:, i], label=f'Joint {i+1} ({joint_names[i]})')
            plt.title(f'Control Inputs (Joint Velocities) ({base_filename})')
            plt.xlabel(x_label)
            plt.ylabel('Velocity (rad/s)')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Move legend outside
            plt.grid(True)
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
            plot_filename = os.path.join(output_dir, f"{base_filename}_control_inputs.png")
            plt.savefig(plot_filename)
            print(f"Saved control inputs plot to: {plot_filename}")
            if show_plots: plt.show()
            plt.close()
        else:
            print("Skipping control input plot (no data).")

        # 4. Computation Time Plot
        if len(computation_times) > 0:
            plt.figure(figsize=(10, 5))
            # Use timing_times if available and matches length, else use index
            time_axis = timing_times if len(timing_times) == len(computation_times) else np.arange(len(computation_times))
            x_label = 'Time (s)' if len(timing_times) == len(computation_times) else 'Computation Step Index'
            plt.plot(time_axis, np.array(computation_times) * 1000) # Convert to ms
            plt.title(f'MPC Computation Time ({base_filename})')
            plt.xlabel(x_label)
            plt.ylabel('Time (ms)')
            plt.grid(True)
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f"{base_filename}_comp_time.png")
            plt.savefig(plot_filename)
            print(f"Saved computation time plot to: {plot_filename}")
            if show_plots: plt.show()
            plt.close()
        else:
            print("Skipping computation time plot (no data).")

        # 5. End-effector Trajectory Plot (XY)
        plt.figure(figsize=(8, 8))
        plt.plot(positions[:, 0], positions[:, 1], label='Actual Trajectory')
        plt.scatter(initial_pos[0], initial_pos[1], color='red', s=100, zorder=5, label='Target/Start Point')
        plt.title(f'End-Effector Trajectory (XY Plane) ({base_filename})')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.grid(True)
        plt.legend()
        plt.axis('equal') # Ensure X and Y axes have the same scale
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, f"{base_filename}_trajectory_xy.png")
        plt.savefig(plot_filename)
        print(f"Saved XY trajectory plot to: {plot_filename}")
        if show_plots: plt.show()
        plt.close()

        print("Plot generation complete.")
    else:
        print("Skipping plot generation due to invalid duration or no data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MPC performance from a ROS bag file and generate plots.")
    parser.add_argument("bag_file", help="Path to the ROS bag file.")
    parser.add_argument("-o", "--output-dir", default="mpc_plots", help="Directory to save plots (default: mpc_plots).")
    parser.add_argument("--show", action="store_true", help="Display plots interactively instead of just saving.")
    # Add arguments for desired pose if needed in the future
    # parser.add_argument('--target-pos', nargs=3, type=float, help="Target position (x y z)")
    # parser.add_argument('--target-ori', nargs=4, type=float, help="Target orientation quaternion (x y z w)")

    args = parser.parse_args()

    # Check if rospy is needed and initialize if necessary (for logwarn_throttle)
    try:
        rospy.get_name()
    except:
        # Only init_node if it's not already running (e.g. script run standalone)
        # This is a basic check; more robust checks might be needed in complex scenarios.
        # If running within a ROS node, this init might cause issues.
        # Consider removing if rospy logging isn't strictly needed or handle initialization externally.
        # rospy.init_node('mpc_evaluator', anonymous=True, disable_signals=True)
        pass # Avoid initializing node here, rely on external ROS environment if needed for logging.

    analyze_bag(args.bag_file, args.output_dir, args.show)
    