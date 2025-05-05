#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import pandas as pd
import rosbag
# Try importing BagException specifically if the direct access failed
try:
    from rosbag import BagException
except ImportError:
    # Fallback if the above doesn't work (may depend on ROS version)
    BagException = Exception # Assign a general exception as a fallback

# Add matplotlib imports
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive saving
import matplotlib.pyplot as plt

# Import tf transformations
from tf.transformations import quaternion_from_euler, quaternion_multiply, quaternion_inverse, quaternion_slerp

# Function to load and process MoCap data from TSV file
def load_mocap_data(filepath, time_col, pos_x_col, pos_y_col, pos_z_col, roll_col, pitch_col, yaw_col, skip_rows):
    """Loads MoCap data, performs unit conversions, and calculates quaternions."""
    print(f"Loading MoCap data from: {filepath}")
    try:
        # Read the TSV file using pandas
        df = pd.read_csv(filepath, sep='\t', skiprows=skip_rows)

        # --- Data Extraction and Validation ---
        required_cols = [time_col, pos_x_col, pos_y_col, pos_z_col, roll_col, pitch_col, yaw_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in MoCap file '{filepath}'. Available columns: {list(df.columns)}")

        # Convert relevant columns to numpy arrays, handling potential non-numeric data
        timestamps = pd.to_numeric(df[time_col], errors='coerce').values
        pos_x = pd.to_numeric(df[pos_x_col], errors='coerce').values
        pos_y = pd.to_numeric(df[pos_y_col], errors='coerce').values
        pos_z = pd.to_numeric(df[pos_z_col], errors='coerce').values
        roll_deg = pd.to_numeric(df[roll_col], errors='coerce').values
        pitch_deg = pd.to_numeric(df[pitch_col], errors='coerce').values
        yaw_deg = pd.to_numeric(df[yaw_col], errors='coerce').values

        # Check for NaNs introduced by coercion
        data_arrays = [timestamps, pos_x, pos_y, pos_z, roll_deg, pitch_deg, yaw_deg]
        nan_indices = [np.isnan(arr) for arr in data_arrays]
        valid_mask = ~np.logical_or.reduce(nan_indices) # Rows where ALL columns are valid numbers

        if not np.all(valid_mask):
             num_invalid = len(timestamps) - np.sum(valid_mask)
             print(f"Warning: Found {num_invalid} rows with non-numeric data in required columns. These rows will be skipped.")
             # Filter out invalid rows
             timestamps = timestamps[valid_mask]
             pos_x = pos_x[valid_mask]
             pos_y = pos_y[valid_mask]
             pos_z = pos_z[valid_mask]
             roll_deg = roll_deg[valid_mask]
             pitch_deg = pitch_deg[valid_mask]
             yaw_deg = yaw_deg[valid_mask]

        if len(timestamps) == 0:
             raise ValueError("No valid numeric data found in required MoCap columns.")

        # Ensure timestamps are monotonically increasing
        if not np.all(np.diff(timestamps) >= 0):
            print("Warning: MoCap timestamps are not strictly monotonically increasing. Attempting to sort...")
            sort_indices = np.argsort(timestamps)
            timestamps = timestamps[sort_indices]
            pos_x = pos_x[sort_indices]
            pos_y = pos_y[sort_indices]
            pos_z = pos_z[sort_indices]
            roll_deg = roll_deg[sort_indices]
            pitch_deg = pitch_deg[sort_indices]
            yaw_deg = yaw_deg[sort_indices]
            if not np.all(np.diff(timestamps) >= 0):
                 raise ValueError("MoCap timestamps still not monotonic after sorting.")


        # --- Unit Conversions ---
        positions = np.vstack([pos_x, pos_y, pos_z]).T / 1000.0 # mm to meters
        orientations_rpy_rad = np.radians(np.vstack([roll_deg, pitch_deg, yaw_deg]).T) # degrees to radians

        # --- Convert RPY to Quaternions ---
        # Assuming 'sxyz' convention (roll, pitch, yaw) - adjust if MoCap uses a different convention
        orientations_quat = np.array([quaternion_from_euler(r, p, y, 'sxyz') for r, p, y in orientations_rpy_rad])

        print(f"  Successfully loaded {len(timestamps)} valid MoCap data points.")
        return {
            "timestamps": timestamps, # Relative time
            "positions": positions, # Shape (N, 3) in meters
            "orientations_quat": orientations_quat # Shape (N, 4) as [x, y, z, w]
        }

    except FileNotFoundError:
        print(f"Error: MoCap file not found at '{filepath}'")
        sys.exit(1)
    except ValueError as e:
        print(f"Error processing MoCap file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading MoCap data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Function to load Odometry data from ROS Bag
def load_odometry_data(bag_path, odom_topic):
    """Loads Odometry data (pose and velocity) from a ROS bag file."""
    print(f"Loading Odometry data from: {bag_path} (topic: {odom_topic})")
    timestamps = []
    positions = []
    orientations_quat = []
    linear_velocities = []
    angular_velocities = []

    odom_msg_count = 0
    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            # Check if the topic exists in the bag
            topic_info = bag.get_type_and_topic_info(topic_filters=[odom_topic])
            if odom_topic not in topic_info.topics:
                print(f"Error: Odometry topic '{odom_topic}' not found in the bag file '{bag_path}'.")
                print(f"Available topics: {list(bag.get_type_and_topic_info().topics.keys())}")
                sys.exit(1)

            for topic, msg, t in bag.read_messages(topics=[odom_topic]):
                if topic == odom_topic:
                    odom_msg_count += 1
                    timestamps.append(t.to_sec())

                    # Pose
                    pos = msg.pose.pose.position
                    positions.append([pos.x, pos.y, pos.z])
                    ori = msg.pose.pose.orientation
                    # Ensure quaternion is in [x, y, z, w] format
                    orientations_quat.append([ori.x, ori.y, ori.z, ori.w])

                    # Twist (Velocity)
                    lin_vel = msg.twist.twist.linear
                    linear_velocities.append([lin_vel.x, lin_vel.y, lin_vel.z])
                    ang_vel = msg.twist.twist.angular
                    angular_velocities.append([ang_vel.x, ang_vel.y, ang_vel.z])

        if odom_msg_count == 0:
            print(f"Warning: No messages found on odometry topic '{odom_topic}' in the bag file.")
            return None # Or raise an error, depending on desired behavior

        # Convert lists to numpy arrays
        timestamps = np.array(timestamps)
        positions = np.array(positions)
        orientations_quat = np.array(orientations_quat)
        linear_velocities = np.array(linear_velocities)
        angular_velocities = np.array(angular_velocities)

        # Ensure timestamps are monotonically increasing (usually true for bags, but good check)
        if not np.all(np.diff(timestamps) >= 0):
            print("Warning: Odometry timestamps are not strictly monotonically increasing. Attempting to sort...")
            sort_indices = np.argsort(timestamps)
            timestamps = timestamps[sort_indices]
            positions = positions[sort_indices]
            orientations_quat = orientations_quat[sort_indices]
            linear_velocities = linear_velocities[sort_indices]
            angular_velocities = angular_velocities[sort_indices]
            if not np.all(np.diff(timestamps) >= 0):
                 raise ValueError("Odometry timestamps still not monotonic after sorting.")

        print(f"  Successfully loaded {odom_msg_count} Odometry messages.")
        return {
            "timestamps": timestamps, # Absolute ROS time
            "positions": positions, # Shape (N, 3)
            "orientations_quat": orientations_quat, # Shape (N, 4)
            "linear_velocities": linear_velocities, # Shape (N, 3)
            "angular_velocities": angular_velocities # Shape (N, 3)
        }

    except BagException as e:
        print(f"Error reading ROS bag file '{bag_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading Odometry data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Function for time synchronization and interpolation
def synchronize_data(mocap_data, odom_data, motion_start_pos_thresh=0.01, motion_start_vel_thresh=0.01):
    """
    Synchronizes MoCap and Odometry data based on detecting the start of motion
    and interpolates Odometry data onto MoCap timestamps.
    """
    print("Synchronizing datasets...")
    if len(mocap_data["timestamps"]) < 2 or len(odom_data["timestamps"]) < 2:
         print("Error: Not enough data points in MoCap or Odometry to synchronize.")
         return None

    # 1. Find start of motion in MoCap (based on position change)
    mocap_pos_diff = np.linalg.norm(mocap_data["positions"] - mocap_data["positions"][0], axis=1)
    start_idx_mocap = np.argmax(mocap_pos_diff > motion_start_pos_thresh)
    if start_idx_mocap == 0 and mocap_pos_diff[0] <= motion_start_pos_thresh: # Check if threshold was never exceeded
        print("Warning: Could not detect start of motion in MoCap data based on position threshold. Using first timestamp.")
        start_idx_mocap = 0 # Default to start if no motion detected
    t_start_mocap_rel = mocap_data["timestamps"][start_idx_mocap]
    print(f"  Detected MoCap motion start at relative time: {t_start_mocap_rel:.3f}s (index {start_idx_mocap})")

    # 2. Find start of motion in Odometry (based on velocity)
    odom_lin_vel_mag = np.linalg.norm(odom_data["linear_velocities"], axis=1)
    odom_ang_vel_mag = np.linalg.norm(odom_data["angular_velocities"], axis=1)
    odom_moving_mask = np.logical_or(odom_lin_vel_mag > motion_start_vel_thresh, odom_ang_vel_mag > motion_start_vel_thresh)
    start_idx_odom = np.argmax(odom_moving_mask)
    if start_idx_odom == 0 and not odom_moving_mask[0]: # Check if threshold was never exceeded
        print("Warning: Could not detect start of motion in Odometry data based on velocity threshold. Using first timestamp.")
        start_idx_odom = 0
    t_start_odom_abs = odom_data["timestamps"][start_idx_odom]
    print(f"  Detected Odometry motion start at absolute time: {t_start_odom_abs:.3f}s (index {start_idx_odom})")


    # 3. Calculate time offset and adjust MoCap timestamps
    time_offset = t_start_odom_abs - t_start_mocap_rel
    mocap_ts_abs = mocap_data["timestamps"] + time_offset
    print(f"  Calculated time offset (Odom - MoCap Start): {time_offset:.3f}s")

    # 4. Filter data to overlapping time range (based on adjusted MoCap times)
    # Find the intersection of time ranges
    common_start_time = max(mocap_ts_abs[0], odom_data["timestamps"][0])
    common_end_time = min(mocap_ts_abs[-1], odom_data["timestamps"][-1])

    # Filter MoCap data
    mocap_valid_mask_sync = (mocap_ts_abs >= common_start_time) & (mocap_ts_abs <= common_end_time)
    mocap_ts_sync = mocap_ts_abs[mocap_valid_mask_sync]
    mocap_pos_sync = mocap_data["positions"][mocap_valid_mask_sync]
    mocap_quat_sync = mocap_data["orientations_quat"][mocap_valid_mask_sync]


    # Filter Odometry data (original, for interpolation source)
    # odom_valid_mask_interp = (odom_data["timestamps"] >= common_start_time) & (odom_data["timestamps"] <= common_end_time)
    # Add buffer points outside the range if available, helps interpolation near edges
    interp_start_idx = np.searchsorted(odom_data["timestamps"], common_start_time, side='right') -1
    interp_end_idx = np.searchsorted(odom_data["timestamps"], common_end_time, side='left') + 1
    interp_start_idx = max(0, interp_start_idx) # Clamp to valid indices
    interp_end_idx = min(len(odom_data["timestamps"]), interp_end_idx) # Clamp to valid indices

    odom_ts_interp_src = odom_data["timestamps"][interp_start_idx:interp_end_idx]
    odom_pos_interp_src = odom_data["positions"][interp_start_idx:interp_end_idx]
    odom_quat_interp_src = odom_data["orientations_quat"][interp_start_idx:interp_end_idx]


    if len(mocap_ts_sync) < 2:
        print("Error: Less than 2 MoCap points found within the common time range.")
        return None
    if len(odom_ts_interp_src) < 2:
        print("Error: Less than 2 Odometry points found within the common time range for interpolation.")
        return None

    print(f"  Using common time range for analysis: {common_start_time:.3f}s to {common_end_time:.3f}s")
    print(f"  Found {len(mocap_ts_sync)} MoCap points and {len(odom_ts_interp_src)} Odometry points in this range (for interpolation).")

    # 5. Interpolate Odometry data onto filtered MoCap timestamps (mocap_ts_sync)
    print("  Interpolating Odometry data onto MoCap timestamps...")

    # Interpolate position using linear interpolation
    odom_pos_interp_x = np.interp(mocap_ts_sync, odom_ts_interp_src, odom_pos_interp_src[:, 0])
    odom_pos_interp_y = np.interp(mocap_ts_sync, odom_ts_interp_src, odom_pos_interp_src[:, 1])
    odom_pos_interp_z = np.interp(mocap_ts_sync, odom_ts_interp_src, odom_pos_interp_src[:, 2])
    odom_pos_interp = np.vstack([odom_pos_interp_x, odom_pos_interp_y, odom_pos_interp_z]).T

    # Interpolate orientation using Spherical Linear Interpolation (SLERP)
    odom_quat_interp = []
    for t_target in mocap_ts_sync:
        # Find bracketing odometry timestamps within the interpolation source
        idx_after = np.searchsorted(odom_ts_interp_src, t_target) # Index of first timestamp >= t_target

        if idx_after == 0: # Target time is before or exactly at the first odom timestamp in source
            q_interp = odom_quat_interp_src[0]
        elif idx_after == len(odom_ts_interp_src): # Target time is after or exactly at the last odom timestamp in source
            q_interp = odom_quat_interp_src[-1]
        else: # Target time is between two odometry timestamps in source
            idx_before = idx_after - 1
            t0 = odom_ts_interp_src[idx_before]
            t1 = odom_ts_interp_src[idx_after]
            q0 = odom_quat_interp_src[idx_before]
            q1 = odom_quat_interp_src[idx_after]

            # Ensure consistent quaternion path (shortest path)
            if np.dot(q0, q1) < 0:
                q1 = -q1 # Invert one quaternion if dot product is negative

            # Calculate interpolation fraction
            # Avoid division by zero if timestamps are identical
            if t1 == t0:
                 fraction = 0.0
            else:
                 fraction = (t_target - t0) / (t1 - t0)
            q_interp = quaternion_slerp(q0, q1, fraction)

        odom_quat_interp.append(q_interp)

    odom_quat_interp = np.array(odom_quat_interp)

    print("  Synchronization and interpolation complete.")
    return {
        "timestamps": mocap_ts_sync, # Synced timestamps (absolute) within common range
        "mocap_positions": mocap_pos_sync,
        "mocap_orientations_quat": mocap_quat_sync,
        "odom_positions_interp": odom_pos_interp,
        "odom_orientations_quat_interp": odom_quat_interp
    }

# Function to calculate relative motions (step-to-step changes)
def calculate_relative_motions(data):
    """Calculates step-to-step position changes and rotations."""
    print("Calculating relative motions...")
    n_points = len(data["timestamps"])
    if n_points < 2:
        print("Error: Need at least 2 synchronized data points to calculate relative motion.")
        return None

    # Timestamps for deltas (from second point onwards)
    timestamps_rel = data["timestamps"][1:]
    # Calculate dt for scaling errors (optional, but can be useful)
    delta_t = np.diff(data["timestamps"])

    # MoCap relative motions
    mocap_delta_pos = np.diff(data["mocap_positions"], axis=0)
    mocap_delta_quat = []
    for k in range(n_points - 1):
        q_inv_k = quaternion_inverse(data["mocap_orientations_quat"][k])
        delta_q = quaternion_multiply(data["mocap_orientations_quat"][k+1], q_inv_k)
        mocap_delta_quat.append(delta_q)
    mocap_delta_quat = np.array(mocap_delta_quat)

    # Odometry relative motions
    odom_delta_pos = np.diff(data["odom_positions_interp"], axis=0)
    odom_delta_quat = []
    for k in range(n_points - 1):
        q_inv_k = quaternion_inverse(data["odom_orientations_quat_interp"][k])
        delta_q = quaternion_multiply(data["odom_orientations_quat_interp"][k+1], q_inv_k)
        odom_delta_quat.append(delta_q)
    odom_delta_quat = np.array(odom_delta_quat)

    print(f"  Calculated {len(timestamps_rel)} relative motion steps.")
    return {
        "timestamps": timestamps_rel,
        "delta_t": delta_t, # Time difference between steps
        "mocap_delta_pos": mocap_delta_pos,
        "mocap_delta_quat": mocap_delta_quat,
        "odom_delta_pos": odom_delta_pos,
        "odom_delta_quat": odom_delta_quat
    }

# Function to calculate relative motion errors and statistics
def calculate_relative_errors(relative_motions):
    """Calculates the error between MoCap and Odometry relative motions."""
    print("Calculating relative motion errors...")
    timestamps = relative_motions["timestamps"]
    delta_t = relative_motions["delta_t"]
    n_steps = len(timestamps)
    if n_steps == 0:
        print("Warning: Zero relative motion steps, cannot calculate errors.")
        return None

    # Position Errors
    error_delta_pos = relative_motions["mocap_delta_pos"] - relative_motions["odom_delta_pos"] # Shape (N-1, 3)
    error_delta_pos_mag = np.linalg.norm(error_delta_pos, axis=1) # Shape (N-1,)

    # Orientation Errors
    error_delta_angle_rad = []
    for k in range(n_steps):
        # Error quaternion: mocap_delta * inv(odom_delta)
        error_quat = quaternion_multiply(
            relative_motions["mocap_delta_quat"][k],
            quaternion_inverse(relative_motions["odom_delta_quat"][k])
        )
        # Angle part of the error quaternion: 2 * acos(|w|)
        # Clip w component to avoid domain errors with arccos due to float precision
        w_component = np.clip(error_quat[3], -1.0, 1.0)
        angle = 2.0 * np.arccos(abs(w_component)) # Use abs(w) for angle magnitude
        error_delta_angle_rad.append(angle)
    error_delta_angle_rad = np.array(error_delta_angle_rad)
    error_delta_angle_deg = np.degrees(error_delta_angle_rad)

    # Calculate Statistics
    stats = {}
    # Position error stats (magnitude) - per step
    stats['pos_mag_rmse_m'] = np.sqrt(np.mean(error_delta_pos_mag**2))
    stats['pos_mag_mean_m'] = np.mean(error_delta_pos_mag)
    stats['pos_mag_std_m'] = np.std(error_delta_pos_mag)
    stats['pos_mag_max_m'] = np.max(error_delta_pos_mag)

    # Position error stats (components) - per step
    stats['pos_x_rmse_m'] = np.sqrt(np.mean(error_delta_pos[:, 0]**2))
    stats['pos_y_rmse_m'] = np.sqrt(np.mean(error_delta_pos[:, 1]**2))
    stats['pos_z_rmse_m'] = np.sqrt(np.mean(error_delta_pos[:, 2]**2))
    stats['pos_x_mean_m'] = np.mean(error_delta_pos[:, 0])
    stats['pos_y_mean_m'] = np.mean(error_delta_pos[:, 1])
    stats['pos_z_mean_m'] = np.mean(error_delta_pos[:, 2])

    # Orientation error stats (degrees) - per step
    stats['angle_rmse_deg'] = np.sqrt(np.mean(error_delta_angle_deg**2))
    stats['angle_mean_deg'] = np.mean(error_delta_angle_deg)
    stats['angle_std_deg'] = np.std(error_delta_angle_deg)
    stats['angle_max_deg'] = np.max(error_delta_angle_deg)

    # Optional: Calculate Drift Rate (error per second or per meter)
    total_time = timestamps[-1] - timestamps[0]
    # Calculate approximate distance traveled from MoCap deltas
    total_dist_mocap = np.sum(np.linalg.norm(relative_motions["mocap_delta_pos"], axis=1))

    if total_time > 1e-3: # Avoid division by zero
        stats['pos_drift_rate_m_per_s'] = np.sum(error_delta_pos_mag) / total_time
        stats['angle_drift_rate_deg_per_s'] = np.sum(error_delta_angle_deg) / total_time
    if total_dist_mocap > 1e-3:
        stats['pos_drift_rate_m_per_m'] = np.sum(error_delta_pos_mag) / total_dist_mocap
        stats['angle_drift_rate_deg_per_m'] = np.sum(error_delta_angle_deg) / total_dist_mocap


    print("  Error calculation complete.")
    return {
        "timestamps": timestamps,
        "delta_t": delta_t,
        "error_delta_pos": error_delta_pos, # (N-1, 3)
        "error_delta_pos_mag": error_delta_pos_mag, # (N-1,)
        "error_delta_angle_rad": error_delta_angle_rad, # (N-1,)
        "error_delta_angle_deg": error_delta_angle_deg, # (N-1,)
        "statistics": stats
    }

# Function to generate and save plots
def plot_results(errors_data, output_dir, base_filename):
    """Generates plots for relative position and orientation errors."""
    print("Generating plots...")
    if errors_data is None or len(errors_data["timestamps"]) == 0:
        print("  Skipping plots: No error data available.")
        return

    timestamps = errors_data["timestamps"]
    times_rel = timestamps - timestamps[0] # Relative time for plotting

    # 1. Relative Position Error Magnitude Plot
    plt.figure(figsize=(12, 6))
    plt.plot(times_rel, errors_data["error_delta_pos_mag"] * 1000) # Convert to mm
    plt.title(f'Relative Position Error Magnitude ({base_filename})')
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error per Step (mm)')
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"{base_filename}_rel_pos_error_mag.png")
    plt.savefig(plot_filename)
    print(f"  Saved plot: {plot_filename}")
    plt.close()

    # 2. Relative Orientation Error Angle Plot
    plt.figure(figsize=(12, 6))
    plt.plot(times_rel, errors_data["error_delta_angle_deg"]) # Plot in degrees
    plt.title(f'Relative Orientation Error Angle ({base_filename})')
    plt.xlabel('Time (s)')
    plt.ylabel('Orientation Error per Step (deg)')
    plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"{base_filename}_rel_ori_error_angle.png")
    plt.savefig(plot_filename)
    print(f"  Saved plot: {plot_filename}")
    plt.close()

    # 3. Relative Position Error Components Plot
    plt.figure(figsize=(12, 8))
    plt.plot(times_rel, errors_data["error_delta_pos"][:, 0] * 1000, label='X Error')
    plt.plot(times_rel, errors_data["error_delta_pos"][:, 1] * 1000, label='Y Error')
    plt.plot(times_rel, errors_data["error_delta_pos"][:, 2] * 1000, label='Z Error')
    plt.title(f'Relative Position Error Components ({base_filename})')
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error per Step (mm)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f"{base_filename}_rel_pos_error_xyz.png")
    plt.savefig(plot_filename)
    print(f"  Saved plot: {plot_filename}")
    plt.close()

    print("Plot generation complete.")


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Analyze state estimation performance by comparing ROS bag odometry data with MoCap ground truth (.tsv).")
    parser.add_argument("mocap_file", help="Path to the MoCap data file (.tsv).")
    parser.add_argument("bag_file", help="Path to the ROS bag file.")
    parser.add_argument("--odom-topic", default="/odometry/filtered", help="ROS topic for the estimated odometry (default: /odometry/filtered).")
    parser.add_argument("-o", "--output-dir", default="state_estimation_plots", help="Directory to save plots (default: state_estimation_plots).")
    parser.add_argument("--mocap-time-col", default="Time", help="Column name for timestamps in the MoCap file.")
    parser.add_argument("--mocap-pos-x-col", default="mobile_platform X", help="Column name for MoCap X position.")
    parser.add_argument("--mocap-pos-y-col", default="mobile_platform Y", help="Column name for MoCap Y position.")
    parser.add_argument("--mocap-pos-z-col", default="mobile_platform Z", help="Column name for MoCap Z position.")
    parser.add_argument("--mocap-roll-col", default="Roll", help="Column name for MoCap Roll.")
    parser.add_argument("--mocap-pitch-col", default="Pitch", help="Column name for MoCap Pitch.")
    parser.add_argument("--mocap-yaw-col", default="Yaw", help="Column name for MoCap Yaw.")
    parser.add_argument("--skip-rows", type=int, default=14, help="Number of header rows to skip in the MoCap TSV file (default: 14 based on screenshot).") # Adjust if your header changes
    parser.add_argument("--motion-start-pos-thresh", type=float, default=0.01, help="Position change (m) threshold to detect MoCap motion start (default: 0.01).")
    parser.add_argument("--motion-start-vel-thresh", type=float, default=0.01, help="Velocity magnitude (m/s or rad/s) threshold to detect Odom motion start (default: 0.01).")


    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # Create a base filename for plots from input filenames
    mocap_basename = os.path.splitext(os.path.basename(args.mocap_file))[0]
    bag_basename = os.path.splitext(os.path.basename(args.bag_file))[0]
    plot_base_filename = f"{mocap_basename}_vs_{bag_basename}"

    print("Starting State Estimation Evaluation...")
    print(f"  MoCap File: {args.mocap_file}")
    print(f"  Bag File: {args.bag_file}")
    print(f"  Odometry Topic: {args.odom_topic}")
    print(f"  Output Directory: {args.output_dir}")

    # --- Load MoCap Data ---
    mocap_data = load_mocap_data(
        args.mocap_file,
        args.mocap_time_col,
        args.mocap_pos_x_col, args.mocap_pos_y_col, args.mocap_pos_z_col,
        args.mocap_roll_col, args.mocap_pitch_col, args.mocap_yaw_col,
        args.skip_rows
    )
    if mocap_data is None:
        print("Error: Failed to load MoCap data.")
        sys.exit(1)

    # --- Load Odometry Data ---
    odom_data = load_odometry_data(args.bag_file, args.odom_topic)
    if odom_data is None:
        print("Error: Failed to load Odometry data.")
        sys.exit(1)

    # --- Time Synchronization & Interpolation ---
    synchronized_data = synchronize_data(mocap_data, odom_data,
                                         args.motion_start_pos_thresh,
                                         args.motion_start_vel_thresh)
    if synchronized_data is None:
        print("Error: Failed to synchronize data.")
        sys.exit(1)

    # --- Calculate Relative Motions ---
    relative_motions = calculate_relative_motions(synchronized_data)
    if relative_motions is None:
        print("Error: Failed to calculate relative motions.")
        sys.exit(1)

    # --- Calculate Relative Errors & Statistics ---
    relative_errors = calculate_relative_errors(relative_motions)
    if relative_errors is None:
        print("Error: Failed to calculate relative errors.")
        sys.exit(1)

    # --- Print Statistics ---
    print("\n--- Relative Motion Error Statistics (Per Step) ---")
    stats = relative_errors["statistics"]
    print("Position Error (Magnitude):")
    print(f"  RMSE: {stats.get('pos_mag_rmse_m', float('nan')) * 1000:.3f} mm") # Convert to mm for printing
    print(f"  Mean: {stats.get('pos_mag_mean_m', float('nan')) * 1000:.3f} mm")
    print(f"  Std Dev: {stats.get('pos_mag_std_m', float('nan')) * 1000:.3f} mm")
    print(f"  Max: {stats.get('pos_mag_max_m', float('nan')) * 1000:.3f} mm")
    print("Position Error (Components - RMSE):")
    print(f"  X RMSE: {stats.get('pos_x_rmse_m', float('nan')) * 1000:.3f} mm")
    print(f"  Y RMSE: {stats.get('pos_y_rmse_m', float('nan')) * 1000:.3f} mm")
    print(f"  Z RMSE: {stats.get('pos_z_rmse_m', float('nan')) * 1000:.3f} mm")
    print("Orientation Error:")
    print(f"  RMSE: {stats.get('angle_rmse_deg', float('nan')):.4f} deg")
    print(f"  Mean: {stats.get('angle_mean_deg', float('nan')):.4f} deg")
    print(f"  Std Dev: {stats.get('angle_std_deg', float('nan')):.4f} deg")
    print(f"  Max: {stats.get('angle_max_deg', float('nan')):.4f} deg")
    # Print drift rates if calculated
    if 'pos_drift_rate_m_per_s' in stats:
        print("Drift Rates (Cumulative Error / Duration or Distance):")
        print(f"  Position Drift / sec: {stats.get('pos_drift_rate_m_per_s', float('nan')) * 1000:.3f} mm/s")
        print(f"  Position Drift / m:   {stats.get('pos_drift_rate_m_per_m', float('nan')) * 1000:.3f} mm/m")
        print(f"  Angle Drift / sec:    {stats.get('angle_drift_rate_deg_per_s', float('nan')):.4f} deg/s")
        print(f"  Angle Drift / m:      {stats.get('angle_drift_rate_deg_per_m', float('nan')):.4f} deg/m")
    print("-------------------------------------------------\n")

    # --- Generate Plots ---
    plot_results(relative_errors, args.output_dir, plot_base_filename)

    print("State estimation evaluation finished.")

if __name__ == "__main__":
    main()
