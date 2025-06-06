#!/usr/bin/env python3

import argparse
import os
import sys
import numpy as np
import pandas as pd
import rosbag
from scipy.interpolate import interp1d
from tf.transformations import quaternion_from_euler

# Add matplotlib imports
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive saving
import matplotlib.pyplot as plt

# --- Data Loading Functions ---

def load_mocap_data(filepath, time_col, pos_x_col, pos_y_col, pos_z_col, roll_col, pitch_col, yaw_col, header_row):
    """Loads MoCap ground truth data for the end-effector from a TSV file."""
    print(f"Loading MoCap data from: {filepath}")
    try:
        df = pd.read_csv(filepath, sep='\t', header=header_row)
        required_cols = [time_col, pos_x_col, pos_y_col, pos_z_col, roll_col, pitch_col, yaw_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in MoCap file '{filepath}'. Available columns: {list(df.columns)}")

        timestamps = pd.to_numeric(df[time_col], errors='coerce').values
        pos_x = pd.to_numeric(df[pos_x_col], errors='coerce').values
        pos_y = pd.to_numeric(df[pos_y_col], errors='coerce').values
        pos_z = pd.to_numeric(df[pos_z_col], errors='coerce').values
        # Load orientation as Roll, Pitch, Yaw in degrees
        roll_deg = pd.to_numeric(df[roll_col], errors='coerce').values
        pitch_deg = pd.to_numeric(df[pitch_col], errors='coerce').values
        yaw_deg = pd.to_numeric(df[yaw_col], errors='coerce').values


        # Filter out rows with non-numeric data
        valid_mask = ~np.isnan(timestamps) & ~np.isnan(pos_x) & ~np.isnan(roll_deg)
        if not np.all(valid_mask):
            num_invalid = len(timestamps) - np.sum(valid_mask)
            print(f"Warning: Found {num_invalid} rows with non-numeric data. These will be skipped.")
            # Apply mask to all data arrays
            timestamps, pos_x, pos_y, pos_z = timestamps[valid_mask], pos_x[valid_mask], pos_y[valid_mask], pos_z[valid_mask]
            roll_deg, pitch_deg, yaw_deg = roll_deg[valid_mask], pitch_deg[valid_mask], yaw_deg[valid_mask]

        # Convert positions to meters (assuming mm input)
        positions = np.vstack([pos_x, pos_y, pos_z]).T / 1000.0
        
        # Convert RPY (degrees) to Quaternions (radians)
        orientations_rpy_rad = np.radians(np.vstack([roll_deg, pitch_deg, yaw_deg]).T)
        # Assuming 'sxyz' convention (static frame, XYZ axes) for Roll, Pitch, Yaw. Adjust if your MoCap system uses a different convention.
        quaternions = np.array([quaternion_from_euler(r, p, y, 'sxyz') for r, p, y in orientations_rpy_rad])

        print(f"  Successfully loaded {len(timestamps)} valid MoCap data points.")
        return {"timestamps": timestamps, "positions": positions, "quaternions": quaternions}

    except FileNotFoundError:
        print(f"Error: MoCap file not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading MoCap data: {e}")
        sys.exit(1)

def load_rosbag_data(bag_path, feedback_topic, time_topic, cmd_vel_topic):
    """Loads MPC feedback, computation time, and command velocities from a ROS bag."""
    print(f"Loading ROS bag data from: {bag_path}")
    feedback_ts, control_inputs, time_ts, comp_times = [], [], [], []
    cmd_vel_ts, cmd_vels = [], []

    try:
        with rosbag.Bag(bag_path, 'r') as bag:
            # Check for topics
            topics_in_bag = bag.get_type_and_topic_info().topics
            if feedback_topic not in topics_in_bag:
                print(f"Warning: Feedback topic '{feedback_topic}' not found in bag.")
            if time_topic not in topics_in_bag:
                 print(f"Warning: Computation time topic '{time_topic}' not found in bag.")
            if cmd_vel_topic not in topics_in_bag:
                 print(f"Warning: Command velocity topic '{cmd_vel_topic}' not found in bag. Motion-based sync will fail.")

            for topic, msg, t in bag.read_messages(topics=[feedback_topic, time_topic, cmd_vel_topic]):
                if topic == feedback_topic:
                    # Assuming feedback message is from MPC.action and has joint_velocities
                    if hasattr(msg, 'feedback') and hasattr(msg.feedback, 'joint_velocities'):
                        feedback_ts.append(t.to_sec())
                        control_inputs.append(msg.feedback.joint_velocities)
                    else:
                        # Use rospy.logwarn_throttle if in a ROS node, otherwise print
                        print(f"Warning: Feedback message on {feedback_topic} does not have expected structure '.feedback.joint_velocities'. Skipping.")

                elif topic == time_topic:
                    time_ts.append(t.to_sec())
                    comp_times.append(msg.data)
                
                elif topic == cmd_vel_topic:
                    cmd_vel_ts.append(t.to_sec())
                    cmd_vels.append([msg.linear.x, msg.linear.y, msg.angular.z])

        
        print(f"  Loaded {len(control_inputs)} control inputs, {len(comp_times)} computation time messages, and {len(cmd_vels)} cmd_vel messages.")
        return {
            "feedback_ts": np.array(feedback_ts),
            "control_inputs": np.array(control_inputs),
            "time_ts": np.array(time_ts),
            "comp_times": np.array(comp_times),
            "cmd_vel_ts": np.array(cmd_vel_ts),
            "cmd_vels": np.array(cmd_vels),
        }
    except Exception as e:
        print(f"An unexpected error occurred while loading rosbag data: {e}")
        sys.exit(1)

# --- Analysis and Calculation Functions ---

def analyze_performance(mocap_data, rosbag_data, motion_start_thresh):
    """
    Calculates all performance metrics based on MoCap and rosbag data.
    """
    print("Analyzing performance metrics...")
    
    # 1. Set Desired Pose from first MoCap measurement
    p_d = mocap_data["positions"][0]
    q_d = mocap_data["quaternions"][0]
    print(f"  Desired Position (p_d) set to: {p_d}")
    print(f"  Desired Orientation (q_d) set to: {q_d}")

    # 2. Synchronize Time using Start of Motion
    # Find start of motion in MoCap data (positional change)
    mocap_pos_diff = np.linalg.norm(mocap_data["positions"] - p_d, axis=1)
    start_idx_mocap = np.argmax(mocap_pos_diff > motion_start_thresh)
    if start_idx_mocap == 0 and mocap_pos_diff[0] <= motion_start_thresh:
        print("Warning: Could not detect motion start in MoCap data. Using first timestamp.")
        t_start_mocap_rel = mocap_data["timestamps"][0]
    else:
        t_start_mocap_rel = mocap_data["timestamps"][start_idx_mocap]
    print(f"  Detected MoCap motion start at relative time: {t_start_mocap_rel:.3f}s")

    # Find start of motion in ROS bag data (non-zero /cmd_vel)
    if len(rosbag_data["cmd_vel_ts"]) > 0:
        cmd_vel_mag = np.linalg.norm(rosbag_data["cmd_vels"], axis=1)
        start_idx_ros = np.argmax(cmd_vel_mag > 1e-4) # Find first non-zero command
        if start_idx_ros == 0 and cmd_vel_mag[0] <= 1e-4:
             print("Warning: Could not detect motion start in /cmd_vel data. Using first bag timestamp.")
             t_start_ros_abs = rosbag_data["cmd_vel_ts"][0]
        else:
             t_start_ros_abs = rosbag_data["cmd_vel_ts"][start_idx_ros]
        print(f"  Detected ROS motion command start at absolute time: {t_start_ros_abs:.3f}s")
    else:
        print("Error: No /cmd_vel messages found. Cannot synchronize based on motion start. Aborting.")
        return None
        
    # Calculate offset and create absolute mocap timestamps
    time_offset = t_start_ros_abs - t_start_mocap_rel
    mocap_ts_abs = mocap_data["timestamps"] + time_offset
    print(f"  Calculated time offset (ROS Start - MoCap Start): {time_offset:.3f}s")

    # Determine common time range using the master ROS timeline
    # Use feedback timestamps as the primary data source from the bag
    common_start_time = max(mocap_ts_abs[0], rosbag_data["feedback_ts"][0])
    common_end_time = min(mocap_ts_abs[-1], rosbag_data["feedback_ts"][-1])
    
    # Filter MoCap data to common time range
    valid_mask = (mocap_ts_abs >= common_start_time) & (mocap_ts_abs <= common_end_time)
    timestamps = mocap_ts_abs[valid_mask]
    positions = mocap_data["positions"][valid_mask]
    quaternions = mocap_data["quaternions"][valid_mask]

    if len(timestamps) < 2:
        print("Error: Not enough overlapping data between MoCap and rosbag. Cannot analyze.")
        return None

    # 3. Calculate Errors
    # Position Error
    e_p = np.linalg.norm(positions - p_d, axis=1) # Shape (N,)
    e_p_xyz = positions - p_d # Shape (N, 3)

    # Orientation Error
    # e_o(t) = 1 - |<q_e(t), q_d>|
    dot_product = np.abs(np.sum(quaternions * q_d, axis=1))
    dot_product = np.clip(dot_product, -1.0, 1.0) # Clip for safety
    e_o = 1 - dot_product # Shape (N,)

    # 4. Interpolate Rosbag data onto the analysis timestamps
    # Control Inputs
    interp_u = interp1d(rosbag_data["feedback_ts"], rosbag_data["control_inputs"], axis=0, bounds_error=False, fill_value="extrapolate")
    u_interp = interp_u(timestamps) # Shape (N, 6)
    
    # Computation Times
    interp_t = interp1d(rosbag_data["time_ts"], rosbag_data["comp_times"], bounds_error=False, fill_value="extrapolate")
    t_comp_interp = interp_t(timestamps)

    # 5. Calculate Metrics from Thesis
    metrics = {}
    duration = timestamps[-1] - timestamps[0]

    # Position Error Metrics
    metrics['e_p_max'] = np.max(e_p)
    metrics['e_p_rms'] = np.sqrt(np.trapz(e_p**2, timestamps) / duration)

    # Orientation Error Metrics
    metrics['e_o_max'] = np.max(e_o)
    metrics['e_o_rms'] = np.sqrt(np.trapz(e_o**2, timestamps) / duration)
    
    # Control Effort
    u_norm_sq = np.linalg.norm(u_interp, axis=1)**2
    metrics['E_c'] = np.trapz(u_norm_sq, timestamps)
    
    # Average Computation Time
    metrics['t_comp_avg'] = np.mean(rosbag_data["comp_times"])

    print("  Analysis complete.")
    return {
        "metrics": metrics,
        "timestamps": timestamps,
        "e_p": e_p,
        "e_p_xyz": e_p_xyz,
        "e_o": e_o,
        "u_interp": u_interp,
        "t_comp_interp": t_comp_interp,
        "positions": positions,
        "p_d": p_d
    }

# --- Plotting Functions ---

def plot_results(results, output_dir, base_filename):
    """Generates and saves all required plots for the analysis."""
    print("Generating plots...")
    if results is None:
        print("  Skipping plots: No analysis results available.")
        return

    ts = results["timestamps"]
    times_rel = ts - ts[0] # Relative time for plotting

    # Plot 1: Position Error
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(times_rel, results["e_p"] * 1000, label='Total Position Error')
    ax1.set_title(f'{base_filename} - End-Effector Position Error')
    ax1.set_ylabel('Position Error (mm)')
    ax1.grid(True)
    ax1.legend()
    ax2.plot(times_rel, results["e_p_xyz"][:, 0] * 1000, label='X Error')
    ax2.plot(times_rel, results["e_p_xyz"][:, 1] * 1000, label='Y Error')
    ax2.plot(times_rel, results["e_p_xyz"][:, 2] * 1000, label='Z Error')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Per-Axis Error (mm)')
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_pos_error.png"))
    plt.close()

    # Plot 2: Orientation Error
    plt.figure(figsize=(12, 6))
    plt.plot(times_rel, results["e_o"])
    plt.title(f'{base_filename} - End-Effector Orientation Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Orientation Error (Quaternion Distance)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_ori_error.png"))
    plt.close()

    # Plot 3: Control Inputs
    plt.figure(figsize=(12, 8))
    for i in range(results["u_interp"].shape[1]):
        plt.plot(times_rel, results["u_interp"][:, i], label=f'Joint {i+1} Vel')
    plt.title(f'{base_filename} - MPC Control Inputs (Joint Velocities)')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Velocity (rad/s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_control_inputs.png"))
    plt.close()

    # Plot 4: Computation Time
    plt.figure(figsize=(12, 6))
    plt.plot(times_rel, results["t_comp_interp"] * 1000)
    plt.title(f'{base_filename} - MPC Computation Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Computation Time (ms)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_comp_time.png"))
    plt.close()

    # Plot 5: End-Effector Trajectory (XY Plane)
    plt.figure(figsize=(8, 8))
    plt.plot(results["positions"][:, 0] * 1000, results["positions"][:, 1] * 1000, label='EE Trajectory')
    plt.scatter(results["p_d"][0] * 1000, results["p_d"][1] * 1000, c='r', marker='x', s=100, label='Target', zorder=5)
    plt.title(f'{base_filename} - End-Effector Trajectory (XY Plane)')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Y Position (mm)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_trajectory_xy.png"))
    plt.close()
    
    print(f"  All plots saved to directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate MPC hardware performance using MoCap and ROS bag data.")
    parser.add_argument("mocap_file", help="Path to the MoCap ground truth data file (.tsv).")
    parser.add_argument("bag_file", help="Path to the ROS bag file.")
    parser.add_argument("-o", "--output-dir", default="hardware_mpc_results", help="Directory to save plots and results.")
    
    # Rosbag topics
    parser.add_argument("--feedback-topic", default="/mpc_controller/feedback", help="Topic for MPC action feedback.")
    parser.add_argument("--time-topic", default="/mpc_controller/computation_time", help="Topic for MPC computation time.")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel", help="Topic for the velocity commands sent to the base, used for synchronization.")

    # MoCap column names
    parser.add_argument("--mocap-time-col", default="Time", help="Column name for timestamps in MoCap file.")
    parser.add_argument("--mocap-pos-x-col", default="end_effector X", help="Column name for MoCap End-Effector X position.")
    parser.add_argument("--mocap-pos-y-col", default="end_effector Y", help="Column name for MoCap End-Effector Y position.")
    parser.add_argument("--mocap-pos-z-col", default="end_effector Z", help="Column name for MoCap End-Effector Z position.")
    parser.add_argument("--mocap-roll-col", default="end_effector Roll", help="Column name for MoCap End-Effector Roll.")
    parser.add_argument("--mocap-pitch-col", default="end_effector Pitch", help="Column name for MoCap End-Effector Pitch.")
    parser.add_argument("--mocap-yaw-col", default="end_effector Yaw", help="Column name for MoCap End-Effector Yaw.")
    parser.add_argument("--header-row", type=int, default=11, help="0-indexed row number for headers in the MoCap TSV file (default: 11 for row 12).")
    parser.add_argument("--motion-start-thresh", type=float, default=0.005, help="Position change (m) threshold to detect MoCap motion start for sync (default: 0.005m / 5mm).")
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    plot_base_filename = f"{os.path.splitext(os.path.basename(args.mocap_file))[0]}"

    # --- Execute Analysis ---
    mocap_data = load_mocap_data(
        args.mocap_file, args.mocap_time_col, 
        args.mocap_pos_x_col, args.mocap_pos_y_col, args.mocap_pos_z_col,
        args.mocap_roll_col, args.mocap_pitch_col, args.mocap_yaw_col,
        args.header_row
    )
    
    rosbag_data = load_rosbag_data(args.bag_file, args.feedback_topic, args.time_topic, args.cmd_vel_topic)

    analysis_results = analyze_performance(mocap_data, rosbag_data, args.motion_start_thresh)
    
    if analysis_results:
        # --- Print Metrics ---
        metrics = analysis_results["metrics"]
        print("\n--- MPC Hardware Performance Metrics ---")
        print(f"  Max Position Error (e_p,max):   {metrics['e_p_max'] * 1000:.3f} mm")
        print(f"  RMS Position Error (e_p,rms):   {metrics['e_p_rms'] * 1000:.3f} mm")
        print(f"  Max Orientation Error (e_o,max): {metrics['e_o_max']:.6f} (quaternion distance)")
        print(f"  RMS Orientation Error (e_o,rms): {metrics['e_o_rms']:.6f} (quaternion distance)")
        print(f"  Control Effort (E_c):           {metrics['E_c']:.4f}")
        print(f"  Avg. Comp. Time (t_comp):      {metrics['t_comp_avg'] * 1000:.3f} ms")
        print("----------------------------------------\n")

        # --- Generate Plots ---
        plot_results(analysis_results, args.output_dir, plot_base_filename)

    print("MPC hardware evaluation finished.")

if __name__ == "__main__":
    main() 