# UR20 Mobile Platform MPC Simulation

This repository contains the ROS packages for simulating a UR20 robot arm mounted on a configurable mobile platform (linear prismatic or angular revolute) and controlling it using Model Predictive Control (MPC).

## Prerequisites

*   **ROS Distribution:** Noetic Ninjemys (or compatible)
*   **Catkin:** Standard ROS build system tools (`catkin_tools` or `catkin_make`)
*   **MoveIt:** The MoveIt motion planning framework.
    ```bash
    sudo apt update
    sudo apt install ros-noetic-moveit
    ```
*   **ROS Control & Controllers:**
    ```bash
    sudo apt install ros-noetic-ros-control ros-noetic-ros-controllers ros-noetic-joint-state-controller ros-noetic-position-controllers ros-noetic-velocity-controllers ros-noetic-joint-trajectory-controller
    ```
*   **Gazebo ROS:**
    ```bash
    sudo apt install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control
    ```
*   **UR ROS Driver/Description:** Ensure you have the Universal Robots ROS driver packages, specifically `ur_description` which is used in the Xacro files. If not installed:
    ```bash
    sudo apt install ros-noetic-universal-robot
    ```
*   **Python Dependencies:** (Check `requirements.txt` if available, otherwise install as needed)
    ```bash
    pip install numpy scipy
    ```
    *(You might need `python3-pip` installed: `sudo apt install python3-pip`)*

## Installation

1.  **Clone the Repository:** Place the `MPC` directory (containing `mpc_simulation` and `ur20_mpc_controller` packages) into your Catkin workspace's `src` directory (e.g., `~/catkin_ws/src/`).
    ```bash
    cd ~/catkin_ws/src
    # Assuming your MPC project folder is already here
    # git clone <your-repo-url> # Or copy the MPC folder here
    ```
2.  **Build the Workspace:** Navigate to the root of your Catkin workspace and build the packages.
    ```bash
    cd ~/catkin_ws
    catkin_make # or catkin build
    ```
3.  **Source the Workspace:** Remember to source your workspace setup file in every new terminal you use for this project.
    ```bash
    source ~/catkin_ws/devel/setup.bash
    ```

## Running the Simulation

The main launch file `mpc_bringup.launch` starts Gazebo, spawns the robot model (UR20 on a mobile base), and loads the necessary controllers.

1.  **Choose Motion Type:** Decide if you want the mobile base to move linearly or angularly.
2.  **Launch:**
    *   **Linear Motion (Default - Prismatic Joint on Y-axis):**
        ```bash
        roslaunch mpc_description mpc_bringup.launch
        ```
    *   **Angular Motion (Revolute Joint around Z-axis):**
        ```bash
        roslaunch mpc_description mpc_bringup.launch motion_type:=angular
        ```

This will open Gazebo with the robot loaded.

## Running the MPC Controller

The MPC controller runs as an action server.

1.  **Ensure Simulation is Running:** The Gazebo simulation (`mpc_bringup.launch`) must be running first.
2.  **Launch the MPC Node:**
    ```bash
    roslaunch ur20_mpc_controller mpc_controller.launch
    ```
    This node loads parameters from `controller_params.yaml` and waits for action goals.

## Running the Platform Motion Simulator (Test Script)

This script sends velocity commands to the platform's joint controller to simulate base movement, allowing you to test the MPC's compensation capabilities.

1.  **Ensure Simulation is Running:** The Gazebo simulation (`mpc_bringup.launch`) must be running with the *matching* `motion_type`.
2.  **Run the Script:**
    *   **If Simulation launched with `motion_type:=linear` (or default):**
        ```bash
        rosrun ur20_mpc_controller simulate_platform_motion.py _motion_type:=linear
        ```
        *(Note: `_motion_type:=linear` is technically the default in the script, but being explicit is good practice)*
    *   **If Simulation launched with `motion_type:=angular`:**
        ```bash
        rosrun ur20_mpc_controller simulate_platform_motion.py _motion_type:=angular
        ```

You should see the mobile base moving sinusoidally in Gazebo (either translating along the Y-axis or rotating around the Z-axis).

## Key Configuration Files

*   `MPC/mpc_simulation/mpc_description/urdf/mpc_robot.xacro`: Defines the robot's structure, including the conditional prismatic/revolute joint for the base.
*   `MPC/mpc_simulation/mpc_description/config/ur20_controllers.yaml`: Configures the `ros_control` controllers (arm trajectory controller, joint state controller, platform velocity controllers).
*   `MPC/ur20_mpc_controller/config/controller_params.yaml`: Configures the MPC algorithm parameters (horizon, dt, weights, etc.) used by the `mpc_action_server.py`.
*   `MPC/mpc_simulation/mpc_description/launch/mpc_bringup.launch`: Main simulation launch file, handles Gazebo startup, robot spawning, and conditional controller loading based on `motion_type`.
*   `MPC/ur20_mpc_controller/ur20_mpc_controller/launch/mpc_controller.launch`: Launches the MPC action server node.
