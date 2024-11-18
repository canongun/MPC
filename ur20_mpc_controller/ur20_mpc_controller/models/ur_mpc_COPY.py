#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.optimize import minimize
import moveit_commander
from typing import Dict, Tuple, List

from ur20_mpc_controller.models.base_estimator import BaseMotionEstimator

class URMPC:
    def __init__(self):
        # Initialize ROS node
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander()
        self.move_group = moveit_commander.MoveGroupCommander("arm")
        
        # Load parameters from config
        self.horizon = rospy.get_param('~mpc_controller/horizon', 15)
        self.dt = rospy.get_param('~mpc_controller/dt', 0.05)
        
        # Load weights
        self.w_ee_pos = rospy.get_param('~mpc_controller/weights/position', 10.0)
        self.w_ee_ori = rospy.get_param('~mpc_controller/weights/orientation', 2.0)
        self.w_control = rospy.get_param('~mpc_controller/weights/control', 0.05)
        
        # Load thresholds
        self.lin_threshold = rospy.get_param('~mpc_controller/thresholds/linear', 0.005)
        self.ang_threshold = rospy.get_param('~mpc_controller/thresholds/angular', 0.005)
        
        # Load active weights
        self.lin_weight_active = rospy.get_param('~mpc_controller/active_weights/linear', 15.0)
        self.lin_weight_idle = rospy.get_param('~mpc_controller/active_weights/linear_idle', 1.0)
        self.ang_weight_active = rospy.get_param('~mpc_controller/active_weights/angular', 5.0)
        self.ang_weight_idle = rospy.get_param('~mpc_controller/active_weights/angular_idle', 0.5)
        
        # Load bounds
        self.pos_bound = rospy.get_param('~mpc_controller/bounds/position', 0.005)
        self.ori_bound = rospy.get_param('~mpc_controller/bounds/orientation', 0.005)
        
        # Load joint scales
        self.joint_scales = np.array(rospy.get_param('~mpc_controller/joint_scales', 
            [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]))
        
        # Load robot configuration
        self.arm_base_offset = {
            'x': rospy.get_param('~mpc_controller/arm_base_offset/x', 0.06),
            'y': rospy.get_param('~mpc_controller/arm_base_offset/y', -0.1),
            'z': rospy.get_param('~mpc_controller/arm_base_offset/z', 0.09),
            'yaw': rospy.get_param('~mpc_controller/arm_base_offset/yaw', np.pi)
        }
        
        # State and control limits
        self.joint_pos_limits = np.array([  # [min, max] for each joint
            [-2*np.pi, 2*np.pi],  # Joint 1
            [-2*np.pi, 2*np.pi],  # Joint 2
            [-2*np.pi, 2*np.pi],  # Joint 3
            [-2*np.pi, 2*np.pi],  # Joint 4
            [-2*np.pi, 2*np.pi],  # Joint 5
            [-2*np.pi, 2*np.pi]   # Joint 6
        ])
        
        # Add joint velocity limits
        self.joint_vel_limits = {
            'shoulder_pan_joint': 2.094395102393195,
            'shoulder_lift_joint': 2.094395102393195,
            'elbow_joint': 2.617993877991494,
            'wrist_1_joint': 3.665191429188092,
            'wrist_2_joint': 3.665191429188092,
            'wrist_3_joint': 3.665191429188092
        }
        
        self.base_estimator = BaseMotionEstimator()
        
        # Add new state-related attributes
        self.state_dim = 18  # 6 (joint pos) + 6 (joint vel) + 3 (EE pos) + 3 (EE ori)
        self.input_dim = 6   # 6 joint velocities
        
    def compute_control(self, 
                current_joint_state: Dict,
                current_ee_pose: Dict,
                target_ee_pose: Dict,
                base_state: Dict) -> np.ndarray:
        """Compute optimal joint velocities using MPC with safety features"""
        try:
            # Get initial state
            x0 = self.get_state(current_joint_state, current_ee_pose)
            
            # Predict base motion over horizon
            base_trajectory = self.predict_base_trajectory(base_state)
            
            # Get compensation velocities as initial guess
            initial_velocities = self.compute_compensation_velocities(x0, base_state)
            
            # Create initial control sequence
            u_sequence = np.zeros((self.horizon, self.input_dim))
            for k in range(self.horizon):
                u_sequence[k] = initial_velocities
            
            # Setup optimization problem
            def objective(u_flat):
                u_seq = u_flat.reshape(self.horizon, self.input_dim)
                x_trajectory = self.predict_state_trajectory(x0, u_seq)
                return self._compute_predicted_cost(x_trajectory, u_seq, target_ee_pose, base_trajectory)
            
            # Flatten control sequence for optimizer
            u0_flat = u_sequence.flatten()
            
            # Solve optimization problem
            result = minimize(
                fun=objective,
                x0=u0_flat,
                method='SLSQP',
                bounds=self._get_bounds(u0_flat),
                constraints=self._get_constraints(x0, current_ee_pose, base_trajectory),
                options={
                    'ftol': 1e-4,
                    'maxiter': 50,
                    'disp': False
                }
            )
            
            if not result.success:
                rospy.logwarn(f"MPC optimization failed: {result.message}")
                return u_sequence[0]  # Return first timestep of initial guess
                
            # Extract optimal control sequence
            u_optimal = result.x.reshape(self.horizon, self.input_dim)
            
            # After optimization succeeds, add safety check
            safe_control = self._get_safe_control(u_optimal[0], x0, base_state)
            
            # Add safety status to debug info
            J = self._get_jacobian(x0[:6])
            manip = self._compute_manipulability(J)
            rospy.logdebug(f"Manipulability measure: {manip}")
            rospy.logdebug(f"Control scaling: {np.linalg.norm(safe_control)/np.linalg.norm(u_optimal[0])}")
            
            return safe_control
            
        except Exception as e:
            rospy.logerr(f"Control computation failed: {str(e)}")
            # Return zero velocities as fallback
            return np.zeros(6)

    def _cost_function(self, x, current_joint_state, current_ee_pose, target_ee_pose, base_state):
        # Keep the current timestep calculation exactly as is
        current_cost = self._compute_current_cost(
            x[6:12],  # Current joint velocities
            current_joint_state,
            base_state
        )
        
        # Add future cost terms if horizon > 1
        future_cost = 0.0
        if self.horizon > 1:
            try:
                # Only add future predictions if they exist
                for i in range(1, self.horizon):
                    future_velocities = x[6+i*12:12+i*12]
                    # Use a decreasing weight for future terms
                    weight = 0.8**i  # Exponential decay of importance
                    future_cost += weight * self._compute_current_cost(
                        future_velocities,
                        current_joint_state,  # Use current state as reference
                        base_state  # Use current base state (conservative)
                    )
            except Exception as e:
                rospy.logwarn(f"Future cost computation failed: {e}")
                # If future computation fails, fall back to current cost only
                return current_cost
                
        return current_cost + 0.5 * future_cost  # Weight future less than present

    def _compute_current_cost(self, joint_velocities, joint_state, base_state):
        """Existing cost computation - exactly as before"""
        # Get full Jacobian (including orientation)
        J = self._get_jacobian(joint_state['position'])
        J_pos = J[:3, :]
        J_ori = J[3:, :]
        
        # End-effector velocities (both linear and angular)
        ee_lin_vel = J_pos @ joint_velocities
        ee_ang_vel = J_ori @ joint_velocities
        
        # Base velocities (invert both linear and angular velocities)
        base_lin_vel = -base_state['linear_velocity'][:3]    
        base_ang_vel = -base_state['angular_velocity']       
        
        # Compensation errors
        pos_compensation_error = ee_lin_vel - base_lin_vel
        ori_compensation_error = ee_ang_vel - base_ang_vel
        
        # Detect active motion directions
        active_lin_dirs = np.abs(base_lin_vel) > 0.005
        active_ang_dirs = np.abs(base_ang_vel) > 0.005
        
        # Weight compensation errors
        lin_weights = np.where(active_lin_dirs, 15.0, 1.0)
        ang_weights = np.where(active_ang_dirs, 5.0, 0.5)
        
        # Compute costs
        position_cost = np.sum((lin_weights * pos_compensation_error)**2)
        orientation_cost = np.sum((ang_weights * ori_compensation_error)**2)
        damping_cost = np.sum(joint_velocities**2) * 0.1
        
        return position_cost + self.w_ee_ori * orientation_cost + damping_cost

        
    def _get_bounds(self, u_flat: np.ndarray) -> List[Tuple[float, float]]:
        """Get bounds for optimization variables"""
        bounds = []
        n_controls = len(u_flat)
        
        # Velocity bounds for all timesteps
        for _ in range(n_controls):
            bounds.append((-self.joint_vel_limits['shoulder_pan_joint'], 
                          self.joint_vel_limits['shoulder_pan_joint']))
        return bounds
    
    def _get_current_ee_pose(self, joint_positions: np.ndarray) -> Dict:
        """Get current end-effector pose using MoveIt"""
        # Set the joint positions
        self.move_group.set_joint_value_target(joint_positions)
        
        # Get the pose
        current_pose = self.move_group.get_current_pose(end_effector_link = "gripper_end_tool_link").pose
        
        return {
            'position': np.array([
                current_pose.position.x,
                current_pose.position.y,
                current_pose.position.z
            ]),
            'orientation': np.array([
                current_pose.orientation.x,
                current_pose.orientation.y,
                current_pose.orientation.z,
                current_pose.orientation.w
            ])
        }

    def _get_constraints(self, x0: np.ndarray, current_ee_pose: Dict, 
                    base_trajectory: List[Dict]) -> List[Dict]:
        """Get constraints for optimization problem"""
        constraints = []
        
        def state_constraint(u_flat):
            u_seq = u_flat.reshape(self.horizon, self.input_dim)
            x_trajectory = self.predict_state_trajectory(x0, u_seq)
            
            # Collect all constraint violations
            violations = []
            for k in range(self.horizon):
                # Joint position limits
                q = x_trajectory[k, :6]
                violations.extend(self.joint_pos_limits[:, 1] - q)  # Upper bounds
                violations.extend(q - self.joint_pos_limits[:, 0])  # Lower bounds
                
                # End-effector constraints can be added here
                
            return np.array(violations)
        
        constraints.append({
            'type': 'ineq',
            'fun': state_constraint
        })
        
        return constraints

    def _transform_base_motion(self, base_state: Dict) -> Dict:
        """Transform base motion from mobile base frame to arm base frame"""
        # Get platform velocities
        platform_vel = base_state['linear_velocity']
        platform_ang_vel = base_state['angular_velocity']
        
        # Get offset vector from rotation center to arm base
        offset = np.array([
            self.arm_base_offset['x'],
            self.arm_base_offset['y'],
            self.arm_base_offset['z']
        ])
        
        # For rotation around z-axis
        omega = platform_ang_vel[2]  # positive is counter-clockwise
        
        # Calculate tangential velocity for counter-rotation
        # When platform rotates clockwise (negative omega):
        # The end-effector should move in the opposite direction in world frame
        tangential_vel = -np.array([
            omega * offset[1],     # Changed sign: -ω * y
            -omega * offset[0],    # Changed sign: ω * x
            0.0
        ])
        
        # Platform linear velocity compensation
        platform_comp_vel = platform_vel
        
        # Total velocity combines both compensations
        total_vel = platform_comp_vel + tangential_vel
        
        # Debug information
        rospy.loginfo("=== Motion Transform Debug ===")
        rospy.loginfo(f"Platform angular velocity: {platform_ang_vel}")
        rospy.loginfo(f"Offset from rotation center: {offset}")
        rospy.loginfo(f"Calculated tangential velocity: {tangential_vel}")
        rospy.loginfo(f"Total compensation velocity: {total_vel}")
        rospy.loginfo("============================")
        
        transformed_state = base_state.copy()
        transformed_state['linear_velocity'] = total_vel
        transformed_state['position'] = base_state['position'] + offset
        transformed_state['angular_velocity'] = platform_ang_vel  # Keep for orientation compensation
        
        return transformed_state

    def _get_jacobian(self, joint_positions: np.ndarray) -> np.ndarray:
        """Get the Jacobian matrix for the current configuration"""
        try:
            # Ensure joint_positions is a list (not numpy array)
            if isinstance(joint_positions, np.ndarray):
                joint_positions = joint_positions.tolist()
            
            # Get Jacobian directly using joint positions
            J = self.move_group.get_jacobian_matrix(joint_positions)
            
            # Convert to numpy array and ensure correct shape (6x6)
            J = np.array(J)
            if J.shape != (6, 6):
                rospy.logwarn(f"Unexpected Jacobian shape: {J.shape}, using fallback")
                return np.eye(6)
                
            return J
            
        except Exception as e:
            rospy.logerr(f"Failed to get Jacobian: {str(e)}")
            return np.eye(6)

    def get_state(self, joint_state: Dict, ee_pose: Dict) -> np.ndarray:
        """
        Construct full state vector from joint states and EE pose
        """
        q = np.array(joint_state['position'])
        qdot = np.array(joint_state['velocity'])
        p_ee = ee_pose['position']
        R_ee = ee_pose['orientation'][:3]  # Using euler angles for now
        
        return np.concatenate([q, qdot, p_ee, R_ee])

    def system_dynamics(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Predict next state given current state and control input
        
        Args:
            x: State vector [q, qdot, p_ee, R_ee]
            u: Control input (joint velocities)
            dt: Time step
            
        Returns:
            Next state prediction
        """
        # Extract current state components
        q = x[:6]
        qdot = x[6:12]
        
        # Predict next joint positions
        q_next = q + u * dt
        qdot_next = u  # For velocity control
        
        # Get Jacobian at predicted configuration
        J = self._get_jacobian(q_next)
        
        # Predict EE motion
        ee_vel = J @ u
        p_ee_next = x[12:15] + ee_vel[:3] * dt
        R_ee_next = x[15:18] + ee_vel[3:] * dt  # Simplified orientation update
        
        return np.concatenate([q_next, qdot_next, p_ee_next, R_ee_next])

    def predict_state_trajectory(self, x0: np.ndarray, u_sequence: np.ndarray) -> np.ndarray:
        """
        Predict state trajectory over horizon given initial state and control sequence
        
        Args:
            x0: Initial state [state_dim]
            u_sequence: Control sequence over horizon [horizon x input_dim]
            
        Returns:
            Predicted state trajectory [horizon x state_dim]
        """
        x_trajectory = np.zeros((self.horizon, self.state_dim))
        x_trajectory[0] = x0
        
        for k in range(self.horizon - 1):
            x_trajectory[k + 1] = self.system_dynamics(
                x_trajectory[k], 
                u_sequence[k],
                self.dt
            )
        
        return x_trajectory

    def _compute_predicted_cost(self, x_trajectory: np.ndarray, u_sequence: np.ndarray, 
                          target_ee_pose: Dict, base_trajectory: List[Dict]) -> float:
        """
        Enhanced cost function with stability terms
        """
        total_cost = 0.0
        
        for k in range(self.horizon):
            # Previous cost terms
            ee_pos = x_trajectory[k, 12:15]
            ee_ori = x_trajectory[k, 15:18]
            
            pos_error = ee_pos - target_ee_pose['position']
            ori_error = ee_ori - target_ee_pose['orientation'][:3]
            
            pos_cost = self.w_ee_pos * np.sum(pos_error**2)
            ori_cost = self.w_ee_ori * np.sum(ori_error**2)
            control_cost = self.w_control * np.sum(u_sequence[k]**2)
            
            # Add manipulability cost
            J = self._get_jacobian(x_trajectory[k, :6])
            manip_measure = self._compute_manipulability(J)
            manip_cost = -0.1 * manip_measure  # Negative because we want to maximize manipulability
            
            # Add joint limit avoidance cost
            q = x_trajectory[k, :6]
            limit_cost = 0.0
            for i in range(6):
                # Quadratic cost that increases near joint limits
                normalized_pos = (2.0 * q[i] - (self.joint_pos_limits[i][0] + self.joint_pos_limits[i][1])) / \
                               (self.joint_pos_limits[i][1] - self.joint_pos_limits[i][0])
                limit_cost += 0.1 * normalized_pos**2
            
            # Add smoothness cost
            if k > 0:
                smoothness_cost = 0.1 * np.sum((u_sequence[k] - u_sequence[k-1])**2)
            else:
                smoothness_cost = 0.0
            
            # Combine all costs
            time_weight = 0.95**k
            total_cost += time_weight * (pos_cost + ori_cost + control_cost + 
                                       manip_cost + limit_cost + smoothness_cost)
        
        return total_cost

    def predict_base_trajectory(self, current_base_state: Dict) -> List[Dict]:
        """
        Predict base motion over the horizon using current state and motion model
        """
        base_trajectory = []
        
        # Current implementation assumes constant velocity model
        # But we can improve it with acceleration-based prediction
        current_vel = current_base_state['linear_velocity']
        current_ang_vel = current_base_state['angular_velocity']
        current_pos = current_base_state['position']
        current_ori = current_base_state['orientation']
        
        for k in range(self.horizon):
            # Predict position: p = p0 + v*t + 1/2*a*t^2
            # For now using just v*t since we don't have acceleration
            predicted_pos = current_pos + current_vel * self.dt * k
            
            # Predict orientation (simple euler integration)
            predicted_ori = current_ori + current_ang_vel * self.dt * k
            
            # Create predicted state
            predicted_state = {
                'position': predicted_pos,
                'orientation': predicted_ori,
                'linear_velocity': current_vel,  # Assuming constant velocity
                'angular_velocity': current_ang_vel
            }
            
            # Transform to arm base frame
            transformed_state = self._transform_base_motion(predicted_state)
            base_trajectory.append(transformed_state)
        
        return base_trajectory

    def compute_compensation_velocities(self, 
                                 current_state: np.ndarray,
                                 base_state: Dict) -> np.ndarray:
        """
        Compute required joint velocities to compensate for base motion
        """
        # Get current joint positions from state
        q = current_state[:6]
        
        # Get Jacobian at current configuration
        J = self._get_jacobian(q)
        
        # Transform base motion to arm base frame
        transformed_base = self._transform_base_motion(base_state)
        
        # Combine linear and angular velocities to compensate
        base_vel = np.concatenate([
            -transformed_base['linear_velocity'],
            -transformed_base['angular_velocity']
        ])
        
        # Compute required joint velocities
        try:
            # Use damped least squares for better numerical stability
            lambda_ = 0.1  # Damping factor
            J_dag = J.T @ np.linalg.inv(J @ J.T + lambda_ * np.eye(6))
            joint_velocities = J_dag @ base_vel
            
            # Scale joint velocities based on joint preferences
            joint_velocities *= self.joint_scales
            
            return joint_velocities
        except np.linalg.LinAlgError as e:
            rospy.logwarn(f"Failed to compute compensation velocities: {e}")
            return np.zeros(6)

    def _check_velocity_limits(self, velocities: np.ndarray) -> np.ndarray:
        """
        Safely scale velocities if they exceed limits
        """
        max_ratio = 1.0
        for i, vel in enumerate(velocities):
            ratio = abs(vel) / self.joint_vel_limits['shoulder_pan_joint']
            if ratio > max_ratio:
                max_ratio = ratio
        
        if max_ratio > 1.0:
            return velocities / max_ratio
        return velocities

    def _check_singularity(self, J: np.ndarray) -> bool:
        """
        Check if current configuration is near singularity
        """
        try:
            # Compute condition number of Jacobian
            s = np.linalg.svd(J, compute_uv=False)
            condition_number = s[0] / s[-1]
            return condition_number > 1e6
        except:
            return True

    def _compute_manipulability(self, J: np.ndarray) -> float:
        """
        Compute manipulability measure
        """
        return np.sqrt(np.linalg.det(J @ J.T))

    def _get_safe_control(self, u: np.ndarray, x: np.ndarray, base_state: Dict) -> np.ndarray:
        """
        Apply safety checks and modifications to control input
        """
        # Get current Jacobian
        J = self._get_jacobian(x[:6])
        
        # Check singularity
        if self._check_singularity(J):
            rospy.logwarn("Near singularity - using damped least squares")
            # Increase damping factor
            self.lambda_ = 1.0
        else:
            # Normal operation
            self.lambda_ = 0.1
        
        # Check joint limits and adjust if necessary
        q_next = x[:6] + u * self.dt
        for i in range(6):
            if q_next[i] > self.joint_pos_limits[i][1]:
                u[i] = (self.joint_pos_limits[i][1] - x[i]) / self.dt
            elif q_next[i] < self.joint_pos_limits[i][0]:
                u[i] = (self.joint_pos_limits[i][0] - x[i]) / self.dt
        
        # Check velocity limits
        u = self._check_velocity_limits(u)
        
        return u

def main():
    """Test the MPC controller"""
    rospy.init_node('test_ur_mpc')
    
    # Initialize MPC
    controller = URMPC()
    
    # Test with dummy data
    current_joint_state = {
        'position': np.zeros(6),
        'velocity': np.zeros(6)
    }
    
    current_ee_pose = {
        'position': np.array([0.5, 0, 0.5]),
        'orientation': np.zeros(3)
    }
    
    target_ee_pose = {
        'position': np.array([0.6, 0, 0.5]),
        'orientation': np.zeros(3)
    }
    
    base_state = {
        'position': np.zeros(3),
        'orientation': np.zeros(3),
        'linear_velocity': np.zeros(3),
        'angular_velocity': np.zeros(3)
    }
    
    # Compute control
    control = controller.compute_control(
        current_joint_state,
        current_ee_pose,
        target_ee_pose,
        base_state
    )
    
    rospy.loginfo(f"Computed control: {control}")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass