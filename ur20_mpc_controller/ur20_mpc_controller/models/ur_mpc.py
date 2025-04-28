#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, List

from ur20_mpc_controller.models.ur20_dynamics import UR20Dynamics

class OptimizationError(Exception):
    """Raised when MPC optimization fails"""
    pass

class URMPC:
    def __init__(self):
        """Initialize MPC controller for UR20 mobile manipulation
        
        State vector structure:
        - joint_positions (6): Current joint angles
        - joint_velocities (6): Current joint velocities  
        - ee_position (3): End-effector position in world frame
        - ee_orientation (3): End-effector orientation in euler angles
        """
        
        # Initialize dynamics model
        self.dynamics: UR20Dynamics = UR20Dynamics()
        
        # Load parameters from config
        self.horizon: int = rospy.get_param('~mpc_controller/horizon', 10)  # Match config default
        self.dt: float = rospy.get_param('~mpc_controller/dt', 0.1)
        
        # Add parameter validation
        if self.horizon <= 0:
            raise ValueError("Horizon must be positive")
        if self.dt <= 0:
            raise ValueError("Time step must be positive")
        
        # Load all weights from params
        self.w_position = rospy.get_param('~mpc_controller/weights/position', 3.0)
        self.w_orientation = rospy.get_param('~mpc_controller/weights/orientation', 2.0)
        self.w_control = rospy.get_param('~mpc_controller/weights/control', 0.005)
        self.w_smooth = rospy.get_param('~mpc_controller/weights/smooth', 0.02)
        
        # Load active motion weights
        self.w_linear = rospy.get_param('~mpc_controller/active_weights/linear', 50.0)
        self.w_angular = rospy.get_param('~mpc_controller/active_weights/angular', 5.0)
        
        # Load robot configuration
        self.arm_base_offset = {
            'x': rospy.get_param('~mpc_controller/arm_base_offset/x', 0.0),
            'y': rospy.get_param('~mpc_controller/arm_base_offset/y', 0.1),
            'z': rospy.get_param('~mpc_controller/arm_base_offset/z', 0.5),
            'yaw': rospy.get_param('~mpc_controller/arm_base_offset/yaw', 0.0)
        }
        
        self.previous_solution = None
        
        # Add damping weight
        self.w_damping = rospy.get_param('~mpc_controller/weights/damping', 0.5)
        
        # ----------  Mobile‑base Dynamic Model Parameters ----------
        dyn_ns = '~mpc_controller/base_dynamics'
        self.use_dynamic_model = rospy.get_param(f'{dyn_ns}/use_dynamic_model', False)

        # Physical parameters (defaults roughly for Innok Heros)
        self.base_mass        = rospy.get_param(f'{dyn_ns}/mass', 180.0)       # kg
        self.base_Iz          = rospy.get_param(f'{dyn_ns}/I_z', 62.0)         # kg·m²
        self.wheel_radius     = rospy.get_param(f'{dyn_ns}/wheel_radius', 0.165)
        self.half_wheelbase   = rospy.get_param(f'{dyn_ns}/half_wheelbase', 0.35)

        # Linear / angular damping
        self.k_v              = rospy.get_param(f'{dyn_ns}/k_v', 25.0)
        self.k_omega          = rospy.get_param(f'{dyn_ns}/k_omega', 8.0)

        # First‑order actuator lags (if wheel commands are known)
        self.tau_linear       = rospy.get_param(f'{dyn_ns}/tau_linear', 0.45)
        self.tau_angular      = rospy.get_param(f'{dyn_ns}/tau_angular', 0.30)

        # Optional caster wheel
        self.use_caster       = rospy.get_param(f'{dyn_ns}/use_caster', False)
        self.tau_caster       = rospy.get_param(f'{dyn_ns}/tau_caster', 0.25)

    def compute_control(self, 
                current_joint_state: Dict,
                current_ee_pose: Dict,
                target_ee_pose: Dict,
                base_state: Dict) -> np.ndarray:
        """Compute optimal control action using MPC
        
        Args:
            current_joint_state: Current joint positions and velocities
            current_ee_pose: Current end-effector pose
            target_ee_pose: Target end-effector pose
            base_state: Current base state
            
        Returns:
            Optimal joint velocity commands
            
        Raises:
            OptimizationError: If optimization fails to converge
            ValueError: If inputs are invalid
        """
        try:
            # Check both linear and angular velocity
            base_vel_mag = np.linalg.norm(base_state['linear_velocity'])
            base_ang_mag = np.linalg.norm(base_state['angular_velocity'])
            
            if base_vel_mag < 0.01 and base_ang_mag < 0.01:
                return np.zeros(self.dynamics.n_controls)

            # 1. Get relevant states in world frame
            platform_pos_w = base_state['position']
            platform_lin_vel_w = base_state['linear_velocity']
            platform_ang_vel_w = base_state['angular_velocity']
            ee_pos_w = current_ee_pose['position']

            # 2. Calculate vector from platform center to EE (world frame)
            r_p_to_ee_w = ee_pos_w - platform_pos_w

            # 3. Calculate velocity induced at EE by platform rotation (world frame)
            # Use np.cross for general 3D rotation
            omega_p_w = base_state['angular_velocity']
            r_p_to_ee_w = current_ee_pose['position'] - base_state['position']
            # If only Z rotation is significant:
            # v_ee_rot_w = np.array([
            #     -omega_p_w[2] * r_p_to_ee_w[1], 
            #      omega_p_w[2] * r_p_to_ee_w[0], 
            #      0.0 
            # ])
            # For full 3D rotation:
            v_ee_rot_w = np.cross(omega_p_w, r_p_to_ee_w)

            # 4. Calculate total velocity induced at EE (world frame)
            v_induced_ee_w = platform_lin_vel_w + v_ee_rot_w

            # 5. Calculate target compensation velocities (world frame)
            target_lin_vel_w = -v_induced_ee_w
            target_ang_vel_w = -platform_ang_vel_w 

            # Convert target velocities from WORLD → ARM-BASE frame
            yaw = base_state['orientation'][2]
            target_lin_vel_b = self._world_to_base(target_lin_vel_w, yaw)
            target_ang_vel_b = self._world_to_base(target_ang_vel_w, yaw)
            
            rospy.logdebug(f"Target Compensation Velocity (World): Linear={target_lin_vel_w}, Angular={target_ang_vel_w}")

            # Get current state vector
            current_state = self.dynamics.get_state_vector(
                current_joint_state, 
                current_ee_pose
            )
            
            # Predict base motion over horizon
            base_trajectory = self.predict_base_motion(base_state, self.horizon)
            
            # Get initial guess using Jacobian
            J = self.dynamics.get_jacobian(current_joint_state['position'])
            J_pos = J[:3, :]
            J_ori = J[3:, :]
            
            # Debug the angular velocity handling
            # rospy.loginfo(f"Raw angular velocity: {base_state['angular_velocity']}")
            # rospy.loginfo(f"Transformed angular velocity: {transformed_base_state['angular_velocity']}")
     
            # Combine into full velocity vector (all in arm-base frame)
            J_full = np.vstack([J_pos, J_ori])
            base_vel_full = np.concatenate([target_lin_vel_b, target_ang_vel_b])
            
            # Compute initial velocities using damped least squares
            lambda_ = 0.01
            J_pinv = J_full.T @ np.linalg.inv(J_full @ J_full.T + lambda_ * np.eye(6))
            initial_vel = J_pinv @ base_vel_full
            
            # Initialize control sequence with warm starting
            if self.previous_solution is not None:
                # Shift previous solution
                x0 = np.roll(self.previous_solution, -self.dynamics.n_controls)
                x0[-self.dynamics.n_controls:] = initial_vel
            else:
                x0 = np.zeros(self.dynamics.n_controls * self.horizon)
                for i in range(self.horizon):
                    x0[i*self.dynamics.n_controls:(i+1)*self.dynamics.n_controls] = initial_vel
            
            # Improve optimization parameters
            result = minimize(
                fun=lambda x: self._cost_function(x, current_state, target_ee_pose, base_trajectory),
                x0=x0,
                method='SLSQP',
                bounds=self._get_bounds(x0),
                constraints=self._get_constraints(current_state, base_trajectory),
                options={
                    'ftol': 1e-3,
                    'maxiter': 50,  # Increased iterations
                    'eps': 1e-3,
                    'disp': True  # Show optimization progress
                }
            )
            
            if not result.success:
                raise OptimizationError(f"Optimization failed: {result.message}")
            
            # Get first control action (receding horizon principle)
            control = result.x[:self.dynamics.n_controls]
            
            # Add control smoothing
            if hasattr(self, 'previous_control'):
                # Use stronger filtering (0.5 means 50% new, 50% old)
                alpha_output = 0.5
                control = alpha_output * control + (1.0 - alpha_output) * self.previous_control
            
            # Store for next iteration
            self.previous_control = control.copy()
            
            # Store solution for warm start in next iteration
            self.previous_solution = result.x
            
            # # Debug output
            # predicted_trajectory = self.predict_trajectory(current_state, 
            #                                              result.x.reshape(self.horizon, -1), 
            #                                              self.horizon)
            # ee_vel = J_pos @ control
            # ee_ang_vel = J_ori @ control
            
            # rospy.loginfo(f"Base linear velocity: {base_state['linear_velocity']}")
            # rospy.loginfo(f"Base angular velocity: {base_state['angular_velocity']}")
            # rospy.loginfo(f"EE linear velocity: {ee_vel}")
            # rospy.loginfo(f"EE angular velocity: {ee_ang_vel}")
            # rospy.loginfo(f"Predicted final EE error: {predicted_trajectory[-1][2*self.dynamics.n_q:2*self.dynamics.n_q + 3] - target_ee_pose['position']}")
            
            # Add velocity sanity check to prevent extreme commands
            max_joint_vel = 0.5  # rad/s
            control = np.clip(control, -max_joint_vel, max_joint_vel)
            
            return control
            
        except OptimizationError as e:
            rospy.logerr(f"Optimization error: {str(e)}")
            return initial_vel
        except ValueError as e:
            rospy.logerr(f"Invalid input: {str(e)}")
            return np.zeros(self.dynamics.n_controls)
        except Exception as e:
            rospy.logerr(f"Unexpected error: {str(e)}")
            return np.zeros(self.dynamics.n_controls)

    def calculate_stage_cost(self, state, target_ee_pose, base_state, control, k):
        """Calculate cost for a single stage in the prediction horizon"""
        # Get current end-effector state components from predicted state 'state'
        ee_pos = state[2*self.dynamics.n_q:2*self.dynamics.n_q + 3]
        ee_ori = state[2*self.dynamics.n_q + 3:2*self.dynamics.n_q + 6]
        
        # Get Jacobian for velocity computation using predicted joint positions 'q'
        q = state[:self.dynamics.n_q]
        J = self.dynamics.get_jacobian(q)
        J_pos = J[:3, :]
        J_ori = J[3:, :]
        
        # Calculate end-effector velocities generated by the arm control 'u_k'
        ee_lin_vel_arm = J_pos @ control # Velocity generated by arm joints
        ee_ang_vel_arm = J_ori @ control # Angular velocity generated by arm joints
        
        # --- Calculate the full velocity induced by the base motion at step k ---
        # Get predicted base state at step k
        platform_pos_w_k = base_state['position']
        platform_lin_vel_w_k = base_state['linear_velocity'][:3]
        platform_ang_vel_w_k = base_state['angular_velocity']

        # Calculate vector from predicted platform center to predicted EE (world frame)
        r_p_to_ee_w_k = ee_pos - platform_pos_w_k

        # Calculate linear velocity induced at EE by platform rotation (world frame)
        v_ee_rot_w_k = np.cross(platform_ang_vel_w_k, r_p_to_ee_w_k)

        # Calculate total *target* velocity the arm should generate to compensate
        target_comp_lin_vel_k = -(platform_lin_vel_w_k + v_ee_rot_w_k)
        target_comp_ang_vel_k = -platform_ang_vel_w_k
        # --- End of induced velocity calculation ---

        # Rotate target compensation velocities into arm base frame
        yaw_k = base_state['orientation'][2]
        target_comp_lin_vel_k_b = self._world_to_base(target_comp_lin_vel_k, yaw_k)
        target_comp_ang_vel_k_b = self._world_to_base(target_comp_ang_vel_k, yaw_k)

        # Velocity tracking errors: arm-generated (base frame) vs. target (base frame)
        lin_vel_error = ee_lin_vel_arm - target_comp_lin_vel_k_b
        ang_vel_error = ee_ang_vel_arm - target_comp_ang_vel_k_b
        
        # Position and orientation errors (comparing predicted EE pose to fixed target)
        pos_error = ee_pos - target_ee_pose['position']
        ori_error = ee_ori - target_ee_pose['orientation']
        
        # Compute costs with time-varying weights
        decay = 0.95**k  # Weight decay over horizon
        
        # Cost for linear velocity compensation error
        lin_vel_cost = self.w_linear * decay * np.sum(lin_vel_error**2)

        # Cost for angular velocity compensation error
        ang_vel_cost = self.w_angular * decay * np.sum(ang_vel_error**2)
        
        # Secondary objectives (position and orientation holding)
        pos_cost = self.w_position * decay * np.sum(pos_error**2)
        ori_cost = self.w_orientation * decay * np.sum(ori_error**2)
        
        # Control regularization 
        control_cost = self.w_control * np.sum(control**2)
        
        # Add damping cost to penalize high velocity changes
        # Note: Accessing self.previous_control here might be problematic if horizon > 1
        # Consider passing previous control or calculating smoothness differently if needed.
        # For now, assuming horizon=1 or this is acceptable.
        if k > 0 and hasattr(self, 'previous_control'):
            damping_cost = self.w_damping * np.sum((control - self.previous_control)**2)
        else:
            damping_cost = 0.0
        
        # Add smoothness cost for sequential control inputs if horizon > 1
        # This penalizes the magnitude of control^2 at step k>0, effectively a control cost again.
        # A true smoothness cost compares control[k] with control[k-1].
        # Let's assume the current implementation is intentional or horizon=1.
        smooth_cost = 0.0
        if k > 0 and self.horizon > 1:
             # Alternative smoothness: cost = self.w_smooth * np.sum((control - control_sequence[k-1])**2)
             # Requires access to control_sequence within stage cost. Pass it if needed.
            smooth_cost = self.w_smooth * np.sum(control**2)
        
        return lin_vel_cost + ang_vel_cost + pos_cost + ori_cost + control_cost + damping_cost + smooth_cost

    def calculate_terminal_cost(self, final_state, target_ee_pose, final_base_state):
        """Calculate terminal cost for both position and orientation"""
        # Get final end-effector state components
        ee_pos_final = final_state[2*self.dynamics.n_q:2*self.dynamics.n_q + 3]
        ee_ori_final = final_state[2*self.dynamics.n_q + 3:2*self.dynamics.n_q + 6]
        
        # Get final joint state and Jacobian
        q_final = final_state[:self.dynamics.n_q]
        dq_final = final_state[self.dynamics.n_q:self.dynamics.n_q + self.dynamics.n_dq] # Final joint velocities
        J_final = self.dynamics.get_jacobian(q_final)
        J_pos_final = J_final[:3, :]
        J_ori_final = J_final[3:, :]
        
        # Calculate final end-effector velocities generated by the arm
        ee_lin_vel_arm_final = J_pos_final @ dq_final
        ee_ang_vel_arm_final = J_ori_final @ dq_final
        
        # --- Calculate the full velocity induced by the final predicted base motion ---
        platform_pos_w_final = final_base_state['position']
        platform_lin_vel_w_final = final_base_state['linear_velocity'][:3]
        platform_ang_vel_w_final = final_base_state['angular_velocity']

        r_p_to_ee_w_final = ee_pos_final - platform_pos_w_final
        v_ee_rot_w_final = np.cross(platform_ang_vel_w_final, r_p_to_ee_w_final)

        target_comp_lin_vel_final = -(platform_lin_vel_w_final + v_ee_rot_w_final)
        target_comp_ang_vel_final = -platform_ang_vel_w_final
        # --- End of induced velocity calculation ---

        # Rotate target compensation velocities into arm base frame
        yaw_final = final_base_state['orientation'][2]
        target_comp_lin_vel_final_b = self._world_to_base(target_comp_lin_vel_final, yaw_final)
        target_comp_ang_vel_final_b = self._world_to_base(target_comp_ang_vel_final, yaw_final)

        # Terminal velocity errors (all in base frame)
        lin_vel_error_final = ee_lin_vel_arm_final - target_comp_lin_vel_final_b
        ang_vel_error_final = ee_ang_vel_arm_final - target_comp_ang_vel_final_b
        
        # Terminal Position and orientation errors
        pos_error_final = ee_pos_final - target_ee_pose['position']
        ori_error_final = ee_ori_final - target_ee_pose['orientation']
        
        # Apply higher weights for terminal cost (encourages convergence at horizon end)
        terminal_weight_factor = 2.0 
        
        # Combine terminal costs
        terminal_cost = (self.w_linear * terminal_weight_factor * np.sum(lin_vel_error_final**2) + 
                         self.w_angular * terminal_weight_factor * np.sum(ang_vel_error_final**2) + 
                         self.w_position * np.sum(pos_error_final**2) +          # Position/Orientation weights might not need factor
                         self.w_orientation * np.sum(ori_error_final**2))
                         
        return terminal_cost

    def _get_constraints(self, current_state: np.ndarray, base_trajectory: List[Dict]) -> List[Dict]:
        """Get optimization constraints over horizon"""
        constraints = []
        
        # Dynamic feasibility constraint
        def dynamics_constraint(x):
            control_sequence = x.reshape(self.horizon, self.dynamics.n_controls)
            trajectory = self.predict_trajectory(current_state, control_sequence, self.horizon)
            
            # Check joint limits and velocity limits
            violations = []
            for state in trajectory:
                q = state[:self.dynamics.n_q]
                dq = state[self.dynamics.n_q:self.dynamics.n_q + self.dynamics.n_dq]
                
                # Joint position limits
                pos_violations = np.minimum(
                    q - self.dynamics.joint_pos_limits[:, 0],  # Lower bounds
                    self.dynamics.joint_pos_limits[:, 1] - q   # Upper bounds
                )
                
                # Joint velocity limits
                vel_limits = np.array(list(self.dynamics.joint_vel_limits.values()))
                vel_violations = vel_limits - np.abs(dq)
                
                violations.extend(pos_violations)
                violations.extend(vel_violations)
            
            return np.array(violations)
        
        constraints.append({
            'type': 'ineq',
            'fun': dynamics_constraint
        })
        
        return constraints

    def _get_bounds(self, x0: np.ndarray) -> List[Tuple[float, float]]:
        """Get bounds for optimization variables"""
        bounds = []
        
        # Apply velocity bounds for each timestep
        vel_limits = list(self.dynamics.joint_vel_limits.values())
        for _ in range(self.horizon):
            for j in range(self.dynamics.n_controls):
                bounds.append((-vel_limits[j], vel_limits[j]))
                
        return bounds

    def _transform_base_motion(self, base_state: Dict, current_ee_pose: Dict = None) -> Dict:
        """Transform base motion from mobile base frame to arm base frame"""
        # Validate inputs
        for key in ['linear_velocity', 'angular_velocity', 'position', 'orientation']:
            if key not in base_state:
                raise ValueError(f"Missing required key: {key}")
            if not all(np.isfinite(base_state[key])):
                raise ValueError(f"Invalid values in {key}")
        
        # Get platform velocities
        platform_vel = base_state['linear_velocity']
        platform_ang_vel = base_state['angular_velocity']
        
        # Get arm base offset vector from rotation center
        arm_offset = np.array([
            self.arm_base_offset['x'],  # x - From TF tree
            self.arm_base_offset['y'],  # y - From TF tree
            self.arm_base_offset['z']   # z - From TF tree
        ])
        
        # Get the rotation between platform and arm base
        yaw = self.arm_base_offset['yaw']
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Transform platform velocity to arm base frame
        platform_vel_arm_frame = rotation_matrix @ platform_vel
        platform_ang_vel_arm_frame = rotation_matrix @ platform_ang_vel
        
        # For rotation around z-axis
        omega = platform_ang_vel_arm_frame[2]  # positive is counter-clockwise
        
        # Calculate tangential velocity at arm base
        arm_tangential_vel = np.array([
            -omega * arm_offset[1],     # -ω * y
            omega * arm_offset[0],      # ω * x
            0.0
        ])
        
        # Get platform's current orientation in world frame
        platform_yaw = base_state['orientation'][2]  # Extract yaw
        
        # Create rotation matrix for platform's current orientation
        world_rotation = np.array([
            [np.cos(platform_yaw), -np.sin(platform_yaw), 0],
            [np.sin(platform_yaw), np.cos(platform_yaw), 0],
            [0, 0, 1]
        ])
        
        # Apply world rotation to the arm offset vector
        rotated_arm_offset = world_rotation @ arm_offset
        
        # Calculate end-effector tangential velocity if pose is provided
        if current_ee_pose is not None:
            # Get vector from rotation center to end-effector
            ee_pos_world = current_ee_pose['position']
            platform_pos = base_state['position']
            
            # Calculate the vector from platform center to end-effector
            platform_to_ee = ee_pos_world - platform_pos
            
            # Calculate tangential velocity at end-effector position
            ee_tangential_vel = np.array([
                omega * platform_to_ee[1],   # ω * y
                -omega * platform_to_ee[0],  # -ω * x
                0.0
            ])
            
            # Total velocity is linear + end-effector tangential
            total_vel = platform_vel_arm_frame + ee_tangential_vel
            
            transformed_state = base_state.copy()
            transformed_state['linear_velocity'] = total_vel
        else:
            # Fall back to arm base calculation if EE pose not provided
            arm_tangential_vel = np.array([
                -omega * arm_offset[1],     # -ω * y
                omega * arm_offset[0],      # ω * x
                0.0
            ])
            
            transformed_state = base_state.copy()
            transformed_state['linear_velocity'] = platform_vel_arm_frame + arm_tangential_vel
        
        # These settings are common for both cases
        transformed_state['position'] = base_state['position'] + rotated_arm_offset
        transformed_state['angular_velocity'] = platform_ang_vel_arm_frame
        
        return transformed_state

    # ------------------------------------------------------------------
    # Kinematic Extrapolation Model
    def _predict_base_kinematic(self, current_base_state: Dict, horizon: int):
        base_trajectory = []
        dt = self.dt
        
        # Estimate acceleration with filtering
        if hasattr(self, 'prev_base_vel'):
            raw_accel = (current_base_state['linear_velocity'] - self.prev_base_vel) / dt
            
            # Apply stronger low-pass filtering on acceleration
            if hasattr(self, 'prev_accel'):
                # More aggressive filtering (0.5 means 50% new, 50% previous)
                alpha = 0.5  # Reduced from 0.7 for stronger filtering
                base_accel = alpha * raw_accel + (1 - alpha) * self.prev_accel
                
                # Add acceleration limiting
                accel_limit = 0.5  # m/s²
                base_accel = np.clip(base_accel, -accel_limit, accel_limit)
            else:
                # First-time initialization
                base_accel = raw_accel * 0.5  # Conservative initial estimate
            
            self.prev_accel = base_accel.copy()
        else:
            base_accel = np.zeros(3)
        
        self.prev_base_vel = current_base_state['linear_velocity'].copy()
        
        # Similarly filter angular acceleration
        if hasattr(self, 'prev_base_ang_vel'):
            raw_ang_accel = (current_base_state['angular_velocity'] - self.prev_base_ang_vel) / dt
            
            if hasattr(self, 'prev_ang_accel'):
                alpha_ang = 0.5
                ang_accel = alpha_ang * raw_ang_accel + (1 - alpha_ang) * self.prev_ang_accel
                
                # Add angular acceleration limiting
                ang_accel_limit = 0.3  # rad/s²
                ang_accel = np.clip(ang_accel, -ang_accel_limit, ang_accel_limit)
            else:
                ang_accel = raw_ang_accel * 0.5
                
            self.prev_ang_accel = ang_accel.copy()
        else:
            ang_accel = np.zeros(3)
            
        self.prev_base_ang_vel = current_base_state['angular_velocity'].copy()
        
        for k in range(horizon):
            # Predict velocity using acceleration with decay factor
            decay = min(1.0, max(0.3, 1.0 - 0.2*k))  # Decay acceleration impact over time
            predicted_vel = current_base_state['linear_velocity'] + k * dt * base_accel * decay
            
            # Predict angular velocity with decay factor for stability
            predicted_ang_vel = current_base_state['angular_velocity'] + k * dt * ang_accel * decay
            
            # Predict position using updated velocity
            predicted_pos = (current_base_state['position'] + 
                            k * dt * current_base_state['linear_velocity'] +
                            0.5 * k * dt**2 * base_accel * decay)
            
            # Predict orientation
            predicted_ori = (current_base_state['orientation'] + 
                            k * dt * current_base_state['angular_velocity'] +
                            0.5 * k * dt**2 * ang_accel * decay)
            
            predicted_state = {
                'position': predicted_pos.copy(),
                'orientation': predicted_ori.copy(),
                'linear_velocity': predicted_vel.copy(),
                'angular_velocity': predicted_ang_vel.copy()
            }
            
            base_trajectory.append(predicted_state)
        
        return base_trajectory
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Complex Dynamic Model
    def _predict_base_dynamic(self, current_base_state: Dict, horizon: int) -> List[Dict]:
        """
        Forward‑integrate a planar rigid‑body model of a differential‑drive base.
        Assumes body‑frame forward speed v and yaw rate ω.
        Uses first‑order actuator lag to move current v, ω towards the last commanded
        values if they are included in base_state, falls back to passive damping 
        otherwise.
        """
        dt = self.dt
        traj = []

        # Unpack current state (world frame)
        pos   = current_base_state['position'].copy()        # [x, y, z]
        yaw   = current_base_state['orientation'][2]         # assume planar
        v_bx  = current_base_state['linear_velocity'][0]     # body‑x speed
        ω     = current_base_state['angular_velocity'][2]    # yaw rate

        # Optional commanded velocities (needed for act. lag model)
        v_cmd = current_base_state.get('v_cmd', v_bx)
        ω_cmd = current_base_state.get('omega_cmd', ω)

        for k in range(horizon):
            # --- longitudinal dynamics ---------------------------------
            F_t  = (self.base_mass / self.tau_linear) * (v_cmd - v_bx)
            a    = (F_t - self.k_v * v_bx) / self.base_mass
            v_bx = v_bx + a * dt

            # --- yaw dynamics ------------------------------------------
            τ_z  = (self.base_Iz / self.tau_angular) * (ω_cmd - ω)
            α    = (τ_z - self.k_omega * ω) / self.base_Iz
            ω    = ω + α * dt

            # --- kinematics (world frame) ------------------------------
            yaw  = yaw + ω * dt
            cosψ, sinψ = np.cos(yaw), np.sin(yaw)
            vx_w = v_bx * cosψ
            vy_w = v_bx * sinψ
            pos  = pos + np.array([vx_w, vy_w, 0.0]) * dt

            traj.append({
                'position':         pos.copy(),
                'orientation':      np.array([0.0, 0.0, yaw]),
                'linear_velocity':  np.array([vx_w, vy_w, 0.0]),
                'angular_velocity': np.array([0.0, 0.0, ω])
            })
        return traj
    # ------------------------------------------------------------------

    # Dispatcher (called by compute_control)
    def predict_base_motion(self, current_base_state: Dict, horizon: int) -> List[Dict]:
        """
        Wrapper that chooses between the old kinematic extrapolation and the new
        dynamic‑model integration based on the `use_dynamic_model` flag.
        """
        if self.use_dynamic_model:
            return self._predict_base_dynamic(current_base_state, horizon)
        else:
            return self._predict_base_kinematic(current_base_state, horizon)

    def _cost_function(self, x: np.ndarray, current_state: np.ndarray, 
                      target_ee_pose: Dict, base_trajectory: List[Dict]) -> float:
        """True MPC cost function evaluating the entire trajectory"""
        # Reshape control sequence
        control_sequence = x.reshape(self.horizon, self.dynamics.n_controls)
        
        # Initialize cost and state
        total_cost = 0.0
        x_current = current_state.copy()
        
        # Evaluate cost over the entire horizon
        for k in range(self.horizon):
            # Get control input and predicted base state for current stage
            u_k = control_sequence[k]
            base_k = base_trajectory[k]
            
            # Add stage cost
            stage_cost = self.calculate_stage_cost(x_current, target_ee_pose, base_k, u_k, k)
            total_cost += stage_cost
            
            # Predict next state using dynamics model
            x_current = self.dynamics.predict_next_state(x_current, u_k)
        
        # Add terminal cost
        terminal_cost = self.calculate_terminal_cost(x_current, target_ee_pose, base_trajectory[-1])
        total_cost += terminal_cost
        
        return total_cost

    def predict_trajectory(self, initial_state: np.ndarray, 
                          control_sequence: np.ndarray, 
                          horizon: int) -> List[np.ndarray]:
        """Predict state trajectory over horizon"""
        trajectory = [initial_state]
        current_state = initial_state.copy()
        
        for k in range(horizon):
            next_state = self.dynamics.predict_next_state(
                current_state,
                control_sequence[k],
                self.dt
            )
            trajectory.append(next_state)
            current_state = next_state
        
        return trajectory

    # ------------------------------------------------------------------
    # Helper: rotate vectors from WORLD → ARM-BASE frame
    def _world_to_base(self, vec_w: np.ndarray, yaw: float) -> np.ndarray:
        """Rotate a 3-D vector from world frame into the robot’s base-link frame."""
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[ c,  s, 0],
                      [-s,  c, 0],
                      [ 0,  0, 1]])
        return R @ vec_w

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