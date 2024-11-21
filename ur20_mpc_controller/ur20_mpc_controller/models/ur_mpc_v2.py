#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.optimize import minimize
import moveit_commander
from typing import Dict, Tuple, List

from ur20_mpc_controller.models.base_estimator import BaseMotionEstimator
from ur20_mpc_controller.models.ur20_dynamics import UR20Dynamics

class URMPC:
    def __init__(self):
        # Initialize ROS node
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander()
        self.move_group = moveit_commander.MoveGroupCommander("arm")
        
        # Initialize dynamics model
        self.dynamics: UR20Dynamics = UR20Dynamics()
        
        # Load parameters from config
        self.horizon: int = rospy.get_param('~mpc_controller/horizon', 15)
        self.dt: float = rospy.get_param('~mpc_controller/dt', 0.05)
        
        # Add parameter validation
        if self.horizon <= 0:
            raise ValueError("Horizon must be positive")
        if self.dt <= 0:
            raise ValueError("Time step must be positive")
        
        # Load weights
        self.w_ee_pos = rospy.get_param('~mpc_controller/weights/position', 8.0)
        self.w_ee_ori = rospy.get_param('~mpc_controller/weights/orientation', 4.0)
        self.w_control = rospy.get_param('~mpc_controller/weights/control', 0.02)
        
        # Load active weights
        self.lin_weight_active = rospy.get_param('~mpc_controller/active_weights/linear', 15.0)
        self.ang_weight_active = rospy.get_param('~mpc_controller/active_weights/angular', 5.0)
        
        # Load robot configuration
        self.arm_base_offset = {
            'x': rospy.get_param('~mpc_controller/arm_base_offset/x', 0.06),
            'y': rospy.get_param('~mpc_controller/arm_base_offset/y', -0.1),
            'z': rospy.get_param('~mpc_controller/arm_base_offset/z', 0.09),
            'yaw': rospy.get_param('~mpc_controller/arm_base_offset/yaw', np.pi)
        }
        
        self.base_estimator = BaseMotionEstimator()
        self.previous_solution = None
        
        # Add damping weight
        self.w_damping = rospy.get_param('~mpc_controller/weights/damping', 0.5)
        
    def compute_control(self, 
                current_joint_state: Dict,
                current_ee_pose: Dict,
                target_ee_pose: Dict,
                base_state: Dict) -> np.ndarray:
        
        # Check both linear and angular velocity
        base_vel_mag = np.linalg.norm(base_state['linear_velocity'])
        base_ang_mag = np.linalg.norm(base_state['angular_velocity'])
        
        # Debug prints
        rospy.loginfo(f"Base linear velocity magnitude: {base_vel_mag}")
        rospy.loginfo(f"Base angular velocity magnitude: {base_ang_mag}")
        
        if base_vel_mag < 0.01 and base_ang_mag < 0.01:
            return np.zeros(self.dynamics.n_controls)

        # Transform base motion to arm base frame
        transformed_base_state = self._transform_base_motion(base_state)
        
        # Get current state vector
        current_state = self.dynamics.get_state_vector(
            current_joint_state, 
            current_ee_pose
        )
        
        # Predict base motion over horizon
        base_trajectory = self.predict_base_motion(transformed_base_state, self.horizon)
        
        # Get initial guess using Jacobian
        J = self.dynamics.get_jacobian(current_joint_state['position'])
        J_pos = J[:3, :]
        J_ori = J[3:, :]
        
        # Compute required velocities for both position and orientation
        base_lin_vel = -transformed_base_state['linear_velocity'][:3]
        base_ang_vel = -transformed_base_state['angular_velocity']
        
        # Combine into full velocity vector
        J_full = np.vstack([J_pos, J_ori])
        base_vel_full = np.concatenate([base_lin_vel, base_ang_vel])
        
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
        
        try:
            # Optimize over the entire horizon
            result = minimize(
                fun=lambda x: self._cost_function(x, current_state, target_ee_pose, base_trajectory),
                x0=x0,
                method='SLSQP',
                bounds=self._get_bounds(x0),
                constraints=self._get_constraints(current_state, base_trajectory),
                options={
                    'ftol': 1e-2,
                    'maxiter': 10,
                    'eps': 1e-2
                }
            )
            
            # Get first control action (receding horizon principle)
            control = result.x[:self.dynamics.n_controls]
            
            # Store solution for warm start in next iteration
            self.previous_solution = result.x
            
            # Debug output
            predicted_trajectory = self.predict_trajectory(current_state, 
                                                         result.x.reshape(self.horizon, -1), 
                                                         self.horizon)
            ee_vel = J_pos @ control
            ee_ang_vel = J_ori @ control
            
            rospy.loginfo(f"Base linear velocity: {base_state['linear_velocity']}")
            rospy.loginfo(f"Base angular velocity: {base_state['angular_velocity']}")
            rospy.loginfo(f"EE linear velocity: {ee_vel}")
            rospy.loginfo(f"EE angular velocity: {ee_ang_vel}")
            rospy.loginfo(f"Predicted final EE error: {predicted_trajectory[-1][2*self.dynamics.n_q:2*self.dynamics.n_q + 3] - target_ee_pose['position']}")
            
            return control
            
        except Exception as e:
            rospy.logwarn(f"Failed to compute control: {str(e)}")
            return initial_vel

    def calculate_stage_cost(self, state, target_ee_pose, base_state, control, k):
        """Calculate cost for a single stage in the prediction horizon"""
        # Get current end-effector state
        ee_pos = state[2*self.dynamics.n_q:2*self.dynamics.n_q + 3]
        ee_ori = state[2*self.dynamics.n_q + 3:2*self.dynamics.n_q + 6]
        
        # Get Jacobian for velocity computation
        q = state[:self.dynamics.n_q]
        J = self.dynamics.get_jacobian(q)
        J_pos = J[:3, :]
        J_ori = J[3:, :]
        
        # Calculate end-effector velocities
        ee_lin_vel = J_pos @ control
        ee_ang_vel = J_ori @ control
        
        # Get base velocities for compensation
        base_lin_vel = base_state['linear_velocity'][:3]
        base_ang_vel = base_state['angular_velocity']
        
        # Velocity tracking errors
        lin_vel_error = ee_lin_vel + base_lin_vel
        ang_vel_error = ee_ang_vel + base_ang_vel
        
        # Position and orientation errors
        pos_error = ee_pos - target_ee_pose['position']
        ori_error = ee_ori - target_ee_pose['orientation']
        
        # Compute costs with time-varying weights
        decay = 0.95**k  # Weight decay over horizon
        
        # Primary objectives (velocity matching)
        lin_vel_cost = 20.0 * decay * np.sum(lin_vel_error**2)
        ang_vel_cost = 30.0 * decay * np.sum(ang_vel_error**2)
        
        # Secondary objectives (position and orientation holding)
        pos_cost = 1.0 * decay * np.sum(pos_error**2)
        ori_cost = 2.0 * decay * np.sum(ori_error**2)
        
        # Control regularization
        control_cost = 0.01 * np.sum(control**2)
        
        return lin_vel_cost + ang_vel_cost + pos_cost + ori_cost + control_cost

    def calculate_terminal_cost(self, final_state, target_ee_pose, final_base_state):
        """Calculate terminal cost for both position and orientation"""
        ee_pos = final_state[2*self.dynamics.n_q:2*self.dynamics.n_q + 3]
        ee_ori = final_state[2*self.dynamics.n_q + 3:2*self.dynamics.n_q + 6]
        
        # Get Jacobian
        q = final_state[:self.dynamics.n_q]
        J = self.dynamics.get_jacobian(q)
        J_pos = J[:3, :]
        J_ori = J[3:, :]
        
        # Terminal velocities should match base velocities
        terminal_lin_vel = J_pos @ final_state[self.dynamics.n_q:self.dynamics.n_q + self.dynamics.n_dq]
        terminal_ang_vel = J_ori @ final_state[self.dynamics.n_q:self.dynamics.n_q + self.dynamics.n_dq]
        
        base_lin_vel = final_base_state['linear_velocity'][:3]
        base_ang_vel = final_base_state['angular_velocity']
        
        lin_vel_error = terminal_lin_vel + base_lin_vel
        ang_vel_error = terminal_ang_vel + base_ang_vel
        
        pos_error = ee_pos - target_ee_pose['position']
        ori_error = ee_ori - target_ee_pose['orientation']
        
        return (30.0 * np.sum(lin_vel_error**2) + 
                15.0 * np.sum(ang_vel_error**2) + 
                2.0 * np.sum(pos_error**2) + 
                2.0 * np.sum(ori_error**2))

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
        tangential_vel = np.array([
            -omega * offset[1],     # -ω * y
            omega * offset[0],      # ω * x
            0.0
        ])
        
        # Total velocity combines both linear and tangential components
        total_vel = platform_vel + tangential_vel
        
        transformed_state = base_state.copy()
        transformed_state['linear_velocity'] = total_vel
        transformed_state['position'] = base_state['position'] + offset
        transformed_state['angular_velocity'] = platform_ang_vel
        
        # Debug information
        rospy.loginfo("=== Motion Transform Debug ===")
        rospy.loginfo(f"Platform velocity: {platform_vel}")
        rospy.loginfo(f"Platform angular velocity: {platform_ang_vel}")
        rospy.loginfo(f"Offset from rotation center: {offset}")
        rospy.loginfo(f"Tangential velocity: {tangential_vel}")
        rospy.loginfo(f"Total compensation velocity: {total_vel}")
        rospy.loginfo("============================")
        
        return transformed_state

    def predict_base_motion(self, current_base_state: Dict, horizon: int) -> List[Dict]:
        """Predict base motion over horizon"""
        base_trajectory = []
        dt = self.dt
        
        # Get current acceleration (could be estimated from velocity history)
        if hasattr(self, 'prev_base_vel'):
            base_accel = (current_base_state['linear_velocity'] - self.prev_base_vel) / dt
        else:
            base_accel = np.zeros(3)
        
        self.prev_base_vel = current_base_state['linear_velocity'].copy()
        
        for k in range(horizon):
            # Predict velocity using acceleration
            predicted_vel = current_base_state['linear_velocity'] + k * dt * base_accel
            
            # Predict position using updated velocity
            predicted_pos = (current_base_state['position'] + 
                            k * dt * current_base_state['linear_velocity'] +
                            0.5 * k * dt**2 * base_accel)
            
            # Predict orientation (could be improved with angular acceleration)
            predicted_ori = (current_base_state['orientation'] + 
                            k * dt * current_base_state['angular_velocity'])
            
            predicted_state = {
                'position': predicted_pos.copy(),
                'orientation': predicted_ori.copy(),
                'linear_velocity': predicted_vel.copy(),
                'angular_velocity': current_base_state['angular_velocity'].copy()
            }
            
            base_trajectory.append(predicted_state)
        
        return base_trajectory

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