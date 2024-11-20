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
        self.dynamics = UR20Dynamics()
        
        # Load parameters from config
        self.horizon = rospy.get_param('~mpc_controller/horizon', 15)
        self.dt = rospy.get_param('~mpc_controller/dt', 0.05)
        
        # Load weights
        self.w_ee_pos = rospy.get_param('~mpc_controller/weights/position', 10.0)
        self.w_ee_ori = rospy.get_param('~mpc_controller/weights/orientation', 2.0)
        self.w_control = rospy.get_param('~mpc_controller/weights/control', 0.05)
        self.w_smooth = rospy.get_param('~mpc_controller/weights/smooth', 0.3)
        self.w_damp = rospy.get_param('~mpc_controller/weights/damp', 0.4)
        
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
        self.previous_solution = None
        
    def compute_control(self, 
                current_joint_state: Dict,
                current_ee_pose: Dict,
                target_ee_pose: Dict,
                base_state: Dict) -> np.ndarray:
        start_time = rospy.Time.now()
        
        # Transform base motion to arm base frame
        transformed_base_state = self._transform_base_motion(base_state)
        
        # Get current state vector
        current_state = self.dynamics.get_state_vector(
            current_joint_state, 
            current_ee_pose
        )
        
        # Predict base motion over horizon (do this once)
        base_trajectory = self.predict_base_motion(transformed_base_state, self.horizon)
        
        # Better initial guess
        if self.previous_solution is not None:
            # Shift previous solution
            x0 = np.roll(self.previous_solution, -self.dynamics.n_controls)
            x0[-self.dynamics.n_controls:] = self.previous_solution[-self.dynamics.n_controls:]
        else:
            x0 = np.zeros(self.dynamics.n_controls * self.horizon)
            # Initialize using Jacobian
            J = self.dynamics.get_jacobian(current_joint_state['position'])
            base_vel = -transformed_base_state['linear_velocity'][:3]
            initial_vel = np.linalg.pinv(J[:3, :]) @ base_vel
            # Apply to all horizon steps
            for i in range(self.horizon):
                x0[i*self.dynamics.n_controls:(i+1)*self.dynamics.n_controls] = initial_vel
        
        # Optimize with reduced tolerance
        try:
            result = minimize(
                fun=lambda x: self._cost_function(x, current_state, target_ee_pose, base_trajectory),
                x0=x0,
                method='SLSQP',
                bounds=self._get_bounds(x0),
                constraints=self._get_constraints(current_state, transformed_base_state),
                options={
                    'ftol': 1e-2,      # Increase tolerance
                    'maxiter': 10,      # Reduce iterations
                    'disp': False,
                    'eps': 1e-2        # Increase step size
                }
            )
            
            # Store solution for warm start
            self.previous_solution = result.x
            
            # Monitor computation time
            compute_time = (rospy.Time.now() - start_time).to_sec()
            rospy.loginfo(f"Computation time: {compute_time:.3f} s")
            
            # Get optimal control
            control = result.x[:self.dynamics.n_controls]
            
            # Calculate distance to target
            ee_error = np.linalg.norm(current_ee_pose['position'] - target_ee_pose['position'])
            
            # Modified velocity scaling
            max_scale = rospy.get_param('~mpc_controller/velocity/max_scale', 0.9)
            min_scale = rospy.get_param('~mpc_controller/velocity/min_scale', 0.3)
            
            # Less aggressive scaling
            scale = min_scale + (max_scale - min_scale) * (1 - np.exp(-1.5 * ee_error))
            
            # Apply scaling
            control = scale * control
            
            return control
            
        except Exception as e:
            rospy.logerr(f"Optimization error: {str(e)}")
            return x0[:self.dynamics.n_controls]

    def calculate_tracking_cost(self, 
                          state: np.ndarray, 
                          target_ee_pose: Dict, 
                          base_state: Dict,
                          k: int,
                          control: np.ndarray) -> float:
        """Calculate tracking cost with velocity compensation"""
        # Get Jacobian at current joint positions
        q = state[:self.dynamics.n_q]
        J = self.dynamics.get_jacobian(q)
        J_pos = J[:3, :]
        J_ori = J[3:, :]
        
        # Calculate end-effector velocities
        ee_lin_vel = J_pos @ control
        ee_ang_vel = J_ori @ control
        
        # Get base velocities (negative for compensation)
        base_lin_vel = -base_state['linear_velocity'][:3]
        base_ang_vel = -base_state['angular_velocity']
        
        # Velocity compensation error
        pos_vel_error = ee_lin_vel - base_lin_vel
        ori_vel_error = ee_ang_vel - base_ang_vel
        
        # Position and orientation error
        ee_pos = state[12:15]
        ee_ori = state[15:18]
        pos_error = ee_pos - (target_ee_pose['position'] + base_state['position'])
        ori_error = ee_ori - (target_ee_pose['orientation'] + base_state['orientation'])
        
        # Combine costs with appropriate weights
        pos_cost = (self.w_ee_pos * np.sum(pos_error**2) + 
                    self.lin_weight_active * np.sum(pos_vel_error**2))
        ori_cost = (self.w_ee_ori * np.sum(ori_error**2) + 
                    self.ang_weight_active * np.sum(ori_vel_error**2))
        
        return pos_cost + ori_cost

    # def calculate_terminal_cost(self, 
    #                       final_state: np.ndarray, 
    #                       target_ee_pose: Dict, 
    #                       final_base_state: Dict) -> float:
    #     """
    #     Calculate terminal cost for final state
    #     """
    #     # Higher weights for terminal state
    #     terminal_weight = 5.0
    #     return terminal_weight * self.calculate_tracking_cost(
    #         final_state, 
    #         target_ee_pose, 
    #         final_base_state, 
    #         k=self.horizon
    #     )

    def _cost_function(self, x: np.ndarray, 
                  current_state: np.ndarray,
                  target_ee_pose: Dict,
                  base_trajectory: List[Dict]) -> float:
        """True MPC cost function with prediction"""
        total_cost = 0.0
        x_current = current_state.copy()  # Start from current state
        
        # Reshape control sequence
        control_sequence = x.reshape(self.horizon, self.dynamics.n_controls)
        
        # Simulate forward over horizon
        for k in range(self.horizon):
            # Get control input for this step
            u_k = control_sequence[k]
            
            # Get predicted base state at time k
            base_k = base_trajectory[k]
            
            # Stage cost - tracking and control
            stage_cost = self.calculate_stage_cost(x_current, target_ee_pose, base_k, u_k, k)
            total_cost += stage_cost
            
            # Predict next state using dynamics
            x_current = self.dynamics.predict_next_state(x_current, u_k)
        
        # Add terminal cost
        terminal_cost = self.calculate_terminal_cost(
            x_current, 
            target_ee_pose, 
            base_trajectory[-1]
        )
        total_cost += terminal_cost
        
        return total_cost

    def calculate_stage_cost(self, 
                        state: np.ndarray,
                        target_ee_pose: Dict,
                        base_state: Dict,
                        control: np.ndarray,
                        k: int) -> float:
        """Calculate cost for a single stage in the prediction horizon"""
        # Get current end-effector state
        ee_pos = state[12:15]
        ee_ori = state[15:18]
        
        # Get Jacobian for velocity computation
        J = self.dynamics.get_jacobian(state[:self.n_q])
        J_pos = J[:3, :]
        J_ori = J[3:, :]
        
        # Compute end-effector velocity
        ee_vel = J_pos @ control
        
        # Predicted base motion compensation
        base_vel = -base_state['linear_velocity'][:3]
        
        # Velocity tracking error
        vel_error = ee_vel - base_vel
        
        # Position tracking error (considering base motion)
        pos_error = ee_pos - (target_ee_pose['position'] + base_state['position'])
        
        # Compute costs with time-varying weights
        time_weight = 0.95**k  # Decrease weight for later predictions
        
        # Stage costs
        vel_cost = self.lin_weight_active * np.sum(vel_error**2)
        pos_cost = self.w_ee_pos * np.sum(pos_error**2)
        control_cost = self.w_control * np.sum(control**2)
        
        return time_weight * (vel_cost + pos_cost + control_cost)

    def calculate_terminal_cost(self,
                          final_state: np.ndarray,
                          target_ee_pose: Dict,
                          final_base_state: Dict) -> float:
        """Calculate terminal cost for stability"""
        # Terminal state error
        ee_pos = final_state[12:15]
        ee_ori = final_state[15:18]
        
        # Position error at terminal state
        pos_error = ee_pos - (target_ee_pose['position'] + final_base_state['position'])
        
        # Higher weights for terminal state
        terminal_weight = 5.0
        terminal_cost = terminal_weight * self.w_ee_pos * np.sum(pos_error**2)
        
        return terminal_cost

    def _get_constraints(self, current_state: np.ndarray, base_state: Dict) -> List[Dict]:
        """Updated constraints using dynamics model"""
        constraints = []
        
        # Dynamic feasibility constraint
        def dynamics_constraint(x):
            control_sequence = x.reshape(self.horizon, self.dynamics.n_controls)
            trajectory = self.dynamics.predict_trajectory(
                current_state,
                control_sequence,
                self.horizon
            )
            # Return array of violations (negative when violated, positive when satisfied)
            violations = []
            for state in trajectory:
                q = state[:self.dynamics.n_q]
                dq = state[self.dynamics.n_q:self.dynamics.n_q + self.dynamics.n_dq]
                
                # Position limits
                pos_violations = np.minimum(
                    q - self.dynamics.joint_pos_limits[:, 0],  # Lower bounds
                    self.dynamics.joint_pos_limits[:, 1] - q   # Upper bounds
                )
                
                # Velocity limits
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

    def predict_base_motion(self, current_base_state: Dict, horizon: int) -> List[Dict]:
        """
        Predict base motion over the horizon
        Args:
            current_base_state: Current base state (position, orientation, velocities)
            horizon: Number of prediction steps
        Returns:
            List of predicted base states over horizon
        """
        base_trajectory = []
        dt = self.dt
        
        for k in range(horizon):
            # Predict position using current velocity
            predicted_pos = current_base_state['position'] + k * dt * current_base_state['linear_velocity']
            
            # Predict orientation using current angular velocity
            predicted_ori = current_base_state['orientation'] + k * dt * current_base_state['angular_velocity']
            
            # Store prediction (assuming constant velocities for now)
            predicted_state = {
                'position': predicted_pos.copy(),
                'orientation': predicted_ori.copy(),
                'linear_velocity': current_base_state['linear_velocity'].copy(),
                'angular_velocity': current_base_state['angular_velocity'].copy()
            }
            
            base_trajectory.append(predicted_state)
            
            # Debug output for first and last prediction
            if k == 0 or k == horizon-1:
                rospy.loginfo(f"Base prediction at step {k}:")
                rospy.loginfo(f"  Position: {predicted_pos}")
                rospy.loginfo(f"  Orientation: {predicted_ori}")
        
        return base_trajectory

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