#!/usr/bin/env python3

import rospy
import numpy as np
from ur20_mpc_controller.models.ur20_interface import UR20Interface
from ur20_mpc_controller.models.base_observer import BaseObserver
from ur20_mpc_controller.models.ur_mpc import URMPC

def main():
    """Test the complete MPC system"""
    rospy.init_node('test_ur_mpc_system')
    rate = rospy.Rate(10)  # 10 Hz
    
    # Initialize all components
    ur20 = UR20Interface()
    base_observer = BaseObserver()
    mpc = URMPC()
    
    # Wait for base observer to receive data
    rospy.loginfo("Waiting for base data...")
    while not base_observer.is_ready() and not rospy.is_shutdown():
        rate.sleep()
    
    # Set initial target as current pose
    target_ee_pose = ur20.get_current_ee_pose()
    
    # Main control loop
    while not rospy.is_shutdown():
        # Get current states
        current_joint_state = ur20.get_current_joint_states()
        current_ee_pose = ur20.get_current_ee_pose()
        base_state = base_observer.get_base_state()
        
        # Compute control
        control = mpc.compute_control(
            current_joint_state,
            current_ee_pose,
            target_ee_pose,
            base_state
        )
        
        # Print information
        rospy.loginfo("\nMPC Test Results:")
        rospy.loginfo(f"Current EE Position: {current_ee_pose['position']}")
        rospy.loginfo(f"Target EE Position: {target_ee_pose['position']}")
        rospy.loginfo(f"Base Position: {base_state['position']}")
        rospy.loginfo(f"Computed Joint Velocities: {control}")
        rospy.loginfo("------------------------")
        
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass