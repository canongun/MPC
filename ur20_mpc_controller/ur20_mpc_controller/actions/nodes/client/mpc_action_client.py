#!/usr/bin/env python3

import rospy
import actionlib
from ur20_mpc_controller.msg import MPCAction, MPCGoal

class MPCActionClient:
    def __init__(self):
        self.client = actionlib.SimpleActionClient(
            'mpc_controller',
            MPCAction
        )
        rospy.loginfo("Waiting for MPC action server...")
        self.client.wait_for_server()
        rospy.loginfo("MPC action server connected!")
        
    def feedback_cb(self, feedback):
        rospy.loginfo("Current EE Position: {}".format(feedback.current_position))
        rospy.loginfo("Current EE Orientation: {}".format(feedback.current_orientation))
        rospy.loginfo("Joint Velocities: {}".format(feedback.joint_velocities))
        rospy.loginfo("------------------------")
    
    def start_compensation(self):
        goal = MPCGoal()
        self.client.send_goal(goal, feedback_cb=self.feedback_cb)
        
    def stop_compensation(self):
        self.client.cancel_goal()

if __name__ == '__main__':
    rospy.init_node('mpc_action_client')
    client = MPCActionClient()
    
    try:
        client.start_compensation()
        rospy.spin()
    except KeyboardInterrupt:
        client.stop_compensation()