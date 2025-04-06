# Advices
**06/01/2025**
- In the context, that down the line if would be quite interesting to see how your approaches deals with situations where the arm is forced into singularities by the mobile base.
- Outlook for a later chapter: As the performance of your state estimation (EKF on mobile platform) is quite crucial for you overall system performance, you should explicitly validate it through the MoCap.
- Create another chapter for parameter identification on mobile platform.
- At the end of 2.5.3 Model Predictive Control, you are talking about the choice of optimization algo depends on mode complexity etc. Your final decision on which solver you used should be explained as well - potentially based on a performance comparison of different solvers.

---

# Structure for the Thesis

### 6. State Estimation Framework
This chapter would be critical since, as your supervisor noted, the MPC performance depends heavily on state estimation accuracy. I recommend including:
- Mathematical formulation of the EKF specific to your system
- Fusion methodology for wheel odometry and IMU data
- Calibration procedures for sensors
- Experimental validation using MoCap as ground truth
- Analysis of estimation errors and their impact on controller performance

### 7. Parameter Identification and System Calibration
This would be an excellent follow-up chapter addressing:
- Methodology for identifying platform parameters (mass distribution, friction, etc.)
- Techniques for determining the transformation between base and manipulator
- Procedures for characterizing sensor uncertainties
- Validation of identified parameters against expected values
- Impact of parameter uncertainty on overall system performance

### 8. Simulation Studies
Before presenting real-world results, a simulation chapter would demonstrate:
- Verification of the MPC algorithm in controlled conditions
- Robustness analysis through parameter variation
- Comparison with alternative control strategies
- Performance under various disturbance scenarios

### 9. Experimental Results and Validation
This chapter would present real-world tests showing:
- Experimental setup and testing methodology
- Quantitative performance metrics of end-effector stabilization
- Comparison between simulation predictions and real-world performance
- Analysis of system performance limitations

### 10. Discussion and Future Work
- Critical analysis of your approach's strengths and limitations
- Comparison with state-of-the-art methods from literature
- Potential improvements and extensions
- Applications in maritime logistics contexts

### 11. Conclusion
- Summary of contributions
- Key findings and insights
- Practical implications for mobile manipulation
