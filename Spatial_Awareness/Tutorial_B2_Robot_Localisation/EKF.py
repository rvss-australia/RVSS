import numpy as np

class RobotEKF:
    # Implementation of an EKF for SLAM
    # The state is ordered as [x; y; theta; l1x; l1y; ...; lnx; lny]

    # Utility
    # -------
    def __init__(self, robot, markers):
        # State components
        self.robot = robot
        self.markers = markers

        # Covariance matrix
        self.P = np.eye(3)*0.5

    # EKF functions
    # -------------
    def predict(self, raw_drive_meas):
        # The prediction step of EKF
        self.robot.drive(raw_drive_meas)
        self.x_hat = self.robot.state
        F = self.state_transition(raw_drive_meas)
        Q = self.predict_covariance(raw_drive_meas)
        self.P = F @ self.P @ F.T + Q

    def update(self, measurements):
        # Construct measurement index list
        tags = [lm.tag for lm in measurements]
        # Stack measurements and set covariance
        z = np.concatenate([lm.position.reshape(-1,1) for lm in measurements], axis=0)
        R = np.zeros((2*len(measurements),2*len(measurements)))
        for i in range(len(measurements)):
            R[2*i:2*i+2,2*i:2*i+2] = measurements[i].covariance

        # Compute own measurements
        z_hat = self.robot.measure(self.markers, tags)
        z_hat = z_hat.reshape((-1,1),order="F")
        H = self.robot.derivative_measure(self.markers, tags)
        
        y = z - z_hat
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        x = self.x_hat + K @ y
        
        self.robot.state = x
        self.P = (np.eye(x.shape[0]) - K @ H) @ self.P

    def state_transition(self, raw_drive_meas):
        F = self.robot.derivative_drive(raw_drive_meas)
        return F
    
    def predict_covariance(self, raw_drive_meas):
        Q = self.robot.covariance_drive(raw_drive_meas)
        return Q
