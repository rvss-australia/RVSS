import numpy as np

class EKF_Robot:
    def __init__(self, wheels_width, wheels_scale):
        # State is a vector of [x,y,theta]'
        self.state = np.zeros((3,1))
        
        # Wheel parameters
        self.wheels_width = wheels_width  # The distance between the left and right wheels
        self.wheels_scale = wheels_scale  # The scaling factor converting ticks/s to m/s

    def drive(self, drive_meas):
        # left_speed and right_speed are the speeds in ticks/s of the left and right wheels.
        # dt is the length of time to drive for

        # Compute the linear and angular velocity
        linear_velocity, angular_velocity = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)

        # Apply the velocities
        dt = drive_meas.dt
        if angular_velocity == 0:
            self.state[0] += np.cos(self.state[2]) * linear_velocity * dt
            self.state[1] += np.sin(self.state[2]) * linear_velocity * dt
        else:
            th = self.state[2]
            self.state[0] += linear_velocity / angular_velocity * (np.sin(th+dt*angular_velocity) - np.sin(th))
            self.state[1] += -linear_velocity / angular_velocity * (np.cos(th+dt*angular_velocity) - np.cos(th))
            self.state[2] += dt*angular_velocity

    def measure(self, markers, idx_list):
        # Markers are 2d landmarks in a 2xn structure where there are n landmarks.
        # The index list tells the function which landmarks to measure in order.
        
        # Construct a 2x2 rotation matrix from the robot angle
        th = self.state[2]
        
        Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        robot_xy = self.state[0:2,:]

        measurements = []
        for idx in idx_list:
            marker = markers[:,idx:idx+1]
            marker_bff = Rot_theta.T @ (marker - robot_xy)
            measurements.append(marker_bff)

        # Stack the measurements in a 2xm structure.
        markers_bff = np.concatenate(measurements, axis=1)
        return markers_bff
    
    def convert_wheel_speeds(self, left_speed, right_speed):
        # Convert to m/s
        left_speed_m = left_speed * self.wheels_scale
        right_speed_m = right_speed * self.wheels_scale

        # Compute the linear and angular velocity
        linear_velocity = (left_speed_m + right_speed_m) / 2.0
        angular_velocity = (right_speed_m - left_speed_m) / self.wheels_width
        
        return linear_velocity, angular_velocity

    # Derivatives and Covariance
    # --------------------------

    def derivative_drive(self, drive_meas):
        # Compute the differential of drive w.r.t. the robot state
        DFx = np.zeros((3,3))
        DFx[0,0] = 1
        DFx[1,1] = 1
        DFx[2,2] = 1

        lin_vel, ang_vel = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)

        dt = drive_meas.dt
        th = self.state[2]
        if ang_vel == 0:
            DFx[0,2] = -np.sin(th) * lin_vel * dt
            DFx[1,2] = np.cos(th) * lin_vel * dt
        else:
            DFx[0,2] = lin_vel / ang_vel * (np.cos(th+dt*ang_vel) - np.cos(th))
            DFx[1,2] = lin_vel / ang_vel * (np.sin(th+dt*ang_vel) - np.sin(th))

        return DFx

    def derivative_measure(self, markers, idx_list):
        # Compute the derivative of the markers in the order given by idx_list w.r.t. robot and markers
        n = 2*len(idx_list)
        m = 3

        DH = np.zeros((n,m))

        robot_xy = self.state[0:2,:]
        th = self.state[2]
        Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        DRot_theta = np.block([[-np.sin(th), -np.cos(th)],[np.cos(th), -np.sin(th)]])

        for i in range(n//2):
            j = idx_list[i]
            # i identifies which measurement to differentiate.
            # j identifies the marker that i corresponds to.

            lmj_inertial = markers[:,j:j+1]
            # lmj_bff = Rot_theta.T @ (lmj_inertial - robot_xy)

            # robot xy DH
            DH[2*i:2*i+2,0:2] = - Rot_theta.T
            # robot theta DH
            DH[2*i:2*i+2, 2:3] = DRot_theta.T @ (lmj_inertial - robot_xy)
            
        return DH
    
    def covariance_drive(self, drive_meas):
        # Derivative of lin_vel, ang_vel w.r.t. left_speed, right_speed
        Jac1 = np.array([[self.wheels_scale/2, self.wheels_scale/2],
                [-self.wheels_scale/self.wheels_width, self.wheels_scale/self.wheels_width]])
        
        lin_vel, ang_vel = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)
        th = self.state[2]
        dt = drive_meas.dt
        th2 = th + dt*ang_vel

        # Derivative of x,y,theta w.r.t. lin_vel, ang_vel
        Jac2 = np.zeros((3,2))
        if ang_vel == 0:
            Jac2[0,0] = dt*np.cos(th)
            Jac2[0,1] = dt*np.sin(th)
        else:
            Jac2[0,0] = 1/ang_vel * (np.sin(th2) - np.sin(th))
            Jac2[0,1] = -1/(ang_vel**2) * (np.sin(th2) - np.sin(th)) + \
                            lin_vel / ang_vel * (dt * np.cos(th2))

            Jac2[1,0] = -1/ang_vel * (np.cos(th2) - np.cos(th))
            Jac2[1,1] = 1/(ang_vel**2) * (np.cos(th2) - np.cos(th)) + \
                            -lin_vel / ang_vel * (-dt * np.sin(th2))
        Jac2[2,1] = dt

        # Derivative of x,y,theta w.r.t. left_speed, right_speed
        Jac = Jac2 @ Jac1

        # Compute covariance
        cov = np.diag((drive_meas.left_cov, drive_meas.right_cov))
        cov = Jac @ cov @ Jac.T
        
        return cov
