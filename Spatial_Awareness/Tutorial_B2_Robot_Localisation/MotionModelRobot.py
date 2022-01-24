import numpy as np

class Robot:
    def __init__(self, wheels_width, wheels_scale):
        # State is a vector of [x,y,theta]'
        self.state = np.zeros((3,1))
        
        # Wheel parameters
        self.wheels_width = wheels_width  # The distance between the left and right wheels
        self.wheels_scale = wheels_scale  # The scaling factor converting ticks/s to m/s
            
    def drive(self, drive_meas):
        # left_speed and right_speed are the speeds in ticks/s of the left and right wheels.
        # dt is the length of time to drive for

        # Compute the linear and angular velocity from wheel speeds
        linear_velocity, angular_velocity = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)

        #This is the current state of the robot
        x_k = self.state[0]
        y_k = self.state[1]
        theta_k = self.state[2]
        
        # Apply the velocities
        dt = drive_meas.dt
        if angular_velocity == 0:
            #-----------------------------FILL OUT DRIVE STRAIGHT CODE--------------
            x_kp1 = x_k + np.cos(theta_k)*linear_velocity*dt
            y_kp1 = y_k + np.sin(theta_k)*linear_velocity*dt
            theta_kp1 = theta_k
            #-----------------------------------------------------------------------
        else:
            #-----------------------------FILL OUT DRIVE CODE-----------------------
            x_kp1 = x_k + linear_velocity / angular_velocity * (np.sin(theta_k+dt*angular_velocity) - np.sin(theta_k))
            y_kp1 = y_k + linear_velocity / angular_velocity * (-np.cos(theta_k+dt*angular_velocity) + np.cos(theta_k))
            theta_kp1 = theta_k + angular_velocity*dt
            #-----------------------------------------------------------------------
       
        #Save our state 
        self.state[0] = x_kp1
        self.state[1] = y_kp1
        self.state[2] = theta_kp1
            
            
            
            
    def convert_wheel_speeds(self, left_speed, right_speed):
        # Convert to m/s
        left_speed_m = left_speed * self.wheels_scale
        right_speed_m = right_speed * self.wheels_scale

        # Compute the linear and angular velocity
        linear_velocity = (left_speed_m + right_speed_m) / 2.0
        angular_velocity = (right_speed_m - left_speed_m) / self.wheels_width
        
        return linear_velocity, angular_velocity

