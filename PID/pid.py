import numpy as np

class PID:

    def __init__(self):
        self.Kp_x = 10.0
        self.Ki_x = 0.0
        self.Kd_x = 0.0
        self.integral_x = 0.0
        self.prev_error_x = 0.0
        
        self.Kp_y = 10.0
        self.Ki_y = 0.0
        self.Kd_y = 0.0
        self.integral_y = 0.0
        self.prev_error_y = 0.0
    

    def update_x(self, position, dt = 0.033):
        error_x = 0.0 - position
        error_x = error_x * 100
        P_x = self.Kp_x * error_x
        # Integral term
        self.integral_x += error_x * dt
        I_x = self.Ki_x * self.integral_x
        # Derivative term
        derivative_x = (error_x - self.prev_error_x) / dt
        D_x = self.Kd_x * derivative_x
        self.prev_error_x = error_x
        # Output is tilt angle in radians
        phi_x = P_x + I_x + D_x
        # Limit to safe tilt range (±15 degrees = ±0.26 rad)
        phi_x = np.clip(phi_x, -15, 25)
        return phi_x
    

    def update_y(self, position, dt = 0.033):
        error_y = 0.0 - position  # Setpoint is 0 (center)
        error_y = error_y * 100
        # Proportional term
        P_y = self.Kp_y * error_y
        # Integral term
        self.integral_y += error_y * dt
        I_y = self.Ki_y * self.integral_y
        # Derivative term
        derivative_y = (error_y - self.prev_error_y) / dt
        D_y = self.Kd_y * derivative_y
        self.prev_error_y = error_y
        # Output is tilt angle in radians
        phi_y = P_y + I_y + D_y
        # Limit to safe tilt range
        phi_y = np.clip(phi_y, -15, 25)
        return phi_y