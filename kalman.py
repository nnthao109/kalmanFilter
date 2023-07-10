# x estimate value ~ convariance P noise_convariance Q
# z measurement noise_convariance R
# F transition model
# H measurement model 
## PREDICT STEP
# x = Fx 
# P = F*P*P^T + Q 
## UPDATE STEP
# z = Hx 
# diff y = Z - z 
# total error = HPH^T + R 
# Kalman Gain K = PH^T / (HPH^T + R) 
# new x = x + K*y 

import numpy as np
from numpy.linalg import inv

class KalmanFilter : 
    def __init__(self, X, F, Q, Z, H, R, P) :
        """ 
            Args:
            X: State estimate
            P: Estimate covariance
            F: State transition model
            Q: Process noise covariance
            Z: Measurement of the state X
            H: Observation model
            R: Observation noise covariance
        """
        self.X = X 
        self.F = F
        self.H = H 
        self.P = P 
        self.Q = Q 
        self.R = R 
        self.Z = Z 
    
    def predict(self):
        """
            X = FX
            P = F*P*F^T + Q
        """
        self.X = self.F @ self.X 
        self.P = self.F @ self.P @ self.F.T + self.Q 

        return self.X

    def correct(self, Z):
        """
            diff y = Z - H*X
            S = H*P*H^T + R 
            K = P*H^T*S^-1
            X = X + K*y
            P = P - K*H*P
        """
        K = self.P @ self.H.T @ inv(self.H @ self.P @ self.H.T + self.R)
        self.X += K @ (Z - self.H @ self.X)
        self.P = self.P - K @ self.H @ self.P

        return self.X
        