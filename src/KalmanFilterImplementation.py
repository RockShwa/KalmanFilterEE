import numpy as np
from numpy import dot, zeros, eye, isscalar, shape
from scipy.linalg import inv

class KalmanFilterImplementation(object):
    
    # multivar Kalman Filter class
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros((dim_x,1)) # state of the system
        self.P = np.eye(dim_x) # covariance matrix
        self.F = np.eye(dim_x) # state transition function
        self.Q = np.eye(dim_x) # process covariance (error of the model)
        self.B = None # control transition matrix
        self.u = None # control input
        self.H = np.zeros((dim_x, dim_x)) # measurement function
        self.z = np.array([None]*self.dim_z).T # measurement
        self.R = np.eye(dim_z) # noise covariance
        self.y = np.zeros((dim_z,1)) # residual
        self.K = np.zeros((dim_x, dim_z)) # kalman gain
    
        # a way to represent 1 in multiple dimensions
        # just creates a diagonal of 1s
        self.I = np.eye(dim_x)
        
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        
    def predict(self, u=None, B=None, F=None, Q=None):
        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q
        
        # x = Fx + Bu
        if B is not None and u is not None:
            # self.x = np.dot(F, self.x) + np.dot(B, u)
            self.x = F @ self.x + B @ u
        else:
            self.x = F @ self.x 
            
        # P = FPF' + Q
        self.P = F @ self.P @ F.T + Q
        
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        
       
        
    def update(self, z, R=None, H=None):

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        if H is None:
            z = np.reshape_z(z, self.dim_z, self.x.ndim)
            H = self.H
            
        self.y = z - H @ self.x
        # K = PH'inverse(HPH'+R)
        self.K = self.P @ H.T @ inv(H @ self.P @ H.T + R)
        
        self.x = self.x + self.K @ self.y
       
        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self.I - self.K @ H
        self.P = I_KH @ self.P @ I_KH.T + self.K @ R @ self.K.T
         
        #print(self.x)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        
        
        
    def batch_filter(self, zs, Fs = None, Qs = None, Hs = None, Rs = None, Bs = None, us = None, saver = None):
        
        # all these are identity matriciese at the first time step, then we multiply by the size number 
        # of measurements to determine the length of each Fs/Qs etc. (n is the number of epochs, it makes n copies of whatever F is)
        n = np.size(zs, 0)
        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n
        if Hs is None:
            Hs = [self.H] * n
        if Rs is None:
            Rs = [self.R] * n
        if Bs is None:
            Bs = [self.B] * n
        if us is None:
            us = [0] * n   
            
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.predict(u=u, B=B, F=F, Q=Q)
                
                self.update(z, R=R, H=H)

                if saver is not None:
                    saver.save()
                    
                    
        