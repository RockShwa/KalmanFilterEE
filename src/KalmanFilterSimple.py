import numpy as np

class KalmanFilter(object):
    
   def __init__ (self, dim_x, dim_z, x, velo, variance, process_error):
       self.dim_x = dim_x
       self.dim_z = dim_z    
       self.x = x
       self.velo = velo
       self.variance = variance
       self.process_error = process_error
       
   
   
   def predict(self):
        # x = x_prior + F(x)
        dt = 1
        self.x = self.x + self.velo * dt
        
        # variance = variance + process error
        self.variance = self.variance + self.process_error
        