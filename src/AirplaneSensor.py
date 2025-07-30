from numpy.random import randn
import matplotlib.pyplot as plt
import numpy as np

class AirplaneSensor(object):
    
    def __init__(self, pos=(0,0), vel=(0,0), noise_std=1.):
        self.pos = [pos[0], pos[1]]
        self.noise_std = noise_std
        self.vel = vel
        
    def read(self):
        self.pos[0] += self.vel[0]   
        self.pos[1] += self.vel[1]
        
        return [self.pos[0] + randn() * self.noise_std, 
                self.pos[1] + randn() * self.noise_std] 
        
pos, vel = (4, 3), (2, 1) 
sensor = AirplaneSensor(pos, vel, noise_std = 1)
ps = np.array([sensor.read() for _ in range(50)])    
plt.scatter(ps[:, 0], ps[:, 1])  
plt.show()
        