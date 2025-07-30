from numpy.random import randn
import matplotlib.pyplot as plt
import numpy as np
import KalmanFilterImplementation
from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver
from Plots import plot_kf_output

class AirplaneFilter(object):
    # TOOD: How to choose noise scale?
    def __init__(self, x0=0, vel=1., noise_scale=0.06):
        self.x = x0
        self.vel = vel
        self.noise_scale = noise_scale
        
    def update(self):
        self.vel += randn() * self.noise_scale
        self.x += self.vel
        return (self.x, self.vel)
        
        
def sense(x, noise_scale = 1.):
    return x[0] + np.abs(np.random.randn()) # * noise_scale

def simulate_system(Q, count):
    # 600 mph -> .17 mps
    obj = AirplaneFilter(x0=0, vel=.17, noise_scale=Q)

    xs, zs = [], []
    for i in range(count):
        x = obj.update()
        z = sense(x)
        print(z)
        xs.append(x)
        zs.append(z)
        
    return np.array(xs), np.array(zs)

# All I need to change for the airplane
def FirstOrderKF(R, Q, dt):
    # pos and velo
    kf = KalmanFilterImplementation.KalmanFilterImplementation(dim_x=2, dim_z=1)
    kf.x = np.array([0, .17])
    # variance of pos and velo
    kf.P *= np.array([[1, 0], [0,1]])
    kf.R *= R
    kf.Q = Q_discrete_white_noise(2, dt, Q)
    # x = x0 + vt
    kf.F = np.array([[1., dt],
                     [0.,1]])
    kf.H = np.array([[1., 0]])
    return kf

def filter_data(kf, zs):
    s = Saver(kf)
    kf.batch_filter(zs, saver=s)
    s.to_array()
    return s

R, Q = 1, 0.03
xs, zs = simulate_system(Q=Q, count=50)

kf = FirstOrderKF(R, Q, dt=1)
data1 = filter_data(kf, zs)
    
# plt.scatter(range(len(zs)), zs)
# plot_kf_output(xs, data1.x, data1.z)

    
    