from matplotlib.ticker import MultipleLocator
from numpy.random import randn
import matplotlib.pyplot as plt
import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver
from Plots import plot_kf_output
from Plots import plot_residuals
from scipy.linalg import inv
from filterpy.kalman import KalmanFilter


class Airplane(object):
    # Create sensor data 
    # xs = predicted position 
    # zs = sensor values, randomized around xs
    def __init__(self, x0=0, vel=.239, noise_scale=0.06):
        self.x = x0
        self.vel = vel
        self.noise_scale = noise_scale
        
    def update(self):
        self.vel += np.abs(np.random.randn()) * self.noise_scale
        self.x += self.vel
        return (self.x, self.vel)
        
# Create randomized sensor data        
def sense(x, noise_scale = 1.):
    return x[0] + np.abs(np.random.randn()) * noise_scale

# create Airplane and gather sensor data
def simulate_system(Q, count):
    # 600 mph -> .17 mps
    obj = Airplane(x0=0, vel=.239, noise_scale=Q)

    xs, zs = [], []
    for i in range(count):
        x = obj.update()
        z = sense(x)
        xs.append(x)
        zs.append(z)
        
    return np.array(xs), np.array(zs)

# init all matricies for the filter and create the filter
def init_filter(R, Q, dt):
    # pos and velo
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([0, .17])
    kf.P = np.array([[1, 0], [0,1]])
    kf.R = R
    kf.Q = Q_discrete_white_noise(2, dt, Q)
    # x = x0 + vt
    kf.F = np.array([[1., dt],
                     [0.,1]])
    kf.H = np.array([[1., 0]])
    
    return kf

# filter the data and save it
def filter_data(kf, zs):
    s = Saver(kf)
    kf.batch_filter(zs, saver=s)
    s.to_array()
    return s

# Executed:
R, Q = 1, 0.03

np.random.seed(5)

xs, zs = simulate_system(Q=Q, count=120)
np.put(zs, 0, 3.2)
print(zs)

kf = init_filter(R, Q, dt=1)
data1 = filter_data(kf, zs)
  
# track is xs (rough estimation) 
# zs are dots
# data1.x is the filter posteriors 
np.random.seed(5) 

plt.scatter(range(len(zs)), zs)
plot_kf_output(xs, data1.x, data1.z)

highSum = 0
lowSum = 0

i = 0

for element in np.nditer(xs[:, 0] - data1.x[:, 0]):
    if i >= 120:
        break
    if element >= 0:
        highSum += (xs[:, 0] - data1.x[:, 0])[i]
    elif element < 0:
        lowSum += (xs[:, 0] - data1.x[:, 0])[i]   
    i += 1     
        
average = (highSum + lowSum)/(xs[:, 0] - data1.x[:, 0]).size

plt.axhline(average, color='g')

plt.axhline(0, color = 'r')

plot_residuals(xs[:, 0], data1, 0,
               title='Kalman Filter Residuals (time series)',
               y_label='Residuals')   


    
    