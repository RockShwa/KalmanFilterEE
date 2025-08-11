from scipy.ndimage import convolve
from scipy.ndimage import shift
import numpy as np
import matplotlib.pyplot as plt
from Plots import bar_plot
from ipywidgets import interact, IntSlider
import random

class Airplane(object):
    def __init__(self, path_len, kernel=[1.], sensor_accuracy=.9):
        self.path_len = path_len
        self.pos = 0
        self.kernel = kernel
        self.sensor_accuracy = sensor_accuracy
        
    def move(self, distance = 1):
         # small chance of error
        self.pos += distance
        # insert random movement error according to kernel
        r = random.random()    
        s = 0
        offset = -(len(self.kernel) - 1) / 2
        for k in self.kernel:
            s += k
            if r <= s:
                break
            offset += 1
        self.pos = int((self.pos + offset) % self.path_len)
        return self.pos
    
    def sense(self):
        pos = self.pos
        # insert random sensor error
        if random.random() > self.sensor_accuracy:
            if random.random() > 0.5:
                pos += 1
            else:
                pos -= 1
        return pos

def predict(posterior, offset, kernal, mode='wrap', cval = 0):
    if mode == 'wrap':
        return convolve(np.roll(posterior, offset), kernal, mode='wrap')
    return convolve(shift(posterior, offset), kernal, cval=cval, mode='constant')
    

def normalize(prior):
    prior /= sum(np.asarray(prior, dtype=float))
    return prior

def update(likelihood, prior):
    return normalize(likelihood * prior)

def discrete_bayes_sim(iterations, kernel, sensor_accuracy, 
                       move_distance, do_print = True):
    path = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    prior = np.array([.9] + [.01]*9)
    posterior = prior[:]
    normalize(prior)
    
    plane = Airplane(len(path), kernel, sensor_accuracy)
    for i in range(iterations):
        plane.move(distance=move_distance)
        
        prior = predict(posterior, move_distance, kernel)
    
        
        m = plane.sense()
        likelihood = lh_positions(path, m, sensor_accuracy)
        posterior = update(likelihood, prior)
        index = np.argmax(posterior)
        
        if do_print:
            print(f'time {i}: pos {plane.pos}, sensed {m}, at position {path[plane.pos]}')
            conf = posterior[index] * 100
            print(f'        estimated position is {index} with confidence {conf:.4f}%:') 
        
    
    bar_plot(posterior)
    if do_print:
        print()
        print('final position is', plane.pos)
        index = np.argmax(posterior)
        conf = posterior[index]*100
        print(f'Estimated position is {index} with confidence {conf:.4f}')
    
   
def lh_positions(positions, z, z_prob):
    """ compute likelihood that a measurement matches
    positions in the hallway."""
    
    try:
        scale = z_prob / (1. - z_prob)
    except ZeroDivisionError:
        scale = 1e8

    likelihood = np.ones(len(positions))
    likelihood[positions==z] *= scale
    return likelihood        

random.seed(3)
discrete_bayes_sim(147, kernel=[.1, .8, .1], sensor_accuracy=.8, move_distance=4, do_print=True)