from numpy.random import randn
import matplotlib.pyplot as plt
import numpy as np
# import KalmanFilterImplementation
from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver
from Plots import plot_kf_output
from Plots import plot_residuals
from Plots import plot_measurements
from Plots import plot_residual_limits
from Plots import set_labels
from Plots import plot_filter
from scipy.linalg import inv
from filterpy.kalman import KalmanFilter

def filter_data(gps_sigma, dvs_sigma, do_plot):
    dt = 1

    # pos and velo
    kf = KalmanFilter(dim_x=2, dim_z=2)
    kf.x = np.array([0, .17])
    kf.P = np.array([[1, 0], [0, 1]])
    kf.R[0, 0] = gps_sigma**2
    kf.R[1, 1] = dvs_sigma**2
    kf.Q = Q_discrete_white_noise(2, dt, .03)
    kf.F = np.array([[1., dt],
                        [0., 1]])
    kf.H = np.array([[1., 0],
                        [1., 0]])
    s = Saver(kf)

    np.random.seed(5)
    s.to_array()
    
    residuals = []

    for i in range(1, 3600):
        x = i + np.abs(np.random.randn()) * gps_sigma
        v = i + np.abs(np.random.randn()) * dvs_sigma
        kf.predict()
        kf.update(np.array([[x], [v]]))
        residuals.append(kf.y.copy())
        s.save()
    s.to_array()
    
    residuals = np.array(residuals)

    if do_plot:
        # ts = np.arange(0.1, 10, .1)
        # plot_measurements(ts, s.z[:, 0], label='GPS1')
        # # print(s.z[:, 0])
        # plt.plot(ts, s.z[:, 1], ls='--', label='GPS2')
        # plot_filter(ts, s.x[:, 0], label='Kalman filter')
        # plt.legend(loc=4)
        # plt.ylim(0, 100)
        # set_labels(x='time (sec)', y='meters') 
        # plt.show()
        
        plt.plot(residuals[:, 0], label="residual[0]")
        plt.plot(residuals[:, 1], label="residual[1]")
        plot_residual_limits(s.P[:, 0, 0], 3)
        plt.xlabel("Time step")
        plt.ylabel("Residual")
        plt.legend()
        plt.title("Kalman Filter Residuals (time series)")
        plt.show()
    

filter_data(5, 5, True)
 
  


    
    