from KalmanFilterSimple import KalmanFilter

class AirplaneFilter(object):
    
    def __init__(self, x0, vel):
        self.x0 = x0
        self.vel = vel
        
plane = AirplaneFilter(0, 0)

def execute_filter():
    filter = KalmanFilter(1, 1, plane.x0, plane.vel, 5)
    filter.predict()