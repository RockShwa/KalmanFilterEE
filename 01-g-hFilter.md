# The g-h Filter

## Building Intuition via Thought Experiments
- **You have to blend prediction and measurement**: prediction without measurement provides no real data and measurement without prediction implies nothing about the validity of the measurement in the real world
- **NEVER THROW INFORMATION AWAY**
- Scaling an estimate:
    - estimate = prediction + 4/10(measurement - prediction)
    - basically saying that our estimate will be 4/10 the measurement and the rest will be from the prediction
        - expressing the belief that the prediction is more accurate than the measurement (this is what the Kalman gain does I believe)
    - measurement - prediction = residual (error, how far off the prediction is from the measurement)
        - smaller residuals = better performance
- predict the next data point, update the estimate with the prediction and next measurement
- Even if our prediction for the gain each day (like gaining 1 lb a day) is wrong, we can use it to update our gain to a more accurate value:
    - new gain = old gain + 1/3((measurement - predicted weight)/1 day)
- this is called the g-h filter, g and h being the two scaling factors (g is for scaling for the measurement and h is for the change in measurement over time)

## They Key Points
- Multiple data points are more accurate than one data point, never throw away inaccurate data
- Always choose a number part way between two data points to create a more accurate estimate
- Predict the next measurement and rate of change based on the current estimate and how much we think it will change
- The new estimate is then chosen as part way between the prediction and next measurement scaled by how accurate each is

## Formal Terminology
- System is the object we want to estimate
- State is the current config or values of the system that interest us
- Measurement is a measured value of the system (can be inaccurate, so not always the same as the state; hidden vs observable)
- State Estimate is the filter's estimate of the state (aka estimate)
- Process model to mathematically model the system
- System error or process eorror is the error in the model
- Predict step = system propagation; uses the process model to form a new state estimate (bc of the process error this estimate is imperfect)
- Update step = measurement update
- One iteration of the system propagation and measurement update is knows as an epoch
- If we're trying to track a thrown ball: the kinematic equation (prediction) doesn't account for drag, but computer vision isn't that accurate either so we would put equal-ish confidence in the measurement in comparison to the prediction
- If we track a helium party balloon in a hurricane, there's no model that would allow us to predict the behavior except over brief time scales, so we would have a filter that emphasizes the measurements over predictions

## The Algorithm

**Initialization**

    1. Initialize the state of the filter
    2. Initialize our belief in the state

**Predict**

    1. Use system behavior to predict state at the next time step
    2. Adjust belief to account for the uncertainty in prediction
    
**Update**

    1. Get a measurement and associated belief about its accuracy
    2. Compute residual between estimated state and measurement
    3. New estimate is somewhere on the residual line

## Notation
- z = Measurement
- subscript k = Time step
- so z<sub>k</sub> = Measurement data at time step k
- Initial time step is k - 1 (so x<sub>k - 1</sub> = x<sub>0</sub> )
- bold font = vector/matrix
- **x** = state, bolded to denote it as a vector
- for instance, it can represent both the initial weight and initial weight gain rate: **x** = [x, x']

## Choice of g and h
- Bayseian aspect of g-h filters
- The filter is only as good as the mathematical model used to express the system

### Varying g
- g is the measurement scale
- as g is larger, we more closely follow the measurement instead of the prediction

### Varying h
- h affects how much we favor the measurement of x' vs our prediction
- If the signal changes a lot (relative to the time stamp), a large h will cause us to react to those changes rapidly
- g and h must reflect the real world situation, if you set it too low it will give you nice looking data but poorly model the situation

- trains can't accelerate/decelerate quickly, so h should be small; measurements are also very inaccurate, so low g too