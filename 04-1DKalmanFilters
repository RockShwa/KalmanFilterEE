## One Dimensional Kalman Filters
- Only tracks one state variable, such as position on the x-axis
- All a Kalman Filter is is a Bayesian filter that uses Gaussians

## Tracking with Gaussian Probabilites
- Bayesian Filter Equations:
$$\begin{aligned} 
\bar {\mathbf x} &= \mathbf x \ast f_{\mathbf x}(\bullet)\, \, &\text{Predict} \\
\mathbf x &= \mathcal L \cdot \bar{\mathbf x}\, \, &\text{Update}
\end{aligned}$$
- Recall that $\bar{\mathbf x}$ is the *prior*, $\mathcal L$ is the *likelihood* of a measurement given the prior $\bar{\mathbf x}$, $f_{\mathbf x}(\bullet)$ is the *process model*, and $\ast$ denotes *convolution*
- We can replace $\mathbf x$ with a Gaussian function: $\mathcal N(x, \sigma^2)$
    -  Note that the state is the mean!
- Replace wuth Gaussians:
$$\begin{array}{l|l|c}
\text{discrete Bayes} & \text{Gaussian} & \text{Step}\\
\hline
\bar {\mathbf x} = \mathbf x \ast f(\mathbf x) & 
\bar {x}_\mathcal{N} =  x_\mathcal{N} \, \oplus \, f_{x_\mathcal{N}}(\bullet) &
\text{Predict} \\
\mathbf x = \|\mathcal L \bar{\mathbf x}\| & x_\mathcal{N} = L \, \otimes \, \bar{x}_\mathcal{N} & \text{Update} 
\end{array}$$
- where $\oplus$ and $\otimes$ is meant to express some unknown operator on Gaussians. The subscript indicates that $x_\mathcal{N}$ is a Gaussian.
- The sum of two Gaussians is another Gaussian! 

$$\begin{gathered}
x=\mathcal N(10, 0.2^2)\\
f_x = \mathcal N (15, 0.7^2)
\end{gathered}$$

If we add these we get:

$$\begin{aligned}\bar x &= \mu_x + \mu_{f_x} = 10 + 15 &&= 25 \\
\bar\sigma^2 &= \sigma_x^2 + \sigma_{f_x}^2 = 0.2^2 + 0.7^2 &&= 0.53\end{aligned}$$

- It makes sense that the predicted position is the previous position plus the movement. What about the variance? We don't really know where the dog is moving, so the confidence should get smaller (variance gets larger). $\sigma_{f_x}^2$ is the amount of uncertainty added to the system due to the imperfect prediction about the movement, and so we would add that to the existing uncertainty. 

## Updates with Gaussians
- Likelihood is the probability of the measurement given the current state, and measurements can be represented as Gaussians
- The prior/prediction can also be represented as a Gaussian
- So multiplying the likelihood and prior = another Gaussian! (once you normalize it)

## Understanding Gaussian Multiplication
- Mean remains unchanged, but variance gets smaller if you multiply two Gaussians with the same mean
    - This makes sense: if I measure something twice and get 10 meters each time, I can conclude the length is very close to 10 meters with more confidence in that measurement (smaller variance)
- If the means are different, and multiply them, the new mean is the average and our certainity in the measurements still gets greater, so the variance is smaller
    - We have no reason to believe one measurement is inaccurate to the point we should throw it out, so we use all information given to us

## First Kalman Filter
~~~ python
# the square root fo the variance is the distance error in meters
process_var = 1. # variance in the dog's movement (process model)
sensor_var = 2. # variance in the sensor

x = gaussian(0., 20.**2)  # dog's position, N(0, 20**2)
velocity = 1
dt = 1. # time step in seconds
# prediction of change at this time stamp 
process_model = gaussian(velocity*dt, process_var) # displacement to add to x
  
# simulate dog and get measurements
dog = DogSimulation(
    x0=x.mean, 
    velocity=process_model.mean, 
    measurement_var=sensor_var, 
    process_var=process_model.var)

# create list of measurements
zs = [dog.move_and_sense() for _ in range(10)]

# perform Kalman filter on measurement z
for z in zs:    
    prior = predict(x, process_model)
    likelihood = gaussian(z, sensor_var)
    x = update(prior, likelihood)

    kf_internal.print_gh(prior, x, z)
~~~
- Prediction step, variance gets bigger bc of addition and loss of information
- When we get the result: (1.352, 1.990), z = 1.354 and we predicted that the dog was at 1.0 with a variance of 401.0, the filter places greater confidence in the measurement since the variance is so high
~~~ python
# all you need for a kalman filter!
prior = predict(x, process_model)
likelihood = gaussian(z, sensor_var)
x = update(prior, likelihood)

# or 

for z in zs:
    # predict
    dx = velocity*dt
    pos = pos + dx
    var = var + process_var

    # update
    # this normalizes the pos by converting it to be in range of the variances and of proper magnitude within the variances
    pos  = (var*z + sensor_var*pos) / (var + sensor_var)
    var = (var * sensor_var) / (var + sensor_var)
~~~

## Kalman Gain
- The posterior x is computed as the likelihood times the prior ($\mathcal L \bar x$), where both are Gaussians
- The mean of the posterior is given by:
$$
\mu=\frac{\bar\sigma^2\, \mu_z + \sigma_z^2 \, \bar\mu} {\bar\sigma^2 + \sigma_z^2}
$$
- The subscript $z$ to denotes the measurement. We can rewrite this as:

$$\mu = \left( \frac{\bar\sigma^2}{\bar\sigma^2 + \sigma_z^2}\right) \mu_z + \left(\frac{\sigma_z^2}{\bar\sigma^2 + \sigma_z^2}\right)\bar\mu$$

In this form it is easy to see that we are scaling the measurement and the prior by weights (weighting each mean differently): 

$$\mu = W_1 \mu_z + W_2 \bar\mu$$

- The weights sum to one because the denominator is a normalized term, so if $K=W_1$

$$\begin{aligned}
\mu &= K \mu_z + (1-K) \bar\mu\\
&= \bar\mu + K(\mu_z - \bar\mu)
\end{aligned}$$

where

$$K = \frac {\bar\sigma^2}{\bar\sigma^2 + \sigma_z^2}$$
- K is the Kalman gain, it's the scaling term that chooses a value partway between $\mu_z$ and $\bar\mu$
- If the measurement is 9x more accurate than the prior, then the weight of the measurement (or Kalman Gain) is larger and contributes more to the mean

~~~ python
def update(prior, measurement):
    x, P = prior        # mean and variance of prior
    z, R = measurement  # mean and variance of measurement
    
    y = z - x        # residual
    K = P / (P + R)  # Kalman gain

    x = x + K*y      # posterior
    P = (1 - K) * P  # posterior variance
    return gaussian(x, P)

def predict(posterior, movement):
    x, P = posterior # mean and variance of posterior
    dx, Q = movement # mean and variance of movement
    x = x + dx
    P = P + Q
    return gaussian(x, P)
~~~
- Q = Proccess noise; R = measurement noise
- You only multiply K by y, so basically if K is large that means the residual (this means the measurement is more accurate) gets more say in the update, but if it's small it means the prediction is more accurate
- The generic algorithm:

Initialization

1. Initialize the state of the filter
2. Initialize our belief in the state
Predict

1. Use system behavior to predict state at the next time step
2. Adjust belief to account for the uncertainty in prediction
Update

1. Get a measurement and associated belief about its accuracy
2. Compute residual between estimated state and measurement
3. Compute scaling factor based on whether the measurement
or prediction is more accurate
4. set state between the prediction and measurement based 
on scaling factor
5. update belief in the state based on how certain we are 
in the measurement

The equations for the univariate Kalman filter are:

<u>Predict</u>

$\begin{array}{|l|l|l|}
\hline
\text{Equation} & \text{Implementation} & \text{Kalman Form}\\
\hline
 \bar x = x + f_x & \bar\mu = \mu + \mu_{f_x} & \bar x = x + dx\\
& \bar\sigma^2 = \sigma^2 + \sigma_{f_x}^2 & \bar P = P + Q\\
\hline
\end{array}$


<u>Update</u>

$\begin{array}{|l|l|l|}
\hline
\text{Equation} & \text{Implementation}& \text{Kalman Form}\\
\hline
 x = \| \mathcal L\bar x\| & y = z - \bar\mu & y = z - \bar x\\
 & K = \frac {\bar\sigma^2} {\bar\sigma^2 + \sigma_z^2} & K = \frac {\bar P}{\bar P+R}\\
 & \mu = \bar \mu + Ky & x = \bar x + Ky\\
 & \sigma^2 = \frac {\bar\sigma^2 \sigma_z^2} {\bar\sigma^2 + \sigma_z^2} & P = (1-K)\bar P\\
\hline
\end{array}$

## Intro to Designing a Filter
- Thermometer outputs a voltage sensor that corresponds to the temperature that is being measured; sensor exhibits white noise with a standard deviation of 0.13 volts
- To allow more error with change over time, change the process error (for the process model)
- Even with large amounts of error, the high certainty in the process error (small variance) means the filter relys on the prediction even with crazy measurements
- However, having too low process error won't follow large changes in a system over time (lag behind the real changes, so if smth increases exponentially the filter would increase linerally)
- Bad Initial Estimate: can recover because we're certain about our sensor values, so that would overcome the wild predictions as a result of a bad estimate and eventually level out
- Large Noise and Bad Initial Estimate: the filter does struggle here bc both the prediction and the measurement are bad (it will get better after a few hundred iterations)
    - You can use the first measurement as the intitial condition to significantly improve results even with lots of noise