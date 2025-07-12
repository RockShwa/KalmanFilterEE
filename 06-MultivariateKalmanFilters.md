# Multivariate Kalman Filters
- Correlation btwn variables drastically improves posterior (like position and velocity)
- When you design a Kalman filter you start with a system of differential equations that describe the dynamics of the system

## Kalman Filter Algorithm
Initialization

1. Initialize the state of the filter
2. Initialize our belief in the state
Predict

1. Use process model to predict state at the next time step
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

<u>**Predict**</u>

$\begin{array}{|l|l|l|}
\hline
\text{Univariate} & \text{Univariate} & \text{Multivariate}\\
& \text{(Kalman form)} & \\
\hline
\bar \mu = \mu + \mu_{f_x} & \bar x = x + dx & \bar{\mathbf x} = \mathbf{Fx} + \mathbf{Bu}\\
\bar\sigma^2 = \sigma_x^2 + \sigma_{f_x}^2 & \bar P = P + Q & \bar{\mathbf P} = \mathbf{FPF}^\mathsf T + \mathbf Q \\
\hline
\end{array}$

Without worrying about the specifics of the linear algebra, we can see that:

$\mathbf x,\, \mathbf P$ are the state mean and covariance. They correspond to $x$ and $\sigma^2$.

$\mathbf F$ is the *state transition function*. When multiplied by $\bf x$ it computes the prior. 

$\mathbf Q$ is the process covariance. It corresponds to $\sigma^2_{f_x}$.

$\mathbf B$ and $\mathbf u$ are new to us. They let us model control inputs to the system.

<u>**Update**</u>

$\begin{array}{|l|l|l|}
\hline
\text{Univariate} & \text{Univariate} & \text{Multivariate}\\
& \text{(Kalman form)} & \\
\hline
& y = z - \bar x & \mathbf y = \mathbf z - \mathbf{H\bar x} \\
& K = \frac{\bar P}{\bar P+R}&
\mathbf K = \mathbf{\bar{P}H}^\mathsf T (\mathbf{H\bar{P}H}^\mathsf T + \mathbf R)^{-1} \\
\mu=\frac{\bar\sigma^2\, \mu_z + \sigma_z^2 \, \bar\mu} {\bar\sigma^2 + \sigma_z^2} & x = \bar x + Ky & \mathbf x = \bar{\mathbf x} + \mathbf{Ky} \\
\sigma^2 = \frac{\sigma_1^2\sigma_2^2}{\sigma_1^2+\sigma_2^2} & P = (1-K)\bar P &
\mathbf P = (\mathbf I - \mathbf{KH})\mathbf{\bar{P}} \\
\hline
\end{array}$
$\mathbf H$ is the measurement function. We haven't seen this yet in this book and I'll explain it later. If you mentally remove $\mathbf H$ from the equations, you should be able to see these equations are similar as well.

$\mathbf z,\, \mathbf R$ are the measurement mean and noise covariance. They correspond to $z$ and $\sigma_z^2$ in the univariate filter (I've substituted $\mu$ with $x$ for the univariate equations to make the notation as similar as possible).

$\mathbf y$ and $\mathbf K$ are the residual and Kalman gain. 

Your job as a designer will be to design the state $\left(\mathbf x, \mathbf P\right)$, the process $\left(\mathbf F, \mathbf Q\right)$, the measurement $\left(\mathbf z, \mathbf R\right)$, and the  measurement function $\mathbf H$. If the system has control inputs, such as a robot, you will also design $\mathbf B$ and $\mathbf u$.

## Tracking a Dog
- Hidden variables used to improve our estimates

### Predict Step
- Design state, covariance, process model and process model noise and control input

#### Design State Variable
- Mean is the most likely position and the variance ($\sigma^2$) represented the probability distribution of the position
    - The position is the state of the system, and $\mu$ is the state variable
- Track both position and velocity, which requires us to use a multivariate Gaussian represented with the state vector $\mathbf x$ and corresponding covariance matrix $\mathbf P$
- Position is the observed variable (observed through a sensor) and velocity is the hidden variable
- $\mathbf x$ is the state (also the mean of the multivariate Gaussian, where we think the dog is):
$$\mathbf x =\begin{bmatrix}x \\ \dot x\end{bmatrix}$$
- Another way to write this is $\mathbf x =\begin{bmatrix}x & \dot x\end{bmatrix}^\mathsf T$ because the transpose of a row vector is a column vector. 

#### Design State Covariance
- We specify an initial value for $\mathbf P$ and the filter updates it during each epoch.
- Top speed for a dog is around 21 m/s, so in the absence of any other information about the velocity we can set $3\sigma_\mathtt{vel}=21$, or $\sigma_\mathtt{vel}^2=7^2=49$. 
- If you don't know the covariances you set them initially to 0
- Linear algebra has a powerful way to express systems of equations. Take this system

$$\begin{cases}
2x+3y=8\\4x-y=2
\end{cases}$$

We can put this in matrix form by writing:

$$\begin{bmatrix}2& 3 \\ 4&-1\end{bmatrix} \begin{bmatrix}x\\y\end{bmatrix} = \begin{bmatrix}8\\2\end{bmatrix}$$

- $\mathbf{\bar x}$ is the *prior*, or predicted state:

$$\mathbf{\bar x} = \mathbf{Fx}$$

which we can make explicit as

$$\begin{bmatrix} \bar x \\ \dot{\bar x}\end{bmatrix} = \begin{bmatrix}? & ? \\? & ?\end{bmatrix}\begin{bmatrix}x\\\dot x\end{bmatrix}$$

This gives us the process model for our system 
- The first equation gives the predicted positon, and the second gives the predicted velocity
$$\begin{cases}
\begin{aligned}
\bar x &= x + \dot x \Delta t \\
\bar{\dot x} &= \dot x
\end{aligned}
\end{cases}$$

This correctly has one equation for each variable in the state, isolated on the left hand side. We need to express this set of equations in the form $\bar{\mathbf x}  = \mathbf{Fx}$. Rearranging terms makes it easier to see what to do.

$$\begin{cases}
\begin{aligned}
\bar x &= 1x + &\Delta t\, \dot x \\
\bar{\dot x} &=0x + &1\, \dot x
\end{aligned}
\end{cases}$$

We can rewrite this in matrix form as

$$\begin{aligned}
\begin{bmatrix}\bar x \\ \bar{\dot x}\end{bmatrix} &= \begin{bmatrix}1&\Delta t  \\ 0&1\end{bmatrix}  \begin{bmatrix}x \\ \dot x\end{bmatrix}\\
\mathbf{\bar x} &= \mathbf{Fx}
\end{aligned}$$

- F is the state transition function or the state transition matrix

#### Design Process 
- We can model this system with the differential equation

$$\dot{\mathbf x} = f(\mathbf x) + w$$

where $f(\mathbf x)$ models the state transition and $w$ is *white process noise*. For now you just need to know that we account for the  noise in the system by adding a process noise covariance matrix $\mathbf Q$ to the covariance $\mathbf P$. We do not add anything to $\mathbf x$ because the noise is *white* - which means that the mean of the noise will be 0. If the mean is 0, $\mathbf x$ will not change.

#### Design the Control Function
- B and u allow us to incorporate the control inputs of systems in airplanes or robots
    - Follow a path: send steering and velocity signals based on current pos vs desired pos

- Here $\mathbf u$ is the *control input*, and $\mathbf B$ is the *control input model* or *control function*. For example, $\mathbf u$ might be a voltage controlling how fast the wheel's motor turns, and multiplying by $\mathbf B$ yields $\Delta[\begin{smallmatrix}x\\\dot x\end{smallmatrix}]$. In other words, it must compute how much $\mathbf x$ changes due to the control input.

### Update Step
- 2 more matricies

#### Design the Measurement Function
- Filter computes update step in the **measurement space**
- Compute the residual bc we scale it by the Kalman Gain to get the new estimate
    - Its like if we tried calculate the residual from voltage and temp: the units don't match
    - We have to supply a measurement function that converts a state into a measurement
    - We can't just convert the voltage to temperature bc the measurements aren't invertible (there's no way to convert a state containing position and velocity into an equivalent measurement of only position )
    - We use the matrix *$\mathbf H$ to perform the conversion (it's the measurement function)
- We need to design $\mathbf H$ so that $\mathbf{H\bar x}$ yields a measurement. For this problem we have a sensor that measures position, so $\mathbf z$ will be a one variable vector:

$$\mathbf z = \begin{bmatrix}z\end{bmatrix}$$

The residual equation will have the form

$$
\begin{aligned}
\textbf{y} &= \mathbf z - \mathbf{H\bar x}  \\
\begin{bmatrix}y \end{bmatrix} &= \begin{bmatrix}z\end{bmatrix} - \begin{bmatrix}?&?\end{bmatrix} \begin{bmatrix}x \\ \dot x\end{bmatrix}
\end{aligned}
$$
- We will want to multiply the position $x$ by 1 to get the corresponding measurement of the position. We do not need to use velocity to find the corresponding measurement so we multiply  $\dot x$ by 0.

$$\begin{aligned}
\textbf{y} &= \mathbf z - \begin{bmatrix}1&0\end{bmatrix} \begin{bmatrix}x \\ \dot x\end{bmatrix} \\
&= [z] - [x]
\end{aligned}$$

And so, for our Kalman filter we set

$$\mathbf H=\begin{bmatrix}1&0\end{bmatrix}$$

#### Design the Measurement
- Implemented with z, the measurement mean, and R, the measurement covariance
- If we have two sensors/measurements we'd have:
$$\mathbf z = \begin{bmatrix}z_1 \\ z_2\end{bmatrix}$$
- The Kalman filter equations uses a covariance matrix $\mathbf R$ for the measurement noise. The matrix will have dimension $m{\times}m$, where $m$ is the number of sensors. It is a covariance matrix to account for correlations between the sensors. We have only 1 sensor so $\mathbf R$ is:

$$\mathbf R = \begin{bmatrix}\sigma^2_z\end{bmatrix}$$
- $$\mathbf R = \begin{bmatrix}5&0\\0&3\end{bmatrix}$$

We put the variances on the diagonal because this is a *covariance* matrix, where the variances lie on the diagonal, and the covariances, if any, lie in the off-diagonal elements. Here we assume there is no correlation in the noise between the two sensors, so the covariances are 0.

### Implementing the Kalman Filter
- **Go to this section for programming the Multivariate filter**
- The covariance matrix $\mathbf P$ tells us the *theoretical* performance of the filter *assuming* everything we tell it is true. Recall that the standard deviation is the square root of the variance, and that approximately 68% of a Gaussian distribution occurs within one standard deviation. If at least 68% of the filter output is within one standard deviation  the filter may be performing well. In the top chart I have displayed the one standard deviation as the yellow shaded area between the two dotted lines. To my eye it looks like perhaps the filter is slightly exceeding that bounds, so the filter probably needs some tuning.

#### Prediction Equations
- Computing the prior:
They compute the prior mean ($\bar{\mathbf x}$)  and covariance ($\bar{\mathbf P}$) of the system.

$$\begin{aligned}
\mathbf{\bar x} &= \mathbf{Fx} + \mathbf{Bu}\\
\mathbf{\bar P} &= \mathbf{FPF}^\mathsf T + \mathbf Q
\end{aligned}$$

- Mean: 
    - Ax = b represents a system of equations, where A holds the coefficients set of equations, x is the vector of variables (multiplying them together gives the set of equations)
    - F is the state transition for a given time stamp, so the product of F 

- Covariance
    - $\mathbf{\bar P} = \mathbf{FPF}^\mathsf T + \mathbf Q$
    - Can't simply add the covariances from the time stamp and error as the state variables are correlated (in all cases, it's a multivariate Gaussian)
- Expressions in the form $\mathbf{ABA}^\mathsf T$ are common in linear algebra. You can think of it as *projecting* the middle term by the outer term. We will be using this many times in the rest of the book. I admit this may be a 'magical' equation to you. Let's explore it.

$$\mathbf{\bar P} = \mathbf{FPF}^\mathsf T + \mathbf Q$$

Expressions in the form $\mathbf{ABA}^\mathsf T$ are common in linear algebra. You can think of it as *projecting* the middle term by the outer term. We will be using this many times in the rest of the book. I admit this may be a 'magical' equation to you. Let's explore it.

When we initialize $\mathbf P$ with

$$\mathbf P = \begin{bmatrix}\sigma^2_x & 0 \\ 0 & \sigma^2_v\end{bmatrix}$$


the value for $\mathbf{FPF}^\mathsf T$ is:

$$\begin{aligned}
\mathbf{FPF}^\mathsf T &= \begin{bmatrix}1&\Delta t\\0&1\end{bmatrix}
\begin{bmatrix}\sigma^2_x & 0 \\  0 & \sigma^2_{v}\end{bmatrix}
\begin{bmatrix}1&0\\\Delta t&1\end{bmatrix} \\
&= \begin{bmatrix}\sigma^2_x&\sigma_v^2\Delta t\\  0 & \sigma^2_{v}\end{bmatrix}
\begin{bmatrix}1&0\\\Delta t&1\end{bmatrix} \\
&= \begin{bmatrix}\sigma^2_x +  \sigma_v^2\Delta t^2  &  \sigma_v^2\Delta t \\
\sigma_v^2\Delta t & \sigma^2_{v}\end{bmatrix}
\end{aligned}$$
- $\mathbf{FPF}^\mathsf T$ uses the process model to automatically compute the covariance between the position and velocity!
- what if $x$ is not correlated to $\dot x$? (set F01 to 0, the rest at defaults)
- what if $x = 2\dot x\Delta t + x_0$? (set F01 to 2, the rest at defaults)
- what if $x = \dot x\Delta t + 2x_0$? (set F00 to 2, the rest at defaults)
- what if $x = \dot x\Delta t$?  (set F00 to 0, the rest at defaults)

#### Update Equations
- Same process: calc the residual, calc the weight of the residual, and add to prior
- System Uncertainty
    - $\textbf{S} = \mathbf{H\bar PH}^\mathsf T + \mathbf R$
    - H is the measurement function
    - P w/ bar is the prior covariance
    - In equation equation P is put into a different space with either the function H or F. Then we add the noise matrix associated with that space
- Kalman Gain
$\mathbf K = \mathbf{\bar PH}^\mathsf T \mathbf{S}^{-1}$
    - I is the identiy matrix (represent 1 in multiple dimensions)
    - H is measurement function (constant)
    - K is the weight of the prediction/measurement

## Summary
$$
\begin{aligned}
\text{Predict Step}\\
\mathbf{\bar x} &= \mathbf{F x} + \mathbf{B u} \\
\mathbf{\bar P} &= \mathbf{FP{F}}^\mathsf T + \mathbf Q \\
\\
\text{Update Step}\\
\textbf{S} &= \mathbf{H\bar PH}^\mathsf T + \mathbf R \\
\mathbf K &= \mathbf{\bar PH}^\mathsf T \mathbf{S}^{-1} \\
\textbf{y} &= \mathbf z - \mathbf{H \bar x} \\
\mathbf x &=\mathbf{\bar x} +\mathbf{K\textbf{y}} \\
\mathbf P &= (\mathbf{I}-\mathbf{KH})\mathbf{\bar P}
\end{aligned}
$$

I want to share a form of the equations that you will see in the literature. There are many different notation systems used, but this gives you an idea of what to expect.

 $$
\begin{aligned}
\hat{\mathbf x}_{k\mid k-1} &= \mathbf F_k\hat{\mathbf x}_{k-1\mid k-1} + \mathbf B_k \mathbf u_k  \\
\mathbf P_{k\mid k-1} &=  \mathbf F_k \mathbf P_{k-1\mid k-1} \mathbf F_k^\mathsf T + \mathbf Q_k \\        	
\tilde{\mathbf y}_k &= \mathbf z_k - \mathbf H_k\hat{\mathbf x}_{k\mid k-1}\\
\mathbf{S}_k &= \mathbf H_k \mathbf P_{k\mid k-1} \mathbf H_k^\mathsf T + \mathbf R_k \\
\mathbf K_k &= \mathbf P_{k\mid k-1}\mathbf H_k^\mathsf T \mathbf{S}_k^{-1}\\
\hat{\mathbf x}_{k\mid k} &= \hat{\mathbf x}_{k\mid k-1} + \mathbf K_k\tilde{\mathbf y}_k\\
\mathbf P_{k|k} &= (I - \mathbf K_k \mathbf H_k) \mathbf P_{k|k-1}
\\\end{aligned}
$$

This notation uses the Bayesian $a\mid b$ notation, which means $a$ given the evidence of $b$. The hat means estimate. Thus $\hat{\mathbf x}_{k\mid k}$ means the estimate of the state $\mathbf x$ at step $k$ (the first $k$) given the evidence from step $k$ (the second $k$). The posterior, in other words. $\hat{\mathbf x}_{k\mid k-1}$ means the estimate for the state $\mathbf x$ at step $k$ given the estimate from step $k - 1$. The prior, in other words. 
