# Designing Kalman Filters
- See Airplane Sensor for code

## Choose the State Variables
- Traking in two dimensions, observed variables x and y
- Still want to incorperate velocity so we can have additional positional data:
$$\mathbf x = 
\begin{bmatrix}x & \dot x & y & \dot y\end{bmatrix}^\mathsf T$$

## Design State Transition Function
- State transition is matrix F that we multiply with the previous state of our system to get the next state
$$\mathbf{\bar x} = \mathbf{Fx}$$
- The state transition equations are

$$
\begin{aligned}
x &= 1x + \Delta t \dot x + 0y + 0 \dot y \\
v_x &= 0x + 1\dot x + 0y + 0 \dot y \\
y &= 0x + 0\dot x + 1y + \Delta t \dot y \\
v_y &= 0x + 0\dot x + 0y + 1 \dot y
\end{aligned}
$$

Laying it out that way shows us both the values and row-column organization required for $\small\mathbf F$. We convert this to matrix-vector form:

$$
\begin{bmatrix}x \\ \dot x \\ y \\ \dot y\end{bmatrix} = \begin{bmatrix}1& \Delta t& 0& 0\\0& 1& 0& 0\\0& 0& 1& \Delta t\\ 0& 0& 0& 1\end{bmatrix}\begin{bmatrix}x \\ \dot x \\ y \\ \dot y\end{bmatrix}$$

## Design the Process Noise Matrix
- Assume noise is constant for each time period: allows us to use a variance to specify how much the model should change between steps

## Design the Measurement Function
- Measurement function H defines how we go from state variables to measurements using the equation z = Hx
- In this case we have measurements for (x,y), so we will design $\mathbf z$ as $\begin{bmatrix}x & y\end{bmatrix}^\mathsf T$ which is dimension 2x1.
- $$(2\times 1) = (a\times b)(4 \times 1) = (2\times 4)(4\times 1)$$

So, $\textbf{H}$ is 2x4.
- If we want to convert to meters, we put that in the measurement function

## Design the Measurement Noise Matrix
- x and y are independent Gaussian processes (white noise)
- The noise in x is not in any way dependent on the noise in y
$$\mathbf R = \begin{bmatrix}\sigma_x^2 & \sigma_y\sigma_x \\ \sigma_x\sigma_y & \sigma_{y}^2\end{bmatrix} 
= \begin{bmatrix}5&0\\0&5\end{bmatrix}$$
- It is a $2{\times}2$ matrix because we have 2 sensor inputs, and covariance matrices are always of size $n{\times}n$ for $n$ variables.

## Initial Conditions
- Init: (0,0) with init velo of (0,0)
- Since this is a large guess: covariance matrix P to a large value
$$ \mathbf x = \begin{bmatrix}0\\0\\0\\0\end{bmatrix}, \,
\mathbf P = \begin{bmatrix}500&0&0&0\\0&500&0&0\\0&0&500&0\\0&0&0&500\end{bmatrix}$$

## Control Inputs
- u is the control input, B is a matrix that transforms the control input into a change in the state (x)
- Suppose the state is $x = \begin{bmatrix} x & \dot x\end{bmatrix}$ for a robot we are controlling and the control input is commanded velocity. This gives us a control input of 

$$\mathbf{u} = \begin{bmatrix}\dot x_\mathtt{cmd}\end{bmatrix}$$

For simplicity we will assume that the robot can respond instantly to changes to this input. That means that the new position and velocity after $\Delta t$ seconds will be

$$\begin{aligned}x &= x + \dot x_\mathtt{cmd} \Delta t \\
\dot x &= \dot x_\mathtt{cmd}\end{aligned}$$

We need to represent this set of equations in the form $\bar{\mathbf x} = \mathbf{Fx} + \mathbf{Bu}$.

## Sensor Fusion
- Make z multi dimensional!
- Easy to fuse sensors measuring the same variable
