# Kalman Filter Math

## Modeling a Dynamic System
- A dynamic system is a physical system whose state (pos, temp, etc.) evolves over time (use differential equations to model dynamic systems)
- Each physical system has a process
    - A car traveling at a certain velocity goes so far in a fixed amount of time
    - Velocity varies as a function of its acceleration
    - $v = at$
    - $x = 0.5at^2+v_0t+x_0$

- Model of the system w/o noise:

$$ \dot{\mathbf x} = \mathbf{Ax}$$

- $\mathbf A$ is known as the *systems dynamics matrix* as it describes the dynamics of the system. Now we need to model the noise. We will call that $\mathbf w$, and add it to the equation. 

$$ \dot{\mathbf x} = \mathbf{Ax} + \mathbf w$$

- Control inputs (u), matrix B to convert u into the effect on the system (pressing the accelerator in you car makes you accelerate)
- Final equation:
$$ \dot{\mathbf x} = \mathbf{Ax} + \mathbf{Bu} + \mathbf{w}$$

### State-Space Representation of Dynamic Systems
- Not interested in derivative of x, but x itself
- Ignoring the noise for a moment, we want an equation that recursively finds the value of $\mathbf x$ at time $t_k$ in terms of $\mathbf x$ at time $t_{k-1}$:

$$\mathbf x(t_k) = \mathbf F(\Delta t)\mathbf x(t_{k-1}) + \mathbf B(t_k)\mathbf u (t_k)$$

Convention allows us to write $\mathbf x(t_k)$ as $\mathbf x_k$, which means the 
the value of $\mathbf x$ at the $k^{th}$ value of $t$.
- $\mathbf F$ is the familiar *state transition matrix*, named due to its ability to transition the state's value between discrete time steps. It is very similar to the system dynamics matrix $\mathbf A$. The difference is that $\mathbf A$ models a set of linear differential equations, and is continuous. $\mathbf F$ is discrete, and represents a set of linear equations (not differential equations) which transitions $\mathbf x_{k-1}$ to $\mathbf x_k$ over a discrete time step $\Delta t$.
- Finding this matrix is often quite difficult. The equation $\dot x = v$ is the simplest possible differential equation and we trivially integrate it as:

$$ \int\limits_{x_{k-1}}^{x_k}  \mathrm{d}x = \int\limits_{0}^{\Delta t} v\, \mathrm{d}t $$
$$x_k-x_{k-1} = v \Delta t$$
$$x_k = v \Delta t + x_{k-1}$$

- This equation is *recursive*: we compute the value of $x$ at time $k$ based on its value at time $k-1$. This recursive form enables us to represent the system (process model) in the form required by the Kalman filter:

$$\begin{aligned}
\mathbf x_k &= \mathbf{Fx}_{k-1}  \\
&= \begin{bmatrix} 1 & \Delta t \\ 0 & 1\end{bmatrix}
\begin{bmatrix}x_{k-1} \\ \dot x_{k-1}\end{bmatrix}
\end{aligned}$$

- We can do that only because $\dot x = v$ is simplest differential equation possible. Almost all other in physical systems result in more complicated differential equation which do not yield to this approach. 

### Forming First Order Equations from Higher Order Equations
- Control input: many physical systems require second or higher order differential equations
- Any higher order system of equations can be reduced to first order by definig extra variables for the derivatives and then solving 
- Given the system $\ddot{x} - 6\dot x + 9x = u$ find the equivalent first order equations. I've used the dot notation for the time derivatives for clarity.
- We define two new variables:

$$\begin{aligned} x_1(t) &= x \\
x_2(t) &= \dot x
\end{aligned}$$

Now we will substitute these into the original equation and solve. The solution yields a set of first-order equations in terms of these new variables. It is conventional to drop the $(t)$ for notational convenience.

We know that $\dot x_1 = x_2$ and that $\dot x_2 = \ddot{x}$. Therefore

$$\begin{aligned}
\dot x_2 &= \ddot{x} \\
         &= 6\dot x - 9x + u\\
         &= 6x_2-9x_1 + u
\end{aligned}$$

Therefore our first-order system of equations is

$$\begin{aligned}\dot x_1 &= x_2 \\
\dot x_2 &= 6x_2-9x_1 + u\end{aligned}$$

If you practice this a bit you will become adept at it. Isolate the highest term, define a new variable and its derivatives, and then substitute.

### Design the Process Noise Matrix
- Q is too small: filter will be overconfident in its prediction model and will diverge from the actual solution
- White noise could be that acceleration is actually not constant
- This is smth I can tune I believe
    - you ca nalso set Q to 0 except for a noise term in the lower rightmost element (usually close to variance (standard dev squared))
    - lower rightmost element is the most rapidly changing term for each variable
- If the state is $x=\begin{bmatrix}x & \dot x & \ddot{x} & y & \dot{y} & \ddot{y}\end{bmatrix}^\mathsf{T}$ Then $\mathbf Q$ will be 6x6; the elements for both $\ddot{x}$ and $\ddot{y}$ will have to be set to non-zero in $\mathbf Q$.