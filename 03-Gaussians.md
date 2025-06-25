# Probabilities, Gaussians, and Bayes' Theorem
- When you think about what filter you want, you have to think about whether it's: continuous or discrete and unimodal or multimodal
- Naviagation is continuous (think normal distributions vs binomial, binomial only uses whole numbers) and unimodal: we only want it to predict ONE location the aircraft could be, not two

## Mean, Variance, and Standard Deviations
- Capital letters = random variables; lower case = vectors; upper case = matricies
- Capital P refers to a single event, while lower case p refers to the probability distribution function
- To be a probability distribution the probability of each value $x_{i}$ must be $x_{i} \geq 0$ and the sum of the probabilities for all values must be equal to one
    - Discrete Distributions: $\sum\limits_{u}P(X=u) = 1$
    - Continuous Distributions: $ \int_{u}P(X=u)du = 1$

## Expected Value of a Random Variable
- The expected value of a random variable is the average value it would have if we took an infinte number of samples and then averaged those samples together
    - You multiply each x<sub>i</sub> by it's probability of occuring

## Variance of a Random Variable
- Variance = variation from the mean, spread of the data, average squared distance from the mean
- $VAR(X) = E[(X-\mu)^2]$
- Standard deviation is the square root of variance to get it back into the units of the mean

## Gaussians
- Normal distribution = Gaussian distribution = probability density function
- The y-axis represents the probability density, the relative amount of cars that are going the speed at the corresponding x-axis
- Gaussians are not perfect, their tails extend to infinity, which is not possible in the real world
- The Gaussian Function: exp[x] = $e^x$
$$ 
f(x, \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\big [{-\frac{(x-\mu)^2}{2\sigma^2} }\big ]
$$
- The area under the curve between two limits is the probability of the state being between those limits

## The Variance and Belief
- If the variance is small the curve will be narrow; if the variance is small there's a large amount of precision
- Gaussians reflect our belief about a measurement, the express the precision of the measurement, and they express how much variance there is in the measurements

## Computational Properties of Normally Distributed Random Variables
- Sum of two independent Gaussian random variables is also normally distributed, the product isn't Gaussian but it is proportional (it's a Gaussian function, so not necessarily summing to 1, so you have to normalize it)
- Product of two independent Gaussians:
$$\begin{aligned}\mu &=\frac{\sigma_1^2\mu_2 + \sigma_2^2\mu_1}{\sigma_1^2+\sigma_2^2}\\
\sigma^2 &=\frac{\sigma_1^2\sigma_2^2}{\sigma_1^2+\sigma_2^2} 
\end{aligned}$$
- Sum of two Gaussian random variables:
$$\begin{gathered}\mu = \mu_1 + \mu_2 \\
\sigma^2 = \sigma^2_1 + \sigma^2_2
\end{gathered}$$

## Bayes Theorem
- Bayes theorem: Updated Knowledge = ||likelihood of new knowledge * prior knowledge|| (it's also the conditional probability formula)
$$ \mathtt{posterior} = \frac{\mathtt{likelihood}\times \mathtt{prior}}{\mathtt{normalization}}$$
- The probability of A happening if B happened (valid with both single probabilites and probability distributions)
- $$p(x_i \mid z) = \frac{p(z \mid x_i) p(x_i)}{p(z)}$$
    - The above is the probable state at step i given a measurement
    - This is HARD to compute
    - Bayes theorem uses the inverse tho: $$p(x_i \mid Z) \propto p(Z\mid x_i)\, p(x_i)$$
    - The inverse however, $p(z \mid x_i)$ is often much easier to compute
    - Basically instead we're computing the likelihood of the sensor readings given that it's raining instead of the likelihood of it raining given the sensor readings

## Total Probability Theorem
- Predict implements the total probability theorem
    - It computes the robability of being at any given position fiven the probability of all the possible movement events
- The probability of being at any position $i$ at time $t$ can be written as $P(X_i^t)$. We computed that as the sum of the prior at time $t-1$ $P(X_j^{t-1})$ multiplied by the probability of moving from cell $x_j$ to $x_i$. That is

$$P(X_i^t) = \sum_j P(X_j^{t-1})  P(x_i | x_j)$$

## Limitations of Gaussians in the Real World
- Nature is full of distributions that are not normal, but when we apply the central limit theorem over large populations we end up with normal distributions
- Most sensors arent perfectly Gaussian, remember the Kalman Filter is an idealized version of the world
- Kalman Filter equations assume noise is normally distributed as well, and perform sub-optimally if this isn't true