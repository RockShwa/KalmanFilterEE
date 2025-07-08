# Multivariate Gaussians
- The goal: model multidimensional data

## Multivariate Normal Distrivutions
- For the 2D scenario: N dimensions, so we need N means
$$
\mu = \begin{bmatrix}\mu_1\\\mu_2\\ \vdots \\\mu_n\end{bmatrix}
$$
- Let's say we believe that $x = 2$ and $y = 17$. We would have
$$
\mu = \begin{bmatrix}2\\17\end{bmatrix} 
$$
- Covariance descrives how much two variables vary together (if two variables are correlated, the have a covariance)
    - If we're trying to model height and weight, we're going to want to also know how much those are correlated: how the weight varies compared to the heights
    - Assuming linear correlation
- The equation for the covariance between $X$ and $Y$ is

$$ COV(X, Y) = \sigma_{xy} = \mathbb E\big[(X-\mu_x)(Y-\mu_y)\big]$$

Where $\mathbb E[X]$ is the *expected value* of X, defined as

$$\mathbb E[X] =  \begin{cases} \sum_{i=1}^n p_ix_i & \mbox{discrete}\\ \int_{-\infty}^\infty f(x)\, x dx & \mbox{continuous}\end{cases}$$

We assume each data point is equally likely, so the probability of each is $\frac{1}{N}$, giving

$$\mathbb E[X] =  \frac{1}{N}\sum_{i=1}^n x_i$$

for the discrete case we will be considering.
- We use a *covariance matrix* to denote covariances of a multivariate normal distribution, and it looks like this:
$$
\Sigma = \begin{bmatrix}
  \sigma_1^2 & \sigma_{12} & \cdots & \sigma_{1n} \\
  \sigma_{21} &\sigma_2^2 & \cdots & \sigma_{2n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  \sigma_{n1} & \sigma_{n2} & \cdots & \sigma_n^2
 \end{bmatrix}
$$
- The diagonal contains the variance for each variable, and the off-diagonal elements contain the covariance between the $i^{th}$ and $j^{th}$ variables. So $\sigma_3^2$ is the variance of the third variable, and $\sigma_{13}$ is the covariance between the first and third variables.
- Covariance of 0 indicates no correlation. If the variance for $x$ is 10, the variance for $y$ is 4, and there is no linear correlation between $x$ and $y$, then we would write

$$\Sigma = \begin{bmatrix}10&0\\0&4\end{bmatrix}$$

- A "small" covariance is small relative to the variances in the matricies
- The covariance between x and y is always equal to the covariance between y and x
- Covariance: when one var increases so does the other, when one decreases the other increases -> how much?

# Multivariate Normal Distribution Equation
- Here is the multivariate normal distribution in $n$ dimensions.

$$
f(\mathbf{x},\, \mu,\,\Sigma) = \frac{1}{\sqrt{(2\pi)^n|\Sigma|}}\, \exp  \Big [{ -\frac{1}{2}(\mathbf{x}-\mu)^\mathsf{T}\Sigma^{-1}(\mathbf{x}-\mu) \Big ]}
$$

- The multivariate version merely replaces the scalars of the univariate equations with matrices. If you are reasonably well-versed in linear algebra this equation should look quite manageable. 
- Kalman filters use the name $\textbf{P}$ for the covariance matrix
- Joint probability (basically just and): $P(x,y)$, is the probability of both $x$ and $y$ happening
- Marginal probability is the probability of an event happening without regard of any other event (marginal of a multivariate gaussian is a gaussian)
- Covariance: if we know x, we can predict something about y because they're correlated
- Independent variables are always uncorrelated
- Radars are correlated (range and bearing)

# Hidden Variables
- Hidden variables: not observed by sensors but we can accurately infer its value since its correlated with an observed variable