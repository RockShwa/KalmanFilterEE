# Discrete Bayes Filter
- Tracking a Dog
    - Doggo has sonar sensor that emits a signal and listens for an echo
- In Bayesian stats the **prior** or prediction is the probability prior to incorporating measurements or other info (prior prob dist)
- Probability distributions are a collection of all possible probabilites for an event
- Belief in this context is the measure of the strength/certainity of our knowledge
- Categorial distribution is a discrete distribution describing the probability of observing n outcomes

## Noisy Sensors
- Normalize the data: divide each element by sum of all elements in the list (so the probabilites correctly sum to one)
- calculate the scale: z_prob / (1. - z_prob)
- This will scale the measurement to add up to 1
- Posterior probability distribution is the prob dist after incorporating the measurement information
- Likelihood: compute how likely each position was given the measurement (not prob dist bc it doesnt sum to one)
- posterior/estimated state = (likelihood * prior) / normalization

## Terminology
- Bar over a variable = predictions
- f<sub>x</sub>(.) is the state propagation for the function x, describin how much x<sub>k</sub> changes over one time step
- f<sub>x</sub>(v<sub>x</sub>, t) = 15 * 2 (15 being the velocity and 2 being the time step) = v<sub>k</sub> * t
- The error in the model is called **system or process error**

## Adding Uncertainty to the Prediction
~~~
belief = [0, 0, .4, .6, 0, 0, 0, 0, 0, 0]
prior = predict_move(belief, 2, .1, .8, .1)
array([0.  , 0.  , 0.  , 0.04, 0.38, 0.52, 0.06, 0.  , 0.  , 0.  ])
~~~
- Basically, if we're not 100% sure where the dog started:
- We know the 0.04 is the 0.4 belief undershot by 1 (it can't be overshoot because it would be beyond its current position)
- We know that if we started at space 3, there's an 80% chance we moved 2 spaces (0.4 * 0.8) but there's also the 10% chance that we undershot if we didn't start at space 3 (so 0.6 * 0.4)
- You can't start at a belief of 0 (which is why none of these can be overshoots, the only places the dog could be is space 3 and 4)

## Generalizing with Convolution
- Convolution modifies one function with another function: this allows us to generalize the algorithm to accept error with two, three, or more positions (instead of just the one)
    - In our case we are modifying a probability distribution with the error function of the sensor
- Basically you multiply the neighbors of a current array cell with the values of the second array. You shift one function across the other, multiplying the overlapping portions at each shift, and then integrate the result (or sum the result if doing discrete functions)
- Now that we're using probabilites, the previous equation that combined the change in position * dt + init pos is now the convolution of the probability of the state at a time step and the state propogation (change over time step)

## The Discrete Bates Algorithm
- A form of the g-h filter, but we use the percentages for the errors to implicity compute the g and h parameters
- Equations:
$$
Predict Step: \bar{x} = x * f_{x}(\bullet)
$$
$$
Update Step: x = ||L * \bar{x}||
$$
- L is the likelihood function, || || denotes taking the norm (normalizing it so it x is a prob dist that sums to one)
- The algorithm:
Initialization

1. Initialize our belief in the state

Predict

1. Based on the system behavior, predict state for the next time step
2. Adjust belief to account for the uncertainty in prediction

Update

1. Get a measurement and associated belief about its accuracy
2. Compute how likely it is the measurement matches each state
3. Update state belief with this likelihood

~~~ py
def discrete_bayes_sim(prior, kernel, measurements, z_prob, hallway):
    # create an array of 10 spaces, with equal probabilites of the dog being there 
    posterior = np.array([.1]*10)
    priors, posteriors = [], []
    # holds index and measurement as it iterates over each
    for i, z in enumerate(measurements):
        # convoludes the posterior with the kernel to predict where the dog is most likely, so in this instance we shift 1 space
        # Use posterior bc it contains all previous measurements/predictions
        prior = predict(posterior, 1, kernel)
        # add to array
        priors.append(prior)

        # compute the likelihood that a measurement matches positions in the hallway (normalizes the distribution too)
        likelihood = lh_hallway(hallway, z, z_prob)
        # normalizes likelihood * prior (likelihood of prediction)
        posterior = update(likelihood, prior)
        # add to array!
        posteriors.append(posterior)
    return priors, posteriors
~~~

## Drawbacks and Limitations
- This filter is best for when you need a multimodal, discrete filter
- First problem is scaling, it's not multidimensional (or if it is, the big O is crazy :sk:)
- The filter is discrete, but we live in a continuous world, so to get LOTS of data/probabilites, the computational power is chopped

# Summary
- THIS is the crucial chapter for understanding the Kalman Filter
- Fundamental Insight - Multiplying probabilities when we measure, and shifting probabilities when we update leads to a converging solution 