# Kalman Filter Research Paper

## Project Description
This repository contains my Kalman Filter algorithm adapted by Labbe R. R.'s Kalman and Bayesian Filters in Python. This project also contains notes from his textbook.

## Research Question
How can a Multivariate Kalman Filter be used to fuse kinematic data in order to more accurately estimate the relative position of an airplane?

## Run and Build Instructions

To run the main program:
```sh
python FusionControlAirplaneFilter.py
```
The expected output is a graph that represented the output of a Kalman Filter using sensor fusion with simulated GPS and INS data. The residuals are primarily between the acceptable output range, marked by the yellow area. Furthermore the idea and actual center residual line should be very close together, which represents that the Kalman Filter's predicted output matches the expected output.
