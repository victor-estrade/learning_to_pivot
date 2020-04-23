# Reproducing "Learning to Pivot" paper with PyTorch

Trying to reproduce the results from the "Learning to Pivot" paper : https://arxiv.org/abs/1611.01046


# Authors Code

The authors code using Keras can be found here :
https://github.com/glouppe/paper-learning-to-pivot/


# Usage

## Run scripts

To run an experiment from root dir :

- `python -m run.classifier` runs a neural network classifier without adversarial training
- `python -m run.pivot` runs the same neural net with adversarial training
- `python -m run.gradient_boost` runs sklearn's GradientBoostingClassifier

Results are saved in the `output` directory

## Extract pivot code

`pivot.py` contains the training algorithm. It depends on `minibatch.py` to split given data.


# Notes

## Mixture density networks

Using a mixture density network with 5 gaussians as adversarials leads to NaN during training.
Did not figured out why/how.

A simple Mean square error regressor is used instead for stability.

## Hyper paramters

Hyper parameters are different from the original paper but are chosen to give similar plot.

