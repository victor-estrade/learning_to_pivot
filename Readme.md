# Reproducing "Learning to Pivot" paper with PyTorch

Trying to reproduce the results from the "Learning to Pivot" paper : https://arxiv.org/abs/1611.01046


# Authors Code

The authors code using Keras can be found here :
https://github.com/glouppe/paper-learning-to-pivot/


# Usage

## Run scripts

To run an experiment from root dir :

- `python -m run.classifier` runs a neural network classifier without adversarial training
- `python -m run.pivot` runs the same neural net with adversarial training againts a mean squared error regressor
- `python -m run.pivot_mdn` runs the same neural net with adversarial training againts a 5 gaussian mixture density network regressor
- `python -m run.gradient_boost` runs sklearn's GradientBoostingClassifier (easy to use and to train)

Results are saved in the `output` directory which is automatically created.

## Extract pivot code

`pivot.py` contains the training algorithm. 
It depends on `minibatch.py` to split given data and on `utils.py` to convert to/from numpy arrays to/from torch tensors.


# Notes

## Mixture density networks

Using Adam optimizer with mixture density network requires to reduce beta values to reduce its inertia.

## Mean squared error regressor

A simpler Mean square error regressor is also available.

## Hyper paramters

Hyper parameters are different from the original paper but are chosen to give almost similar plot.

Do not hesitate to play with it.

## Seed

Scripts are seeded to help reproduce bugs or unwanted behaviour (especially NaN values).

Do not hesitate to change the seeds although each scripts already runs multiple seeds.
