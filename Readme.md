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


## Extract pivot code

`pivot.py` contains the training algorithm. It depends on `minibatch.py` to split given data.


# TODO

- Use a mixture density network with 5 gaussians like in the notebook :
	https://github.com/glouppe/paper-learning-to-pivot/blob/master/code/Toy.ipynb
- Use same hyper-parameters as in the paper
