"""
RNN child classes for angular velocity/heading integration.
See parent class in ../parent_model.py for more details.
"""

import numpy as np
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.utils import to_categorical

import os
import sys
parent_directory = os.path.dirname(os.getcwd())
sys.path.append(parent_directory)
from parent_model import RNN


class ZeroDiagonal(Constraint):
    """
    Model constraint that prevents self connections by constraining diagonal
    entries of recurrent weight matrix to equal 0.
    """
    def __call__(self, w):
        dim = w.shape[0]
        return w * (np.identity(dim) == 0)


class HDRNN(RNN):
    """
    Maps start position/angular velocity sequences to sine/cosine of
    heading direction at each time step, minimizing mean squared error.
    """
    def __init__(self, **kwargs):
        super(HDRNN, self).__init__(input_dim=1, output_dim=2, **kwargs,
                                    recurrent_constraint=ZeroDiagonal())


    def loss(self, target, input_sequences, start_states=None):
        return super(HDRNN, self).loss(trig(target), input_sequences, start_states)


class SimpleHDRNN(RNN):
    """
    Maps start position/angular velocity sequences to heading direction
    (angle in radians) at each time step, minimizing mean squared error.
    """
    def __init__(self, **kwargs):
        super(SimpleHDRNN, self).__init__(input_dim=1, output_dim=1, **kwargs)


    def loss(self, target, input_sequences, start_states=None):
        return super(SimpleHDRNN, self).loss(target, input_sequences, start_states)


class ClassifierHDRNN(RNN):
    """
    Maps start position/angular velocity sequences to 'sector' of heading
    direction at each time step, minimizing cross entropy.
    """
    def __init__(self, **kwargs):
        super(ClassifierHDRNN, self).__init__(input_dim=1, **kwargs,
			                                  loss_fun={"class_name": "CategoricalCrossentropy",
			                                            "config": {"from_logits": True}})


    def loss(self, target, input_sequences, start_states=None):
        discretized_target = discretize(target, self.output_dim)
        return super(ClassifierHDRNN, self).loss(discretized_target, input_sequences, start_states)


############################## HELPER METHODS ##############################

def trig(matrix):
    """
    Given an array of angles in radians, this helper method for HDRNN.
    converts each angle to its [sin, cos].

    Args:
        matrix: Array of angles in radians of shape [..., 1].

    Return:
        Array of shape [..., 2] wherein each angle in matrix has been replaced
        by its sin and cos.
    """
    return np.concatenate([np.sin(matrix), np.cos(matrix)], axis=-1)


def discretize(matrix, num_bins, min_value=0, max_value=2 * np.pi):
    """
    Given an array of values between min_val and max_val of shape [..., 1],
    this helper method for ClassifierHDRNN returns an array of shape [..., num_bins]
    wherein the -1th dimension of each feature is a discretized one-hot encoding of the
    corresponsing value in the -1th dimension original array.

    Args:
        matrix: Array of values between min_val and max_val of shape [..., 1].
        num_bins: Number of discrete bins in which to map continuous values of `matrix`.
        min_value: Value corresponding to lower bound of first bin.
        max_value: Value corresponding to upper bound of last bin.

    Return:
        one-hot encoding as described above.
    """
    normed = (matrix + min_value) / max_value * num_bins    
    return to_categorical(normed, num_bins)

