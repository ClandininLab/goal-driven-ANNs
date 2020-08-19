import numpy as np
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.utils import to_categorical

import os
import sys
parent_directory = os.path.dirname(os.getcwd())
sys.path.append(parent_directory)
from parent_model import RNN
import util


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
        super(HDRNN, self).__init__(input_dim=1, recurrent_constraint=ZeroDiagonal(), **kwargs,
                                    loss_fun={"class_name": "CategoricalCrossentropy",
                                              "config": {"from_logits": True} })


    def loss(self, target, input_sequences, start_states=None):
        discretized_target = discretize(target, self.output_dim)
        return super(HDRNN, self).loss(discretized_target, input_sequences, start_states)


############################## HELPER METHODS ##############################

def trig(sequence):
    """
    Given an array of shape (..., 1), returns an array of shape (..., 2) whose
    stacks contain the sine and cosine of the original array, respectively.
    """
    return np.concatenate([np.sin(sequence), np.cos(sequence)], axis=-1)


def discretize(sequence, precision, min_val=0, max_val=2 * np.pi):
    """
    Given an array of values between min_val and max_val of shape (..., 1),
    returns an array of shape (..., precision) wherein each feature in the
    -1th dimension is a discretized one-hot encoding of the corresponsing value
    in the original array.
    """
    normed = (sequence + min_value) / max_value * precision
    return to_categorical(normed, precision)

