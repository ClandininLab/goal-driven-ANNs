"""
Helper methods for visualization in IPython notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpcol


def insert_discontinuity(sequence, threshold):
    """
    This helper method for plot_sequence inserts discontinuities in the
    given sequence where the absolute difference between subsequent values
    exceeds some maximum value.

    Args:
        sequence: Sequence to be plotted.
        theshold: Minimum absolute difference between successive values in
                  `sequence` to insert discontinuity.

    Return:
        formatted: Sequence with discontinuities inserted at appropriate locations.
    """
    if threshold is None:
        return sequence
    discontinuity = np.where(np.abs(np.diff(position)) >= threshold)[0] + 1
    formatted = np.insert(sequence, discontinuity, np.nan)
    return formatted


def plot_sequence(ground_truth, prediction=None, mode=None,
                  discontinuity_threshold=None):
    """
    Given a sequence of ground truth and/or model outputs, this helper method
    visualizes it.

    Args:
        ground_truth: Target output sequence for network (solid line in plot)
                      of shape [batch_size, sequence_length, output_dim].
        prediction: Output sequence produced by network (dashed line in plot)
                    of shape [batch_size, sequence_length, output_dim].
        mode: Format of output ('sine' for models that output sine/cosine of angle vs.
                                'radians' for models that output angle in radians)
        discontinuity_treshold: threshold for inserting discontinuities in plot of
                                sequence (purely aesthetic, unnecessary if mode is specified)
    """
    batch_size, sequence_length, _ = ground_truth.shape
    cmap = plt.get_cmap(name='hsv', lut=batch_size)
    if mode == 'radians':
        axis = [0, sequence_length - 1, 0, 2 * np.pi]
        discontinuity_threshold = 3 * np.pi / 2
    elif mode == 'sine':
        axis = [0, sequence_length - 1, -1, 1]
    else:
        axis = [0, sequence_length - 1, np.min(ground_truth) - 1, np.max(ground_truth) + 1]
    
    for i in range(batch_size):
        y_true = insert_discontinuity(ground_truth[i, :, 0], discontinuity_threshold)
        plt.plot(y_true, color=cmap(i), alpha=.8)
        if prediction is not None:
            y_pred = insert_discontinuity(prediction[i, :, 0], discontinuity_threshold)
            plt.plot(y_pred, color=cmap(i), alpha=.8, linestyle='dashed')

    plt.xlabel('time step')
    plt.ylabel('position')
    plt.axis(axis)
    plt.show()


def visualize_recurrent(matrix, seperators=[]):
    """
    Method to visualize the recurrent weight matrix of an RNN.

    Args:
        matrix: Recurrent weight matrix in the form of a numpy array.
        seperators: Locations to insert black dotted lines in the plot to
                    visually seperate neuronal subpopulations of interest.
    """
    plt.set_cmap('bwr')
    figure = plt.figure() 
    axes = figure.add_subplot(111) 
      
    caxes = axes.matshow(matrix)
    caxes.set_norm(mpcol.Normalize(-.75, .75, clip=True))   # +/- .75 chosen arbitrarily
    figure.colorbar(caxes)

    hidden_dim = matrix.shape[0]
    for seperator in seperators:
        axes.hlines(seperator, 0, hidden_dim - 1, linestyles='dotted')
        axes.vlines(seperator, 0, hidden_dim - 1, linestyles='dotted')

    plt.xlabel('Presynaptic ID')
    plt.ylabel('Postsynaptic ID')
    plt.show() 


def plot_loss(loss, title='Training Loss', ylabel='Loss'):
    """
    Plot training/validation loss of a model as a function of epoch.

    Args:
        loss: List of model losses where `loss[i]` is the loss for epoch i.
        title: Plot title.
        ylabel: Plot y-axis label.
    """
    plt.plot(loss)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.show()

