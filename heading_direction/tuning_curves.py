"""
Methods for inspecting RNN response properties.
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_tuning(position, activation_history, num_bins):
    """
    Computes the trial averaged activation of each RNN unit as a
    function of the agent's heading direction.

    Args:
        position: Array of shape [batch_size, sequence_length, 1] containing
                  agent heading directions at each time step.
        activation_history: Array of shape [batch_size, sequence_length, hidden_dim]
                            containing unit activations at each time step.
        num_bins: Number of discrete slices to divide angle space into. For example, if
                  num_bins = 2 then 0th bin is [0, pi) and 1th bin is [pi, 2pi).

    Return: 
        tuning: Array of shape [hidden_dim, num_bins] containing trial averaged responses.
    """
    normalized_position = position.flatten() / (2 * np.pi)
    bins = (num_bins * normalized_position).astype(int)

    hidden_dim = activation_history.shape[-1]
    activations = activation_history.reshape((-1, hidden_dim))
    
    counts = np.zeros(num_bins)
    np.add.at(counts, bins, 1)
    counts += (counts == 0)     # avoid division by zero

    totals = np.zeros((num_bins, hidden_dim))
    np.add.at(totals, bins, activations)
    
    tunings = totals.T / counts
    return tunings


def plot_tuning(tuning, title='Trained Network Unit {} Tuning Curve'):
    """
    Plots the tuning curve of every RNN unit in the network.

    Args:
        tuning: Array of shape [hidden_dim, num_bins] containing
                trial-averaged unit responses.
        title: Plot title.
    """
    hidden_dim, num_bins = tuning.shape
    x = [2 * np.pi * i / num_bins for i in range(num_bins)]
    axis = [0, x[-1], 0, 1]
    for i in range(hidden_dim):
        plt.plot(x, tuning[i])
        plt.axis(axis)
        plt.xlabel('Heading Angle (radians)')
        plt.ylabel('Trial Averaged Response')
        plt.title(title.format(i))
        plt.show()
        plt.clf()


def sort_units_by_peaks(weights, tuning):
    """
    Sorts the recurrent weight matrices and tuning matrix according
    to the tuning curve angle of peak activation.

    Args:
        weights: List of weight matrices each of shape [hidden_dim, hidden_dim] to sort.
        tuning: Array of shape [hidden_dim, num_bins] containing 
                trial-averaged unit responses.
    Return:
        sorted_weights: Each matrix in weights sorted by tuning properties.
        sorted_tuning: Tuning curves sorted by their peaks.
        sorted_peaks: Tuning curve peaks in sorted order.
    """
    peaks = np.argmax(tuning, axis=1)
    order = np.argsort(peaks)
    sorted_peaks = peaks[order]
    sorted_weights = []
    for W in weights:
        sorted_weights.append(W[order][:, order])
    sorted_tuning = tuning[order]
    return sorted_weights, sorted_tuning, sorted_peaks

