import numpy as np
import matplotlib.pyplot as plt
import pickle


def save_model(model, file_name):
    """
    Saves a model using pickel.
    """
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)


def load_model(file_name):
    """
    Loads a saved model using pickel.
    """
    model = None
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
    return model


def insert_discontinuity(sequence, threshold):
    """
    For plotting purposes, inserts discontinuities in the given sequence
    where the absolute difference between subsequent values exceeds some
    maximum value.
    """
    if threshold is None:
        return sequence
    discontinuity = np.where(np.abs(np.diff(position)) >= threshold)[0] + 1
    formatted = np.insert(sequence, discontinuity, np.nan)
    return formatted


def plot_sequence(ground_truth, prediction=None, mode=None,
                  discontinuity_threshold=None):
    """
    Given a sequence of shape (batch_size, sequence_length, feature_dim),
    plots each sequence in the batch. Only the 0 index value of each feature
    is plotted (i.e. we plot sequence[i, :, 0])
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


def visualize_recurrent(model):
    """
    Given an RNN model, displays its recurrent weight matrix.
    """
    W_recurrent = model.rnn.weights[1].numpy()
    plt.matshow(W_recurrent)


def plot_training_loss(trainer):
    plt.plot(trainer.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

