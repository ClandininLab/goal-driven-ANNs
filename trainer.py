"""
Class for optimization of models.
"""

import tensorflow as tf


class Trainer(object):
    """
    Class to decompose training of the model. 
    """
    def __init__(self, model, trajectory_generator, learning_rate, validation_dataset=None):
        self.model = model
        self.generator = trajectory_generator
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.training_loss = []
        self.validation_dataset = validation_dataset
        self.validation_loss = []


    def train_step(self, start, velocity, position):
        """ 
        Train on one batch of trajectories.
        Args:
            start, velocity, and position for a single batch of inputs.
            
        Returns:
            loss: Average loss for input batch.
        """
        with tf.GradientTape() as tape:
            loss = self.model.loss(position, velocity, start)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


    def train(self, n_epochs, batch_size=100, print_every=100):
        """ 
        Perform gradient-based optimization for desired number of epochs.
        Args:
            n_epochs: Number of batches to train the network on.
            batch_size: Number of examples per batch.
            print_every: Frequency (in terms of epochs) of printing loss.
        """
        for epoch in range(n_epochs):
            batch = self.generator.generate_trajectory(batch_size)
            loss = self.train_step(*batch)
            self.training_loss.append(loss.numpy())
            if epoch % print_every == 0:
                print('Epoch {}. loss = {}'.format(epoch, loss))
            if self.validation_dataset is not None:
                val_loss = self.model.loss(*self.validation_dataset)
                self.validation_loss.append(val_loss.numpy())
        print('{} Epochs Completed. final loss = {}'.format(n_epochs, self.training_loss[-1]))

