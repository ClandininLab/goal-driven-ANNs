import tensorflow as tf
import numpy as np
import pickle


class Trainer(object):
    def __init__(self, model, trajectory_generator, learning_rate):
        self.model = model
        self.generator = trajectory_generator
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.losses = []


    def train_step(self, start, velocity, position):
        ''' 
        Train on one batch of trajectories.
        Args:
            
        Returns:
        '''
        with tf.GradientTape() as tape:
            loss = self.model.loss(position, velocity, start)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


    def train(self, n_epochs, batch_size=100, print_every=100):
        ''' 
        ...
        Args:
        '''
        for epoch in range(n_epochs):
            batch = self.generator.generate_trajectory(batch_size)
            loss = self.train_step(*batch)
            self.losses.append(loss.numpy())
            if epoch % print_every == 0:
                print('Epoch {}. loss = {}'.format(epoch, loss))

