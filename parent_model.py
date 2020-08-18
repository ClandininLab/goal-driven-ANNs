# Based on ganguli-lab/grid-pattern-formation repo

import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN, Activation
from tensorflow.keras.models import Model
import util
import pickle


class RNN(Model):
    def __init__(self, input_dim, hidden_dim, output_dim, sequence_length,
                 activation='tanh', loss_fun='MSE', weight_decay=0,
                 metabolic_cost=0, use_bias=True, recurrent_constraint=None):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.activation = activation

        self.encoder = Dense(hidden_dim, name='encoder', use_bias=use_bias)
        self.rnn = SimpleRNN(hidden_dim, 
                             return_sequences=True,
                             activation=tf.keras.layers.Activation(activation),
                             recurrent_initializer='glorot_uniform',
                             recurrent_constraint=recurrent_constraint,
                             name='RNN',
                             use_bias=use_bias)
        self.decoder = Dense(output_dim, name='decoder', use_bias=use_bias)

        # Loss function
        self.loss_fun = tf.keras.losses.get(loss_fun)
        self.weight_decay = weight_decay
        self.metabolic_cost = metabolic_cost
    

    def g(self, input_sequences, start_states=None):
        '''
        Compute hidden unit activations.

        Args:
            input_sequences: Batch of input sequences with shape [batch_size, sequence_length, 2].
            start_states: Batch of start states with shape [batch_size, feature_dim]

        Returns: 
            activation_history: Batch of hidden unit activations with shape [batch_size, sequence_length, output_dim].
        '''
        init_activations = None
        if start_states is not None:
            init_activations = self.encoder(start_states)
        else:
            batch_size = input_sequences.shape[0]
            init_activations = tf.zeros([batch_size, self.hidden_dim])
        activation_history = self.rnn(input_sequences, initial_state=init_activations)
        return activation_history
    

    def call(self, input_sequences, start_states=None, return_activation_history=False):
        '''
        Predict place cell code.

        Args:
            input_sequences: Batch of input sequences with shape [batch_size, sequence_length, 2].
            start_states: Batch of start states with shape [batch_size, feature_dim]

        Returns: 
            output_sequences: Predicted output sequence with shape [batch_size, sequence_length, output_dim].
        '''
        activation_history = self.g(input_sequences, start_states)
        output_sequences = self.decoder(activation_history)
        if return_activation_history:
            return output_sequences, activation_history
        else:
            return output_sequences


    def loss(self, targets, input_sequences, start_states=None):
        '''
        Compute avg. loss.

        Args:

        Returns:
        '''
        preds, activation_history = self.call(input_sequences, start_states,
                                              return_activation_history=True)
        loss = tf.reduce_mean(self.loss_fun(targets, preds))

        # L2 weight regularization 
        loss += self.weight_decay * tf.reduce_sum(self.rnn.weights[1]**2)
        preds += self.metabolic_cost * tf.reduce_sum(activation_history**2)

        return loss

