import numpy as np


class TrajectoryGenerator(object):
    def __init__(self, sequence_length, standard_deviation, momentum,
                 step_size=1, start_pos=None, modulo=True):
        """
        """
        self.sequence_length = sequence_length
        self.standard_deviation = standard_deviation
        self.momentum = momentum
        self.step_size = step_size
        self.start_pos = start_pos
        self.modulo = modulo


    def generate_trajectory(self, batch_size):
        """
        """
        start = None
        if self.start_pos is None:   # randomize start positions
            start = np.random.uniform(0, 2 * np.pi, (batch_size, 1)).astype(np.float32)
        else:
            start = (np.zeros((batch_size, 1)) + self.start_pos).astype(np.float32)

        velocity = np.zeros((batch_size, self.sequence_length, 1), dtype=np.float32)
        for i in range(self.sequence_length):
            velocity[:, i, :] = np.random.normal(0, self.standard_deviation, (batch_size, 1))
            velocity[:, i, :] += self.momentum * velocity[:, i - 1, :]
        
        displacement = np.cumsum(self.step_size * velocity, axis=1)
        position = (start.T + displacement.T).T
        if self.modulo:
            position = position % (2 * np.pi)
        return start, velocity, position


    def generate_linear(self, batch_size, num_rotations=1):
        
        start = None
        if self.start_pos is None:   # randomize start positions
            start = np.random.uniform(0, 2 * np.pi, (batch_size, 1)).astype(np.float32)
        else:
            start = self.start_pos + np.zeros((batch_size, 1)).astype(np.float32)

        speed = num_rotations * 2 * np.pi / (self.step_size * self.sequence_length)
        if np.random.random() < .5:
            speed *= -1
        velocity = speed = np.ones((batch_size, self.sequence_length, 1), dtype=np.float32)
        
        displacement = np.cumsum(self.step_size * velocity, axis=1)
        position = (start.T + displacement.T).T
        if self.modulo:
            position = position % (2 * np.pi)
        return start, velocity, position

