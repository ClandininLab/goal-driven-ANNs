import numpy as np
import matplotlib.pyplot as plt

def plot_average_activations(net, generator, num_trajectories=500, precision=100):
    precision = 100
    num_trajectories = 500

    start, velocity, position = generator.generate_trajectory(num_trajectories)
    history = net.g(velocity, start)

    position = (precision * position / (2 * np.pi)).astype(int)
    averages = {i:{p:[] for p in range(precision)} for i in range(net.hidden_dim)}
    for i in range(net.hidden_dim):
        for j in range(num_trajectories):
            for t in range(net.sequence_length):
                p = position[j, t, 0]
                g = history[j, t, i]
                averages[i][p].append(g)
                
    for i in range(net.hidden_dim):
        for p in range(precision):
            if averages[i][p] == []:
                averages[i][p] = 0
            else:
                averages[i][p] = np.mean(averages[i][p])

    for i in range(net.hidden_dim):
        plt.bar(averages[i].keys(), averages[i].values())
        plt.ylim(bottom=0, top=1)
        plt.show()
        plt.clf()

