import matplotlib.pyplot as plt
import numpy as np

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def generate_plot(bp, tag, sliding_window=100):
    taken_history = np.array(bp.taken_history[tag])
    prediction_history = np.array(bp.prediction_history[tag])

    static = sum(taken_history) / len(taken_history)
    static = 1 - static if static < 0.5 else static

    running_accuracy = running_mean(taken_history == prediction_history, sliding_window)

    plt.figure()
    plt.plot(running_accuracy)
    plt.axhline(0.5, label="Random Guess", color='m')
    plt.axhline(static, label="Static Prediction", color='g')
    plt.axhline(np.mean(running_accuracy), label="Average Perceptron Accuracy", color='b')

    plt.title(f"{sliding_window} Window Accuracy vs. Iteration for {tag}")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
