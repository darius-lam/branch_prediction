import random
import collections

import matplotlib.pyplot as plt

from algorithms import loop
from branch_predictor import BranchPredictor, TruePerceptron
import plotter


iterations = 512
history_length = 64
N_max = 128
N_vals = range(8, N_max, 4)

data = []
data_no_warmup = []

for N in N_vals:
    bp = BranchPredictor(history_length, TruePerceptron)
    for _ in range(iterations):
        loop(N, bp, "branch")

    wrong = bp.total["branch"] - bp.correct["branch"]
    data.append(bp.get_accuracies()[0]["branch"])

plt.figure()
plt.plot(list(N_vals), [1 - 1 / x for x in N_vals], label="static")
plt.plot(list(N_vals), data, '-', label="perceptron")
plt.axvline(history_length, label="history register length", color='k')
plt.title("Accuracy vs. Dimension for Fixed-Length Loops")
plt.xlabel("Loop Length")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
