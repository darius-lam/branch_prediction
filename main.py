import random
import collections

import matplotlib.pyplot as plt

from algorithms import bubble_sort
from branch_predictor import BranchPredictor

history_lengths = list(range(2, 26, 2))
accuracies = collections.defaultdict(list)
movings_accs = collections.defaultdict(list)

input = list(range(10))


for i in history_lengths:

    bp = BranchPredictor(i)
    for _ in range(200):
        random.shuffle(input)
        bubble_sort(input, bp)

    accs, movings = bp.get_accuracies()
    for key in accs:
        accuracies[key].append(accs[key])
    for key in movings:
        movings_accs[key].append(movings[key])

plt.figure()

plt.subplot(1,2,1)
for key in accuracies:
    plt.plot(history_lengths, accuracies[key], label=key)
plt.legend()
plt.title("Overall Accuracy")

plt.subplot(1,2,2)
for key in movings_accs:
    plt.plot(history_lengths, movings_accs[key], label=key)
plt.legend()
plt.title("Moving Average of Accuracy")

plt.show()
