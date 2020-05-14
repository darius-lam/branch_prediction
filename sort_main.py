import random
import collections

import matplotlib.pyplot as plt

from algorithms import bubble_sort
from branch_predictor import BranchPredictor, TruePerceptron
import plotter


history_length = 64
bp = BranchPredictor(history_length, TruePerceptron)

data = list(range(64))

# for _ in range(history_length):
#     random.shuffle(data)
#     result, result_length = select(data, 0, 100, 16, bp)


iters = []
for _ in range(3):
    random.shuffle(data)
    bubble_sort(data, bp)
    iters.append(len(bp.prediction_history["flip"]))

bp.print_accuracies()

plotter.generate_plot_sort(bp, "flip", iters, 200)
