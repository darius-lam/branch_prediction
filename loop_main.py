import random
import collections

import numpy as np
import matplotlib.pyplot as plt

from algorithms import loop
from branch_predictor import BranchPredictor, TruePerceptron
import plotter


iterations = 512
N_max = 256
N_vals = range(8, N_max, 4)

data = []
data_no_warmup = []
data_random = []

for N in N_vals:
    bp = BranchPredictor(N, TruePerceptron)
    loop(N-2, bp)
    for _ in range(iterations):
        loop(N-2, bp, "branch")

    wrong = bp.total["branch"] - bp.correct["branch"]
    data.append(wrong)


    bp = BranchPredictor(N, TruePerceptron)
    for _ in range(iterations):
        loop(N-2, bp, "branch")

    wrong = bp.total["branch"] - bp.correct["branch"]
    data_no_warmup.append(wrong)


    bp = BranchPredictor(N, TruePerceptron)
    bp.perceptrons["branch"].weights = np.random.randn(N + 1) * np.sqrt(2 / (N + 1))
    for _ in range(iterations):
        loop(N-2, bp, "branch")

    wrong = bp.total["branch"] - bp.correct["branch"]
    data_random.append(wrong)



plt.figure()
plt.plot(list(N_vals), data, '.-', label="with warmup")
plt.plot(list(N_vals), data_no_warmup, '.-', label="w/o warmup")
plt.plot(list(N_vals), data_random, '.-', label="random")
plt.title("Mistakes vs. Dimension for Fixed-Length Loops")
plt.xlabel("Length")
plt.ylabel("Mistakes")
plt.legend()
plt.show()
