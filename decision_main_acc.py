import time

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from decision_list import DecisionList
from algorithms import toy_example
from branch_predictor import BranchPredictor, TruePerceptron
import plotter

def worst_case(N):
    node_idx = np.arange(0, N)
    node_values = np.ones((N,))
    N_prime = N // 2 + 1
    output_values = np.array([1, 0] * N_prime)[:(N+1)]

    return DecisionList(N, N, node_idx, node_values, output_values)


data_length = 10000
n_val = 8

bp = BranchPredictor(n_val, TruePerceptron)
dl = DecisionList(n_val, n_val)
input = np.random.randint(0, 2, (data_length, n_val))

toy_example(input, dl, bp)

plotter.generate_plot(bp, "branch", 100)
