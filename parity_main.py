import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from decision_list import ParityFunction
from algorithms import toy_example
from branch_predictor import BranchPredictor, TruePerceptron
import plotter


N = 8
data_length = 1000

bp = BranchPredictor(N, TruePerceptron)
parity = ParityFunction(N)

input = np.random.randint(0, 2, (data_length, N))

toy_example(input, parity, bp)


bp.print_accuracies()

plotter.generate_plot(bp, "branch", 20)
