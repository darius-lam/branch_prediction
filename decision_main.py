import numpy as np

from decision_list import DecisionList
from algorithms import toy_example
from branch_predictor import BranchPredictor
import plotter


N = 5
data_length = 5000
dl = DecisionList(N, N)
bp = BranchPredictor(N)
input = np.random.randint(0, 2, (data_length, N))

toy_example(input, dl, bp)

bp.print_accuracies()


plotter.generate_plot(bp, "branch", 200)
