import random
import collections

import matplotlib.pyplot as plt

from algorithms import select
from branch_predictor import BranchPredictor


history_length = 64
bp = BranchPredictor(history_length)

data = list(range(100))


for _ in range(500):
    random.shuffle(data)
    result, result_length = select(data, 25, 50, 10, bp)

bp.print_accuracies()
