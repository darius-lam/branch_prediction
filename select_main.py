import random
import collections

import matplotlib.pyplot as plt

from algorithms import select
from branch_predictor import BranchPredictor


history_length = 128
bp = BranchPredictor(history_length)

data = list(range(200))


for _ in range(200):
    random.shuffle(data)
    result, result_length = select(data, 0, 100, 16, bp)

bp.print_accuracies()
