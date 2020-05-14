import random
import collections

import matplotlib.pyplot as plt

from algorithms import full_random, malicious_random
from branch_predictor import BranchPredictor, BpLambdaPerceptron
import plotter
import time


history_length = 128
bp = BranchPredictor(history_length, BpLambdaPerceptron, .01)
#bp = BranchPredictor(history_length)

adversarial = False
s = time.time()
for _ in range(400):
    result = malicious_random(0.4, 10, bp, adversarial)
print("Took %f seconds..." % (time.time()-s))
bp.print_accuracies()

#plotter.generate_plot(bp, "full_random")
plotter.generate_plot(bp, "adversarial_noise" if adversarial else "random_noise")
