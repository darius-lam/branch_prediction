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

data_length = 100000
iterations = 10
N_max = 9
N_vals = range(2, N_max)

wc_data = []
random_data = []

for N in N_vals:
    temp = []
    for _ in range(iterations):
        bp = BranchPredictor(N, TruePerceptron)
        dl = worst_case(N)
        input = np.random.randint(0, 2, (data_length, N))

        toy_example(input, dl, bp)

        wrong = bp.total["branch"] - bp.correct["branch"]
        temp.append(wrong)

    wc_data.append(np.median(temp))

    temp = []
    for _ in range(iterations):
        bp = BranchPredictor(N, TruePerceptron)
        dl = DecisionList(N, N)
        input = np.random.randint(0, 2, (data_length, N))

        toy_example(input, dl, bp)

        wrong = bp.total["branch"] - bp.correct["branch"]
        temp.append(wrong)

    random_data.append(np.median(temp))


plt.figure()
plt.plot(list(N_vals), wc_data, label="worst case")
plt.plot(list(N_vals), random_data, label="Random")
plt.title("Mistakes vs. Dimension for Decision Lists")
plt.xlabel("Dimension")
plt.ylabel("Mistakes")
plt.legend()
plt.show()
