import collections

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class BranchPredictor():
    def __init__(self, n):
        """
        This class is responsible for storing the table of perceptrons and
        keeping track of the global history register.
        """
        self.global_history = np.zeros((n,))
        self.total = collections.defaultdict(int)
        self.correct = collections.defaultdict(int)
        self.moving_accuracy = collections.defaultdict(float)
        self.perceptrons = collections.defaultdict(lambda : BpPerceptron(n))

    def __call__(self, condition, tag=None):
        """
        This is essentially the wrapper for the if statement or while condition

        If it is given a tag then it will make a prediction, if not it will
        just update the global history register
        """
        if tag:
            self.total[tag] += 1
            p = self.perceptrons[tag].predict_and_update(self.global_history,
                                                         int(condition))
            self.moving_accuracy[tag] *= 0.9
            if (condition == p):
                self.correct[tag] += 1
                self.moving_accuracy[tag] += 0.1

        self.update_history(condition)

        return condition

    def update_history(self, condition):
        """
        Appends the condition to the global history register
        """
        self.global_history = np.roll(self.global_history, 1)
        if condition:
            self.global_history[0] = 1
        else:
            self.global_history[0] = 0

    def print_accuracies(self):
        for key in self.perceptrons:
            acc = self.correct[key] / self.total[key]
            mv = self.moving_accuracy[key]
            print(f"Branch: {key} Accuracy {acc} Moving {mv}")

    def get_accuracies(self):
        acc = {}
        for key in self.perceptrons:
            acc[key] = self.correct[key] / self.total[key]
        return acc, self.moving_accuracy

class BpPerceptron():
    """
    Wrapper for perceptron. Essentially just has logic for training
    """
    def __init__(self, n):
        self.net = ClassicalPerceptron(n)
        self.optimizer = optim.SGD(self.net.parameters(), lr=1e-3)

    def predict(self, X):
        pred = self.net(torch.tensor(X, dtype=torch.float))

        return pred

    def predict_and_update(self, X, y):
        self.optimizer.zero_grad()

        pred = self.predict(X)
        loss = torch.abs(pred - y)

        #Backpropagate
        loss.backward()
        self.optimizer.step()

        return (pred > 0.5).numpy()


class ClassicalPerceptron(nn.Module):
    def __init__(self, n):
        super(ClassicalPerceptron, self).__init__()
        self.fc1 = nn.Linear(n,1)

    def forward(self, x):
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    bp = BranchPredictor(10)

    i = 0
    i_max = 1000
    while bp(i < i_max, "condition"):
        if bp(np.random.random((1)) > 0.5, "random"):
            pass

        i+= 1

    bp.print_accuracies()
