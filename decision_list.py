import numpy as np

class DecisionList:
    def __init__(self, N, length, node_idx=None, node_values=None,
                 output_values=None):
        """

        """
        self.N = N
        self.length = length

        if not node_idx:
            self.node_idx = np.random.randint(0, 2, (self.length))
        else:
            self.node_idx = node_idx

        if not node_values:
            self.node_values = np.random.randint(0, 2, (self.length))
        else:
            self.node_values = node_values

        if not node_values:
            self.output_values = np.random.randint(0, 2, (self.length+1))
        else:
            self.output_values = output_values

    def __call__(self, x):
        """
        Evaluates the decision list given the literal values x
        """
        for i in range(self.length):
            idx = self.node_idx[i]
            if (self.node_values[idx] == x[idx]):
                return self.output_values[idx]

        return self.output_values[self.length]
