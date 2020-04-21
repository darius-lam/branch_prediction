from collections import deque
import torch
from torch.autograd import Variable
import functools
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

n = 5
global_branch_history = deque([0] * n)
global_losses = []
global_preds = []

#A helper function that appends 'value' to 'queue' and
#subsequently pops left, leaving n intact.
#This function modifies queue in place.
def push(queue, value):
	queue.append(value)
	queue.popleft()

def t():
	push(global_branch_history,1)

def nt():
	push(global_branch_history,0)

class Perceptron(nn.Module):

    def __init__(self,n):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(n,2)

    def forward(self, x):
        x = self.fc1(x)
        #x = F.softmax(x)
        return x

def perceptron_decorator(net, optimizer, criterion):

	def wrap(func):
		def wrapper(*args, **kwargs):
			#print([i.data for i in net.parameters()])

			optimizer.zero_grad()

			value = func(*args, **kwargs)
			pred = net(torch.tensor(global_branch_history,dtype=torch.float))

			#Calculate binary cross entropy for this example
			loss = criterion(pred.view(1,-1),
				torch.tensor(value,dtype=torch.long).view(-1))
			global_losses.append(loss.item())
			global_preds.append((pred.detach().numpy(), value))

			#Backpropagate
			loss.backward()
			optimizer.step()

			#Add the next value to global history
			if value == 1:
				t()
			else:
				nt()

		return wrapper

	return wrap

#Set up the perceptron
net = Perceptron(n)
optimizer = optim.Adam(net.parameters(),lr=1e-3)
loss = nn.CrossEntropyLoss()

def f1():
	for i in range(10):
		if i % 3 == 0:
			t()
			for j in range(3):
				if j == 0:
					t()
				else:
					nt()
		else:
			nt()

@perceptron_decorator(net, optimizer, loss)
def f2(x,y,i):
	#populate global history register
	for j in range(n):
		if x[i][j]:
			t()
		else:
			nt()

	#Add the output of the function
	if y[i]:
		return 1
	else:
		return 0

number_of_iterations = 500

#Setup training examples
x = []
y = []
for i in range(n):
	x.append([0] * (i) + [1] + [0] * (n-i))
	y.append(1)
	x.append([1] * (i) + [0] + [1] * (n-i))
	y.append(0)

m = len(x)

##In this case, we expect the perceptron to learn perfectly
#Because it will "memorize" the examples
"""
for _ in tqdm(range(number_of_iterations)):
	for i in range(m):
		f2(x,y,i)
		#print(global_branch_history)"""

#Now, we randomize the selection of examples at each iteration
for _ in tqdm(range(number_of_iterations)):
	for i in range(m):
		i = np.random.choice(np.arange(m))
		f2(x,y,i)

### Display training results

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#print([(np.argmax(i[0]), i[1]) for i in global_preds])
accuracy = np.sum([np.argmax(i[0]) == i[1] for i in global_preds])/len(global_preds)
print("Accuracy: %f" % accuracy)
plt.plot(moving_average(global_losses,n=50))
plt.show()