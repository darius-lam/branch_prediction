import random

def bubble_sort(input, bp):
    i = 1
    n = len(input)
    swapped = True

    while bp(swapped, "outer"):
        swapped = False
        i = 1
        while bp(i < n, "inner"):
            if bp(input[i-1] > input[i], "flip"):
                temp = input[i-1]
                input[i-1] = input[i]
                input[i] = temp
                swapped = True
            i += 1

def malicious_random(eta, n, bp, adversarial=True):
    """
    Eta: Float indicating probability of adversarial example
    n: Integer number of times to loop
    """

    if adversarial:
        name="adversarial_noise"
    else:
        name="random_noise"

    #With probability 1-eta, we return either the
    #opposite label (adversarial) or
    #a random label (random)
    for i in range(n):
        val = random.random()
        true_label = (i%3) == 0 #Should be  0,0,0,1,0,0,0,1 ...

        if val < (1-eta):
            if not adversarial:
                retval = (1-true_label) if random.random() < 0.5 else true_label
            else:
                retval = 1-true_label

            bp(retval,name) 
        else:
            bp(true_label, name)
         


def full_random(eta, bp):
    if bp(random.random() < 1-eta, "full_random"):
        return True

    return False

def select(input, low, high, batch_size, bp):
    result = []
    result_length = 0
    num_batches = len(input) // batch_size

    i = 0
    while bp(i < num_batches, "outer"):
        if bp(result_length + batch_size > len(result), "expand"):
            result.extend([0] * batch_size)

        j = 0
        while bp(j < batch_size, "inner"):
            idx = i * batch_size + j
            if bp(input[idx] >= low and input[idx] < high, "qualify"):
                result[result_length] = input[idx]
                result_length += 1
            j += 1

        i += 1

    return result, result_length


def select_no_tags(input, low, high, batch_size, bp):
    result = []
    result_length = 0
    num_batches = len(input) // batch_size

    i = 0
    while bp(i < num_batches):
        if bp(result_length + batch_size > len(result)):
            result.extend([0] * batch_size)

        j = 0
        while bp(j < batch_size):
            idx = i * batch_size + j
            if bp(input[idx] >= low and input[idx] < high):
                result[result_length] = input[idx]
                result_length += 1
            j += 1

        i += 1

    return result, result_length


def toy_example(input, conditional, bp):
    """
    input should be a 2D list
    """
    for i in range(input.shape[0]):
        bp.global_history = input[i]
        bp(conditional(input[i]), "branch")
