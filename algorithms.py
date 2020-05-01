

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
