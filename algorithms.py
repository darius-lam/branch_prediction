

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
