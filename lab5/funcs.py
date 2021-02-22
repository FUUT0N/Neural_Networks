import numpy

def func(y_prev, net):
    if net > 0:
        return 1
    elif net == 0:
        return y_prev
    else:
        return -1


def get_matrix(patterns):
    matrix = numpy.zeros((len(patterns[0]), len(patterns[0])))
    for i in range(len(patterns[0])):
        for j in range(len(patterns[0])):
            matrix[i][j] = find_weight(patterns, i, j)
    return matrix

def find_weight(patterns, i, j):
    result = 0
    if i != j:
        for s in patterns:
            result += s[i] * s[j]
    return result

def visual_print(s, newline):
    if s == 1:
        if newline == 'n':
            print("%3s" % s, end='')
        else:
            print("%3s" % s)
    else:
        if newline == 'n':
            print("%3s" % ' ', end='')
        else:
            print("%3s" % ' ')

def print_vec(pattern):
    for i in range(5):
        index = i
        for j in range(2):
            visual_print(pattern[index], 'n')
            index += 5
        visual_print(pattern[index], 'y')
