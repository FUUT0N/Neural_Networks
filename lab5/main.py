from funcs import *

vec_1 = [1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1]
vec_2 = [1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1]
vec_3 = [1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1]

def recover_RNN(vec_x, vec_target, matrix_w):
    vector_y = vec_x
    vector_prev = list()

    tmp = list()
    epoch = 0
    while vector_y != tmp:
        tmp = vec_target
        if epoch == 0:
            vector_prev = vector_y
        epoch += 1

        y_curr = list()
        for k in range(len(vec_x)):
            sum_1 = 0
            for j in range(k - 1):
                sum_1 += matrix_w[j][k] * vector_y[j]
            sum_2 = 0
            for f in range(k + 1, len(vec_x)):
                sum_2 += matrix_w[f][k] * vector_prev[f]
            net = sum_1 + sum_2

            y = func(vector_prev[k], net)
            y_curr.append(y)

        vector_prev = vector_y
        vector_y = y_curr

    return vector_y


if __name__ == "__main__":
    etalons = [vec_1, vec_2, vec_3]
    matrix = get_matrix(etalons)
    test = [1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1]
    print_vec(test)
    Y = recover_RNN(test, vec_3, matrix)
    print(Y)
