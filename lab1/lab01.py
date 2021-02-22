#LAB_1 Variant 4

from math import *
import copy
import itertools
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# Булева функция на заданном наборе
def boolean_func(X):
    return (((not X[0]) or X[2]) and X[1]) or (X[1] and X[3])  # второй вариант

# Пороговая ФА
def porog_FA(net):
    return 1 if net >= 0 else 0, 1

# ФА с модулем
def module_FA(net):
    fnet = (1/2) * ((net / (1 + abs(net))) + 1)
    dfnet = (1/2) / ((abs(net) + 1) ** 2)
    return 1 if fnet >= 0.5 else 0, dfnet

# Инициализация необходимых компонентов, вывод таблицы истинности
def func(n):
    X = generation(n)
    F = get_F(X)
    t = PrettyTable(['X', 'F'])
    for x, f in zip(X, F):
        t.add_row([x, f])
    print(t)
    return F

# Возвращает значения БФ на заданных наборах
def generation(n):
    X = list()
    for i in range(0, 2 ** n):
        X.append(toByte(i, n))
    return X

# Функция перевода в двоичное число
def toByte(x, n):
    v = [0 for _ in range(n)]
    i = n - 1;
    while x > 0:
        v[i] = x % 2
        x = x // 2
        i = i - 1
    return v

# Функция построения графика
def plot(x_data, y_data, x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    ax.plot(x_data, y_data, lw=2, marker='o', color='#034bbf', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.grid()
    plt.show()

# Функция возвращает значения БФ на заданном наборе переменных
def get_F(X):
    F = list()
    for x in X:
        F.append(int(boolean_func(x)))
    return F


# Возвращает значение У реальный выход НС
def findY(X, W, func):
    Y = [0 for _ in range(len(X))]
    for j in range(len(X)):
        net = W[0]
        for i in range(0, len(X[j])):
            net += X[j][i] * W[i + 1]
        if func == '1':
            f, df = porog_FA(net)
        elif func == '2':
            f, df = module_FA(net)
        Y[j] = f
    return Y

# Расстояние Хемминга
def hamming(F, Y):
    return sum(f != y for f, y in zip(F, Y))

# Все расчёт обучения
def calculate(X, W, F, func, norma, normamberSets, x0=1):
    for l in range(0, normamberSets):
        net = W[0]  # считаем net
        for i in range(0, len(X[l])):
            net += X[l][i] * W[i + 1]
        if func == '1':
            f, df = porog_FA(net)
        elif func == '2':
            f, df = module_FA(net)
        d = F[l] - f  # F[l] = t правило Видроу - Хоффа
        W[0] += norma * d * df * x0  # новый вес
        for i in range(0, len(X[l])):
            W[i + 1] += norma * d * df * X[l][i]
    return W, findY(X, W, func)

# Функция обучения
def learn(F, norma, func, n):
    k = 0  # эпоха
    X = generation(n)
    W = [0 for _ in range(n + 1)]
    Y = findY(X, W, func)
    error = hamming(F, Y)  # квадратична ошибка
    t = PrettyTable(['K', 'W', 'Y', 'E'])
    t.add_row([k, copy.copy(W), copy.copy(Y), error])
    K = [k]
    E = [error]
    while error != 0:
        W, Y = calculate(X, W, F, func, norma, len(X))
        error = hamming(F, Y)
        E.append(error)
        k += 1
        K.append(k)
        t.add_row([k, copy.copy(W), copy.copy(Y), error])
    print(t)
    return K, W, Y, E

# Обучение на выборочных наборах
def select_learn(F, norma, func, n):
    k = 0  # эпоха
    X = generation(n)
    W = [0 for _ in range(n + 1)]
    normamberSets = 2  # начнём перебор с 2 наборов
    while True:
        for setX, setF in zip(itertools.combinations(X, normamberSets), itertools.combinations(F, normamberSets)):
            setY = [1 for _ in range(normamberSets)]
            e = hamming(setF, setY)  # квадратична ошибка
            t = PrettyTable(['K', 'W', 'Y', 'E'])
            t.add_row([k, copy.copy(W), copy.copy(setY), e])
            K = [k]
            E = [e]
            while e != 0 and k < 100:
                W, setY = calculate(setX, W, setF, func, norma, normamberSets)
                setY = findY(setX, W, func)
                e = hamming(setF, setY)
                E.append(e)
                k += 1
                K.append(k)
                t.add_row([k, copy.copy(W), copy.copy(setY), e])
            Y = findY(X, W, func)  # получаем Y по нашим весам и считаем ошибку
            if hamming(F, Y) == 0:
                print('Удалось обучить на', normamberSets, 'наборах')
                for i in range(normamberSets):
                    print('X' + str(i + 1), '=', setX[i], end=' ')
                print()
                print(t)
                return K, W, Y, E
            else:
                k = 0
                W = [0 for _ in range(n + 1)]
                K.clear()
                E.clear()
                t.clear()
        normamberSets += 1


if __name__ == "__main__":
    F = func(n=4)
    print('Задание 1')
    function = input('Введите ФА (1 = ФА пороговая, 2 = ФА c модулем):')
    norma = float(input('Введите норму обучения'))
    K, W, Y, E = learn(F, norma, function, n=4)
    plot(K, E, "Error E", "Era K", "E(k)")
    print('Задание 2')
    function = input('Введите ФА (1 = ФА пороговая, 2 =ФА  с модулем):')
    norma = float(input('Введите норму обучения'))
    K, W, Y, E = select_learn(F, norma, function, n=4)
    plot(K, E, "Error E", "Era K", "E(k)")
