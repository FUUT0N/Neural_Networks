#LAB_2 Variant 4

import numpy as np
import matplotlib.pyplot as plt

# возвращает результат моделируемой функции от значения времени
def X(T):
   # return [np.sin(t) ** 2  for t in T]
    return [0.5 * np.exp(0.5 * np.cos(0.5 * t)) + np.sin(0.5 * t) for t in T]

N = 20 
a = -5
b = 3

T = list(np.linspace(a, 2 * b - a, 2 * N))
RightX = X(T)
TryX = [0 for i in range(N)]


def DeltaW(x, q, n):
    return n * q * x

# значение сетевого входа НС
def Net(x, w):
    return sum([w_i * x_i for w_i, x_i in zip(w, x)])

def MeanSquareError(RightX, TryX, p):
    summa = 0
    for rx_i, tx_i in zip(RightX[p:], TryX[p:]):
        summa += (rx_i - tx_i) ** 2
    return summa ** 0.5

# обучение НС методом скользящего окна
def Learning(p, n, m):

    for k in range(p): TryX[k] = RightX[k]
    w = [0] * p
    era = 0

    while(era < m):

        for l in range(p, N): # 16 шагов эпохи

            TryX[l] = Net(RightX[l - p:l-1], w)
            q = RightX[l] - TryX[l]
            for k in range(0, p):
                w[k] += DeltaW(RightX[l - p + k], q, n)

        era += 1

        # print("\nera = ",  np.round(era, 3))
        # print("TryX : ", np.around(TryX, 3))
        # print("w : ", np.around(w, 3))
        # print("e = ", np.round(e, 3))

    print(np.around(TryX, 3))
    return list(TryX), w

# график
def Graph(TryX, p, arg = "", name = ""):
    fig, ax = plt.subplots()
    ax.plot(T, RightX, 'bo-', linewidth=3, markersize=5)
    ax.plot(T, TryX, 'ro-', linewidth=2, markersize=3)
    plt.title("X(t)\n" + name + str(arg))
    plt.xlabel('t')
    plt.ylabel('X')
    plt.axvline(x=a,  linestyle='--')
    plt.axvline(x=b, linestyle='--')
    plt.axvline(x=T[p], linestyle='--', color = 'g')
    plt.grid(True)
    plt.show()

# строит график зависимости ошибки от n, p, m
def Graph_E(e, arg, name):
    plt.plot(arg, e, 'bo-', linewidth=2, markersize=5)
    plt.title("E(" + name + ")")
    plt.xlabel(name)
    plt.ylabel('E')
    plt.grid(True)
    plt.show()

def Forecast(E, n, p, m):
    TryX, w = Learning(p, n, m)
    TryX.extend(np.zeros(N))
    for l in range(N, 2 * N):
        TryX[l] = Net(RightX[l - p: l - 1], w)
    E.append(MeanSquareError(RightX[N:], TryX[N:], p))
    return TryX


if __name__=="__main__":

# норма обучения
    ny = 0.5
    range_n = np.around(np.linspace(0.1, 1, 10), 1)
# размер окна
    pOkno = 4
    range_p = range(1, 17)
# эпохи
    era = 1000 
    range_m = range(400, 5001, 100)

    E = []

    for p in range_p:
        print("\n\np = ", p)
        Forecast(E, ny, p, era)
    Graph_E(E, range_p, "p")

    E.clear()

    for n in range_n:
        print("\n\nn = ", n)
        Forecast(E, n, pOkno, era)
    Graph_E(E, range_n, "n")

    E.clear()

    for m in range_m:
        print("\n\nM = ", m)
        Forecast(E, ny, pOkno, m)
    Graph_E(E, range_m, "M")

    Graph(Forecast(E, ny, pOkno, 30000), pOkno)