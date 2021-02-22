from os.path import isfile
import numpy as np
import os

models = 'models'
test_model = 'test/testModel.txt'
sizeofvect = 32
modelHeight = 5
modelWidth = 3
values = {1: '1', -1: '0'}
use = {'1': 1, '0': -1}


class NS_Hopfield:
    def __init__(self, height, width):
        self.models = []
        self.n = height * width
        self.height = height
        self.width = width
        self.w = np.array([np.zeros(self.n) for i in range(self.n)])

    def training_mode(self, x):
        self.models.append(x)

        for i in range(self.n):
            for j in range(self.n):
                if (i == j):
                    self.w[i][j] = 0
                else:
                    self.w[i][j] += x[i] * x[j]

    def FuncActiv(self, net, y):
        if net > 0:
            return 1
        elif net == 0:
            return 0
        else:
            return -1

    def net(self, x):
        for i in range(self.n):
            net_y = sum([self.w[j][i] * x[j] for j in range(self.n - 1)]) + sum(
                [self.w[j][i] * x[j] for j in range(self.n)])
            y = self.FuncActiv(net_y, x[i])
            if y != x[i] and y != 0:
                print("Изменение нейрона №%d с %d на %d" % (i, x[i], y))
                x[i] = y
        if x not in self.models:
            print("Распознования не произошло!")
            return 0
        return x

    models = 'models'
    test_model = 'test/testModel.txt'

    def parse(self, directory):
        shapes_files = []
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            if isfile(path):
                shapes_files.append(path)
        shapes = []
        for path in shapes_files:
            shape = self.parseModel(path)
            shapes.append(shape)
        return shapes

    def parseModel(self, path):
        with open(path) as f:
            model = f.read(sizeofvect)
            model = model.replace("\n", "")
            model = model.replace("\r", "")
            models = []
            for c in model:
                models.append(use[c])

            return models

    def get_Etalon(self, obraz, heigth, widht):
        for_print = "".join([values[a] for a in obraz])
        for i in range(heigth):
            print(for_print[i * widht: i * widht + widht])
        print('')


N = NS_Hopfield(modelHeight, modelWidth)

# 2 4 8
if __name__ == '__main__':
    etalons = N.parse(models)
    test_etalon = N.parseModel(test_model)

    print("Эталоны:")
    i = 1
    for e in etalons:
        print("Эталон №%d:" % i)
        N.get_Etalon(e, modelHeight, modelWidth)
        i = i + 1
    for e in etalons:
        N.training_mode(e)
    print("Веса РНС Хопфилда в векторно-матричном виде:\n W = ", N.w)
    print("\n")
    print("Распознаваемая модель:")
    N.get_Etalon(test_etalon, modelHeight, modelWidth)
    new_model = N.net(test_etalon)
    if new_model == 0:
        exit(0)
    print("\n")
    print("Распознование прошло успешно!")
    N.get_Etalon(new_model, modelHeight, modelWidth)