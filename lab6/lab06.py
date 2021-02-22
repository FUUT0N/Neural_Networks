import matplotlib.pyplot as plt
import requests
import json
import pandas as pd
import numpy as np

# Округа Москвы
COLORS = ['y', 'b', 'r', 'g', 'c', 'm', 'lime', 'gold', 'orange', 'coral', 'purple', 'grey']

DISTRICT = {"Восточный административный округ": [55.787710, 37.775631],
            "Западный административный округ": [55.728003, 37.443533],
            "Зеленоградский административный округ": [55.987583, 37.194250],
            "Новомосковский административный округ": [55.558121, 37.370724],
            "Северный административный округ": [55.838384, 37.525765],
            "Северо-Восточный административный округ": [55.863894, 37.620923],
            "Северо-Западный административный округ": [55.829370, 37.451546],
            "Троицкий административный округ": [55.355771, 37.146990],
            "Центральный административный округ": [55.753995, 37.614069],
            "Юго-Восточный административный округ": [55.692019, 37.754583],
            "Юго-Западный административный округ": [55.662735, 37.576178],
            "Южный административный округ": [55.610906, 37.681479]}

# Название округа
DISTRICT_NAME = ['ВАО', 'ЗАО', 'ЗелАО', 'Новомосковский АО', 'САО', 'СВАО', 'СЗАО', 'Троицкий АО', 'ЦАО', 'ЮВАО', 'ЮЗАО', 'ЮАО']

# POST запрос данных
def get_data(url, filename):
    URL = url
    client = requests.session()
    client.get(URL)
    res = requests.post(URL, headers=dict(Referer=URL))

    with open(filename, 'w') as outfile:
        json.dump(res.json(), outfile, ensure_ascii=False, separators=(',', ': '), indent=4, sort_keys=False)


class network_KH:
    def __init__(self, values, centers):
        self.values = np.array(values)
        self.centers = np.array(centers)
        self.weights = np.zeros((len(values), len(centers)))

    def euclidDist(self, a, b):
        return np.linalg.norm(a - b)

    def find_weights(self):
        for value_i in range(len(self.values)):
            for center_i in range(len(self.centers)):
                self.weights[value_i][center_i] = self.euclidDist(self.values[value_i], self.centers[center_i])
        for value_i in range(len(self.values)):
            min_index = self.weights[value_i].argmin()
            self.weights[value_i][min_index] = 1
            self.weights[value_i][0:min_index] = 0
            self.weights[value_i][min_index + 1:] = 0

        return self.weights


class ClusterAnalysis():
    def __init__(self, data_lyceums):
        self.read_json(data_lyceums)
        _, self.ax = plt.subplots()
        self.save_data('init.png')

# Считывание JSON
    def read_json(self, data_lyceums):
        json_data = open(data_lyceums).read()
        data = json.loads(json_data)
        lyceums_data = [data['features'][i]['geometry']['coordinates'] for i in
                         range(len(data['features']))]
        dist_data = [data['features'][i]['properties']['Attributes']['okrug'] for i in
                     range(len(data['features']))]

        name_data = [data['features'][i]['properties']['Attributes']['name'] for i in
                     range(len(data['features']))]

        lyceums = pd.DataFrame(lyceums_data, columns=['x', 'y'])
        lyceums['districts'] = dist_data
        lyceums['color'] = 'k'
        lyceums['size'] = 6

        lyceums['name'] = name_data

        self.lyceums = lyceums

        districts_data = DISTRICT.values()
        districts = pd.DataFrame(districts_data, columns=['y', 'x'])
        districts['color'] = COLORS
        districts['size'] = 26
        self.districts = districts

    def save_data(self, filename):
        self.ax.scatter(x=self.lyceums['x'], y=self.lyceums['y'],
                        s=self.lyceums['size'],
                        c=self.lyceums['color'])

        for i in range(len(COLORS)):
            self.ax.scatter(x=self.districts['x'][i],
                            y=self.districts['y'][i],
                            s=self.districts['size'][i],
                            marker='s',
                            c=COLORS[i],
                            label=DISTRICT_NAME[i])

        self.ax.legend(fontsize=7, loc='lower right')
        self.ax.axis('off')
        plt.savefig(filename)

# Кластеризация
    def clustering(self):
        network = network_KH(self.lyceums[['x', 'y']].values, self.districts[['x', 'y']].values)
        weights = network.find_weights()

        lyceums_data = self.lyceums.values
        districts_data = self.districts.values

        error_list = list()
        for i in range(len(lyceums_data)):
            center = weights[i].argmax()
            lyceums_data[i][3] = COLORS[center]
            if lyceums_data[i][2] != list(DISTRICT.keys())[center]:
                lyceums_data[i][3] = 'k'
                error_list.append(
                    f' {lyceums_data[i][5]} Должен принадлежать {lyceums_data[i][2]}, а принадлежит {list(DISTRICT.keys())[center]}')

        print(len(error_list))
        self.lyceums['color'] = lyceums_data.T[3]
        self.districts['color'] = districts_data.T[3]

        self.ax.clear()
        self.save_data('result.png')
        plt.show()


if __name__ == "__main__":
    get_data('https://apidata.mos.ru/v1/datasets/552/features', 'lyceums.json')

    analysis = ClusterAnalysis('lyceums.json')

    analysis.clustering()
