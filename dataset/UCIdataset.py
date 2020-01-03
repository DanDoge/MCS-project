import pandas
import numpy as np
import os

dataset_name_list = ["bost", "conc", "ener", "kin8", "nava", "powe", "prot", "wine", "yach"]
dataset_not_implemented = ["bost", "kin8", "wine"]
class UCIDataset(object):
    def __init__(self, name, ratio=0.5):
        if name not in dataset_name_list:
            print("Error: dataset not found!")
            print("Availble datasets are...")
            for dataset in dataset_name_list:
                print(dataset)
            return
        if name in dataset_not_implemented:
            print("Error: dataset not implemented!")
            return
        if ratio >= 1. or ratio <= 0.:
            print("train/test split ratio should be in (0, 1), 0.5 by default")
        self.Xtrain, self.Ytrain, self.Xtest, self.Ytest = self.getdata(name, ratio)

    def getdata(self, name, ratio):
        if name == "conc":
            data = np.array(pandas.read_excel(os.path.join(os.path.dirname(__file__), "Concrete_Data.xls")))
            feature_split = 8


        if name == "ener":
            data = np.array(pandas.read_excel(os.path.join(os.path.dirname(__file__), "ENB2012_data.xlsx")))
            feature_split = 8

        if name == "nava":
            nava_data = []
            with open(os.path.join(os.path.dirname(__file__), "data.txt"), "r") as f:
                for line in f:
                    nava_data.append([float(num) for num in line.split()])
            data = np.array(nava_data)
            feature_split = 16

        if name == "powe":
            data = np.array(pandas.read_excel(os.path.join(os.path.dirname(__file__), "Folds5x2_pp.xlsx")))
            feature_split = 4

        if name == "prot":
            data = np.genfromtxt(os.path.join(os.path.dirname(__file__), "CASP.csv"), delimiter=",")
            data[:, [0, 9]] = data[:, [9, 0]]
            data = data[1:, ]
            feature_split = 9

        if name == "yach":
            yach_data = []
            with open(os.path.join(os.path.dirname(__file__), "yacht_hydrodynamics.data"), "r") as f:
                for line in f:
                    datum = [float(num) for num in line.split()]
                    if len(datum) == 7:
                        yach_data.append(datum)
            data = np.array(yach_data)
            feature_split = 6

        datalen = data.shape[0]
        data_per = np.random.permutation(data)
        return data_per[ :int(datalen * ratio), :feature_split], data_per[ :int(datalen * ratio), feature_split:], data_per[int(datalen * ratio): , :feature_split], data_per[int(datalen * ratio): , feature_split:]

if __name__ == '__main__':
    for ds in dataset_name_list:
        print(ds)
        data = UCIDataset(ds, 0.9)
        try:
            print(data.Xtrain.shape, data.Ytrain.shape, data.Xtest.shape, data.Ytest.shape)
            print(data.Ytrain[0])
        except:
            pass
