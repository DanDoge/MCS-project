import os
import numpy as np

class FaceDataset(object):
    def __init__(self, path="./", ratio=0.9):
        self.imgpath = os.path.join(path, "face_image")
        self.labelpath = os.path.join(path, "face_label")
        self.Xtrain, self.Ytrain, self.Xtest, self.Ytest = self.getdata(ratio)
        
    def getdata(self, ratio):
        import pickle
        with open(self.imgpath, "rb") as f:
            X = np.array(pickle.load(f)).astype(np.float32)
        X = X / 255 - 0.5
        with open(self.labelpath, "rb") as f:
            Y = np.array(pickle.load(f))
            
        datalen = X.shape[0]
        return X[: int(datalen * ratio)], Y[: int(datalen * ratio)], X[int(datalen * ratio): ], Y[int(datalen * ratio): ]