import os
import numpy as np

class FaceDataset(object):
    def __init__(self, ratio=0.9):
        self.imgpath = os.path.join(os.path.dirname(__file__), "face_image")
        self.labelpath = os.path.join(os.path.dirname(__file__), "face_label")
        self.Xtrain, self.Ytrain, self.Xtest, self.Ytest = self.getdata(ratio)

    def getdata(self, ratio):
        import pickle
        with open(self.imgpath, "rb") as f:
            X = np.array(pickle.load(f)).astype(np.float32)
        X = X / 255 - 0.5
        with open(self.labelpath, "rb") as f:
            Y = np.array(pickle.load(f))
        Y = np.reshape(Y, (-1, 1))

        datalen = X.shape[0]
        X_per = np.random.permutation(X)
        Y_per = np.random.permutation(Y)
        return X_per[: int(datalen * ratio)], Y_per[: int(datalen * ratio)], X_per[int(datalen * ratio): ], Y_per[int(datalen * ratio): ]
