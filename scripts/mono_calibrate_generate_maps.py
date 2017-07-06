import environ

import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def loadData(filename):
    data = []
    f = open(filename,'r')
    while True:
        try:
            d = pickle.load(f)

            if not outlier(d):
                data.append(d)

        except EOFError:
            break

    return data



def outlier(d):
    arm1, arm2 = d
    pos1, _ , cX, cY = arm1
    pos2, _ , cX, cY = arm2

    if np.ravel(pos1[2])[0] > -0.135 or \
        np.ravel(pos2[2])[0] > -0.105:
        return True

    else:

        return False


def dataToMatrix(data):
    X = np.zeros((len(data),2))
    Y = np.zeros((len(data),4))
    
    for i, d in enumerate(data):
        arm1, arm2 = d
        pos1, _ , cX, cY = arm1
        pos2, _ , cX, cY = arm2

        Y[i,0] = np.ravel(pos1[0])[0]
        Y[i,1] = np.ravel(pos1[1])[0]
        Y[i,2] = np.ravel(pos2[0])[0]
        Y[i,3] = np.ravel(pos2[1])[0]

        X[i,0] = cX
        X[i,1] = cY

    return X,Y


def train(X,Y):
    reg = RandomForestRegressor(n_estimators=1)
    reg.fit(X,Y)
    return reg


data = loadData('config/left.p')
X,Y = dataToMatrix(data)
regl = train(X,Y[:,0:2])

data = loadData('config/left.p')
X,Y = dataToMatrix(data)
regr = train(X,Y[:,2:4])

pickle.dump( (regl, regr), open('mono-model.p','wb'))

