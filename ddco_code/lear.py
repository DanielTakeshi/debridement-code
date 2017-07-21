import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

data = []
f = open('data.p','r')

while True:
    try:
        d = pickle.load(f)
        data.append(d)
    except EOFError:
        break

X = np.zeros((len(data),2))
Y = np.zeros((len(data),2))

for i,d in enumerate(data):
    Y[i,0] = np.ravel(d['pos'][0])[0]
    Y[i,1] = np.ravel(d['pos'][1])[0]
    X[i,0] = d['estimate'][0]
    X[i,1] = d['estimate'][1]

print(X, Y)

regx = RandomForestRegressor(n_estimators=3)
regx.fit(X,Y[:,0])
regy = RandomForestRegressor()
regy.fit(X,Y[:,1])
#Yp = reg.predict(X)
pickle.dump((regx,regy), open('model-sep.p','wb'))

import matplotlib.pyplot as plt
plt.scatter(Y[:,0], Y[:,1], c='r')
print(X, Y)
plt.scatter(regx.predict(X), regy.predict(X),c='b', marker='x')
plt.show()
