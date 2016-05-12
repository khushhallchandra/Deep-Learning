from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.advanced_activations import LeakyReLU
import numpy as np

model = Sequential()
model.add(Dense(output_dim=4, input_dim=2, init='uniform'))
model.add(LeakyReLU(alpha=0.01))

model.add(Dense(output_dim=1, input_dim=4, init='uniform'))
model.add(LeakyReLU(alpha=0.01))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=sgd)

X = np.zeros((4, 2), dtype='uint8')
y = np.zeros(4, dtype='uint8')

X[0] = [0, 0]
y[0] = 0
X[1] = [0, 1]
y[1] = 1
X[2] = [1, 0]
y[2] = 1
X[3] = [1, 1]
y[3] = 0

history = model.fit(X, y, nb_epoch=100000, batch_size=4, show_accuracy=True, verbose=0)
print model.predict(X)