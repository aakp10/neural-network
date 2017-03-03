from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import numpy as np
#f=file("/home/aakp/Downloads/pima-indians-diabetes.csv")
#dataset=np.load(f) 
#f.close()
dataset=np.loadtxt("/home/aakp/Downloads/pima-indians-diabetes.csv",delimiter=",")
data=dataset[:,0:8]
labels=dataset[:,8]

model=Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(data, labels, nb_epoch=150, batch_size=10,  verbose=2)
scores = model.evaluate(data, labels)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))