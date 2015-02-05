#from load_data import load_gray
from load_data import load_color
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from preproc import ZCA
import numpy as np
import matplotlib.pyplot as plt

#train, valid, test = load_gray()
train, valid, test = load_color()
train_x, train_y = train
valid_x, valid_y = valid

mean = np.mean(train_x, axis=0)
std = np.std(train_x, axis=0)

train_x -= mean
valid_x -= mean
train_x /= std
valid_x /= std

tf = ZCA()
tf.fit(train_x)

train_x = tf.transform(train_x)
valid_x = tf.transform(valid_x)

clf = RandomForestClassifier(n_estimators=128)
clf.fit(train_x, train_y)
train_y_hat = clf.predict(train_x)
valid_y_hat = clf.predict(valid_x)
print(accuracy_score(train_y, train_y_hat))
print(accuracy_score(valid_y, valid_y_hat))
